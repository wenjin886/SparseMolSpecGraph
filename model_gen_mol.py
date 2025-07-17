import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.utils import to_dense_batch

from transformers import (PreTrainedTokenizerFast, GenerationMixin,
                          GenerationConfig, PretrainedConfig,
                          BeamSearchScorer)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import math
import copy
import rdkit.Chem as Chem

from model import PeakGraphModule

def pad_str_ids(str_ids_list, str_len, pad_token=0):
    """
    将一个list中的str_ids补齐成tensor batch，同时返回padding mask
    """
    max_length = max(str_len)
    
    padded_str_ids = []
    padding_masks = []
    for ids in str_ids_list:
        pad_len = max_length - len(ids)
        padded_ids = ids + [pad_token] * pad_len
        padded_str_ids.append(padded_ids)
        
        # 创建padding mask: 1表示真实token，0表示padding token
        mask = [1] * len(ids) + [0] * pad_len
        padding_masks.append(mask)

    return torch.tensor(padded_str_ids, dtype=torch.long), torch.tensor(padding_masks, dtype=torch.bool)

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)
    
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    




def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e4)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class  LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
 
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist

        return self.criterion(x, true_dist.clone().detach())

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)  
    

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)
    
class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)
    
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class NMRFormulaEncoder(nn.Module):
    def __init__(self, 
                 formula_vocab_size,
                 d_model=512, d_ff=2048, 
                 encoder_head=8, encoder_layer=4,
                 dropout=0.1, 
                 ):
        super().__init__()

        self.formula_vocab_size = formula_vocab_size
        
        c = copy.deepcopy
        pe = PositionalEncoding(d_model, dropout=dropout)
        self.formula_embed = nn.Sequential(Embeddings(d_model=d_model, vocab=self.formula_vocab_size), c(pe))

        encoder_attn = MultiHeadedAttention(h=encoder_head,d_model=d_model)
        ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.spec_formula_encoder = Encoder(EncoderLayer(d_model, c(encoder_attn),c(ff), dropout), N=encoder_layer)

        self.__init_weights__()
    
    def __init_weights__(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
    
    def forward(self, spec_hidden_states, spec_attention_mask, formula_ids, formula_att_mask):
        formula_embed = self.formula_embed(formula_ids)
        src = torch.cat([spec_hidden_states, formula_embed], dim=1) # (batch_size N_spec + N_formula, d_model)

        if len(formula_att_mask.shape) == 2:
            formula_att_mask = formula_att_mask.unsqueeze(1)
        src_mask = torch.cat([spec_attention_mask, formula_att_mask], dim=-1) # (batch_size, 1, N_spec + N_formula)

        src_embed = self.spec_formula_encoder(src, src_mask)

        return src_embed, src_mask
        
        
        

class NMR2MolDecoder(nn.Module, GenerationMixin):
    def __init__(self, 
                 smiles_vocab_size,
                 d_model=512, d_ff=2048, 
                 decoder_head=8, N_decoder_layer=4, dropout=0.1, 
                 ):
        super().__init__()
        c = copy.deepcopy
        
        self.smiles_vocab_size = smiles_vocab_size
        pe = PositionalEncoding(d_model, dropout=0.1)
        self.smiles_embed = nn.Sequential(Embeddings(d_model=d_model, vocab=self.smiles_vocab_size), c(pe))
        
        multi_att = MultiHeadedAttention(h=decoder_head, d_model=d_model, dropout=dropout)
        ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.decoder = Decoder(DecoderLayer(d_model, c(multi_att), c(multi_att), c(ff), dropout), N_decoder_layer)

        self.lm_head = nn.Linear(d_model, self.smiles_vocab_size)

        # Add generation_config for GenerationMixin
        self.generation_config = GenerationConfig()
        
        # Add config attribute for GenerationMixin
        self.config = PretrainedConfig()
        self.config.is_encoder_decoder = False
        self.config.vocab_size = smiles_vocab_size
        self.config.pad_token_id = 0
        self.config.bos_token_id = 1
        self.config.eos_token_id = 2
        
        # Add base_model_prefix for GenerationMixin
        self.base_model_prefix = "model"
        # Add main_input_name for GenerationMixin
        self.main_input_name = "input_ids"
        # Add _supports_cache_class for GenerationMixin compatibility
        self._supports_cache_class = False

        self.__init_weights__()
    
    def __init_weights__(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)
    
    def _subsequent_mask(self, size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
            torch.uint8
        )
        return subsequent_mask == 0
    
    def _get_tgt_mask(self, smiles_ids, smiles_att_mask, for_generation=False):
        # for_generation: True if generating, False if training/validation
        if for_generation or smiles_ids.shape[1] == 1:
            self.tgt = smiles_ids
            self.tgt_y = smiles_ids
            tgt_mask = torch.ones(self.tgt.shape).bool().unsqueeze(-2).type_as(self.tgt.data)
            tgt_mask = tgt_mask & self._subsequent_mask(self.tgt.shape[-1]).type_as(self.tgt.data)
        else:
            self.tgt = smiles_ids[:, :-1]
            self.tgt_y = smiles_ids[:, 1:]
            tgt_mask = smiles_att_mask[:, :-1].bool().unsqueeze(-2)
            tgt_mask = tgt_mask & self._subsequent_mask(self.tgt.shape[-1]).type_as(tgt_mask.data)
        self.ntokens = (self.tgt_y != 0).data.sum()
        return tgt_mask

    # def decode(self, smiles_ids, smiles_att_mask, src, src_mask):
    #     tgt_mask = self._get_tgt_mask(smiles_ids, smiles_att_mask)
    #     smiles_embed = self.smiles_embed(self.tgt)

    #     # for beam search
    #     if (src.shape[0] < smiles_embed.shape[0]) and (smiles_embed.shape[0] % src.shape[0] == 0):
    #         n_beams = int(smiles_embed.shape[0] / src.shape[0])
    #         src = src.repeat_interleave(n_beams, dim=0)
    #         src_mask = src_mask.repeat_interleave(n_beams,dim=0)

    #     return self.decoder(x=smiles_embed,
    #                         memory=src,
    #                         src_mask=src_mask, 
    #                         tgt_mask=tgt_mask)
    
    
    def forward(self, input_ids, encoder_hidden_states, encoder_attention_mask, 
                use_cache=False, past_key_values=None, **kwargs):
        """
        input_ids: (B, T)
        encoder_hidden_states: src
        encoder_attention_mask: src_mask
        """
        smiles_att_mask = torch.where(input_ids == 0, False, True).float()
        for_generation = (use_cache or (past_key_values is not None)) or (input_ids.shape[1] == 1)
        tgt_mask = self._get_tgt_mask(input_ids, smiles_att_mask, for_generation=for_generation)
        tgt_embed = self.smiles_embed(self.tgt)

        decoder_outputs = self.decoder(
            x=tgt_embed,
            memory=encoder_hidden_states,
            src_mask=encoder_attention_mask, 
            tgt_mask=tgt_mask
        )
        logits = self.lm_head(decoder_outputs)
        
        return CausalLMOutputWithCrossAttentions(logits=logits)
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, 
                                      encoder_hidden_states=None, 
                                      encoder_attention_mask=None, **kwargs):
        return {
            "input_ids": input_ids,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_attention_mask": encoder_attention_mask,
            "past_key_values": past_key_values
        }
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def get_decoder(self):
        return self.decoder
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    
    
    
    

class NMR2MolGenerator(pl.LightningModule):
    def __init__(self, mult_class_num, nH_class_num, 
                 smiles_tokenizer_path,
                 # NMR graph encoder
                 mult_embed_dim=16, nH_embed_dim=8, c_w_embed_dim=8,
                 edge_dim=1,
                 num_layers=4, num_heads=4, graph_dropout=0.1,
                 mult_class_weights=None,
                 # formula
                 use_formula=True,
                 formula_vocab_size=None,
                 spec_formula_encoder_head=8, 
                 spec_formula_encoder_layer=4,
                 # SMILES decoder
                 d_model=512, d_ff=2048, 
                 decoder_head=8, N_decoder_layer=4, dropout=0.1, 
                 # optimizer
                 warm_up_step=None, lr=None):
        super().__init__()
        self.save_hyperparameters()

        self.smiles_tokenizer = PreTrainedTokenizerFast(tokenizer_file=smiles_tokenizer_path,
                                                        bos_token="[BOS]",
                                                        eos_token="[EOS]",
                                                        pad_token="[PAD]",
                                                        unk_token="[UNK]",
                                                        padding_side="right" )
        smiles_vocab_size = len(self.smiles_tokenizer.get_vocab())
        

        self.graph_encoder = PeakGraphModule(mult_class_num=mult_class_num, nH_class_num=nH_class_num, 
                                mult_embed_dim=mult_embed_dim, nH_embed_dim=nH_embed_dim, c_w_embed_dim=c_w_embed_dim,
                                num_layers=num_layers, num_heads=num_heads, edge_dim=edge_dim, dropout=graph_dropout)

        spec_embed_dim = self.graph_encoder.in_node_dim
        # self.spec_embed_proj = nn.Linear(spec_embed_dim, d_model)
        
        self.use_formula = use_formula
        if self.use_formula:
            self.spec_formula_encoder = NMRFormulaEncoder(formula_vocab_size,
                                                     d_model=d_model, d_ff=d_ff, 
                                                     encoder_head=spec_formula_encoder_head, 
                                                     encoder_layer=spec_formula_encoder_layer,
                                                     dropout=dropout)
            
        self.smiles_decoder = NMR2MolDecoder(smiles_vocab_size,
                                             formula_vocab_size,
                                             d_model=d_model, d_ff=d_ff, 
                                             decoder_head=decoder_head, 
                                             N_decoder_layer=N_decoder_layer, 
                                             dropout=dropout)
        
        self.d_model = d_model
        self.warm_up_step = warm_up_step
        self.lr = lr
        self.criterion = LabelSmoothing(size=smiles_vocab_size, padding_idx=0, smoothing=0.1)

    def _get_spec_embed(self, batch, node_embeddings):
        # 使用to_dense_batch将稀疏的node_embeddings转换为密集的batch格式
        # src shape: (batch_size, max_nodes, embedding_dim)
        # src_mask shape: (batch_size, 1, max_nodes)
        
        spec_embed, spec_mask = to_dense_batch(node_embeddings, batch)
        # src_mask需要调整维度以匹配transformer的期望格式，并转换为浮点类型
        spec_mask = spec_mask.unsqueeze(1).float()  # (batch_size, 1, max_nodes)
        
        return spec_embed, spec_mask
    
    def encode(self, data, padding_formula_ids=None, padding_formula_att_mask=None):
        node_embeddings  = self.graph_encoder.encode(data)
        node_embeddings = node_embeddings.float()  # Ensure float dtype
        src_embed, src_mask = self._get_spec_embed(data.batch, node_embeddings) # spec_embed: (batch_size, max_nodes, in_node_dim)
        # src_embed = self.spec_embed_proj(src_embed) # (batch_size, max_nodes, d_model)

        if self.use_formula:
            src_embed, src_mask = self.spec_formula_encoder(src_embed, src_mask, padding_formula_ids, padding_formula_att_mask)

        return src_embed, src_mask
    
    def forward(self, data, padding_smiles_ids, padding_smiles_masks, padding_formula_ids=None, padding_formula_att_mask=None):
        
        src_embed, src_mask = self.encode(data, padding_formula_ids, padding_formula_att_mask)

        if padding_smiles_masks.dtype == torch.bool:
            padding_smiles_masks = padding_smiles_masks.float()
        decoder_output = self.smiles_decoder( input_ids=padding_smiles_ids,  encoder_hidden_states=src_embed, encoder_attention_mask=src_mask)
        # Extract logits from the decoder output
        if hasattr(decoder_output, 'logits'):
            logits = decoder_output.logits
        else:
            logits = decoder_output
        return logits
    
    def _cal_loss(self, logits, tgt, norm):
        """
        SimpleLossCompute
        logits: output of self.lm_head
        tgt
        norm: ntokens
        """
        # preds = torch.argmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        sloss = (
            self.criterion(
                log_probs.contiguous().view(-1, log_probs.size(-1)), tgt.contiguous().view(-1)
            )
            / norm
        )
        preds = torch.argmax(log_probs, dim=-1)
        return sloss, preds
    
    def _cal_token_acc(self, logits, tgt):
        mask = tgt != 0
        token_preds = torch.argmax(logits, dim=-1)
        correct = ((token_preds == tgt) & mask).sum().item()
        total = mask.sum().item()
        token_acc = correct / total if total > 0 else 0.0
        return token_acc
    
    def _postprocess_smiles(self, decoded_str):
        if not isinstance(decoded_str, str):
            decoded_str = str(decoded_str)
        if '[BOS]' in decoded_str:
            split_list = decoded_str.split('[BOS]')
            if len(split_list) == 0: return None
            else: decoded_str = split_list[-1]
        if '[EOS]' in decoded_str:
            split_list = decoded_str.split('[EOS]')
            if len(split_list) == 0: return None
            else: decoded_str = split_list[0]
        return ''.join(decoded_str.split(' '))
    
    def _cal_mol_acc(self, preds, tgts):
        preds = self.smiles_tokenizer.batch_decode(preds.tolist())
        tgts = self.smiles_tokenizer.batch_decode(tgts.tolist())

        correct_pred = 0
        total = len(preds)  # 所有生成的数量
        for pred, tgt in zip(preds, tgts):
            
            pred = self._postprocess_smiles(pred)
            # print(pred)
            if pred is None: continue
            try:
                mol = Chem.MolFromSmiles(pred)
                pred = Chem.MolToSmiles(mol)
            except Exception as e:
                continue
            
            tgt = self._postprocess_smiles(tgt)
            try:
                tgt = Chem.MolToSmiles(Chem.MolFromSmiles(tgt))
            except Exception as e:
                print(f"tgt has problems: {tgt}")
                continue
                # return -1
            if pred == tgt: correct_pred += 1

        # print(correct_pred/len(preds)) 
        return correct_pred/len(preds) if total > 0 else 0.0
    
    def _step(self, batch, batch_idx):
        device = batch.id.device

        padding_smiles_ids, padding_smiles_masks = pad_str_ids(batch.smiles_ids, batch.smiles_len)
        if self.use_formula:
            padding_formula_ids, padding_formula_masks = pad_str_ids(batch.formula_ids, batch.formula_len)
            padding_formula_ids, padding_formula_masks = padding_formula_ids.to(device), padding_formula_masks.to(device)
        else:
            padding_formula_ids, padding_formula_masks = None, None
        logits = self(batch, 
                      padding_smiles_ids.to(device), 
                      padding_smiles_masks.to(device),
                      padding_formula_ids,
                      padding_formula_masks
                    )
        return logits

    def training_step(self, batch, batch_idx):
        logits = self._step(batch, batch_idx)

        smiles_pred_loss, _ = self._cal_loss(logits, self.smiles_decoder.tgt_y, norm=self.smiles_decoder.ntokens)

        batch_size = len(batch.id)
        self.log("train_loss", smiles_pred_loss, batch_size=batch_size)
        return smiles_pred_loss
    
    def validation_step(self, batch, batch_idx):
        logits = self._step(batch, batch_idx)

        smiles_pred_loss, preds = self._cal_loss(logits, self.smiles_decoder.tgt_y, norm=self.smiles_decoder.ntokens)
        token_acc = self._cal_token_acc(logits, self.smiles_decoder.tgt_y)
        acc = self._cal_mol_acc(preds, self.smiles_decoder.tgt_y)

        batch_size = len(batch.id)
        # 提取目标：batch.masked_node_target 是 dict of tensors
        self.log("val_loss", smiles_pred_loss, batch_size=batch_size)
        self.log("val_token_acc", token_acc, batch_size=batch_size)
        self.log("val_acc", acc, batch_size=batch_size)
        return smiles_pred_loss
    
    def test_step(self, batch, batch_idx):
        logits = self._step(batch, batch_idx)

        smiles_pred_loss, preds = self._cal_loss(logits, self.smiles_decoder.tgt_y, norm=self.smiles_decoder.ntokens)
        token_acc = self._cal_token_acc(logits, self.smiles_decoder.tgt_y)
        acc = self._cal_mol_acc(preds, self.smiles_decoder.tgt_y)

        batch_size = len(batch.id)
        # 提取目标：batch.masked_node_target 是 dict of tensors
        self.log("test_loss", smiles_pred_loss, batch_size=batch_size)
        self.log("test_token_acc", token_acc, batch_size=batch_size)
        self.log("test_acc", acc, batch_size=batch_size)
        return smiles_pred_loss 
    
    def configure_optimizers(self):
        # # 使用 AdamW 优化器
        # optimizer = torch.optim.AdamW(
        #     self.parameters(),
        #     lr=self.lr,
        #     amsgrad=True,
        #     weight_decay=1e-12
        # )
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(0.9, 0.998),
            eps=1e-9
        )

        def rate(step):
           
            if step == 0:
                step = 1
            if self.warm_up_step > 0:
                value = self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warm_up_step ** (-1.5))
            else:
                value = self.d_model ** (-0.5) * step ** (-0.5)
            lr_scale = 1 * value

            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=rate
        )

        # 返回优化器和调度器
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
   
        
        
        