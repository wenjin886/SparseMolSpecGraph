import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
import copy
from transformers import PreTrainedTokenizerFast
import rdkit.Chem as Chem
from model import PeakGraphModule

def pad_smiles_ids(smiles_ids_list, smiles_len, pad_token=0):
    """
    将一个list中的smiles_ids补齐成tensor batch，同时返回padding mask
    """
    max_length = max(smiles_len)
    
    padded_smiles_ids = []
    padding_masks = []
    for ids in smiles_ids_list:
        pad_len = max_length - len(ids)
        padded_ids = ids + [pad_token] * pad_len
        padded_smiles_ids.append(padded_ids)
        
        # 创建padding mask: 1表示真实token，0表示padding token
        mask = [1] * len(ids) + [0] * pad_len
        padding_masks.append(mask)

    return torch.tensor(padded_smiles_ids, dtype=torch.long), torch.tensor(padding_masks, dtype=torch.bool)

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
        # print(x.size(1), self.size)
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        # print(x)
        # print(target.data)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        # print(x)
        # print(true_dist)
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
    
class NMR2MolDecoder(nn.Module):
    def __init__(self, 
                 smiles_vocab_size,
                 d_model=512, d_ff=2048, 
                 decoder_head=8, N_decoder_layer=4, dropout=0.1, 
                 ):
        super().__init__()
        c = copy.deepcopy
        
        self.smiles_vocab_size = smiles_vocab_size
        
        pe = PositionalEncoding(d_model, dropout=0.1)
        self.smiles_embed = nn.Sequential(Embeddings(d_model=d_model, vocab=self.smiles_vocab_size),
                                          c(pe))

        multi_att = MultiHeadedAttention(h=decoder_head, d_model=d_model,
                                           dropout=dropout)
        ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.1)
        self.decoder = Decoder(DecoderLayer(d_model, c(multi_att), c(multi_att), c(ff), dropout), N_decoder_layer)

        self.lm_head = nn.Linear(d_model, self.smiles_vocab_size)

        self.criterion = LabelSmoothing(size=self.smiles_vocab_size,padding_idx=0,smoothing=0.1)

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
    
    def _get_tgt_mask(self, smiles_ids, smiles_att_mask):
        """
        tgt: 
            smiles tokens, shape: B x len
        """
        # print("smiles_ids: ",smiles_ids.shape)

        if smiles_att_mask is None:
            # for generation
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
    
    def decode(self, smiles_ids, smiles_att_mask, src, src_mask):
        tgt_mask = self._get_tgt_mask(smiles_ids, smiles_att_mask)
        smiles_embed = self.smiles_embed(self.tgt)

        # for beam search
        if (src.shape[0] < smiles_embed.shape[0]) and (smiles_embed.shape[0] % src.shape[0] == 0):
            n_beams = int(smiles_embed.shape[0] / src.shape[0])
            src = src.repeat_interleave(n_beams, dim=0)
            src_mask = src_mask.repeat_interleave(n_beams,dim=0)

        return self.decoder(smiles_embed,
                                   memory=src,
                                   src_mask=src_mask, 
                                   tgt_mask=tgt_mask,
                                   )
    
    def forward(self, smiles_ids, smiles_att_mask, src, src_mask):
        
        dec_out = self.decode(smiles_ids, smiles_att_mask, src, src_mask)
        
        logits = F.log_softmax(self.lm_head(dec_out), dim=-1)

        return logits
    
    
    
    
    

class NMR2MolGenerator_(pl.LightningModule):
    def __init__(self, mult_class_num, nH_class_num, 
                 smiles_tokenizer_path,
                 # NMR graph encoder
                 mult_embed_dim=16, nH_embed_dim=8, c_w_embed_dim=8,
                 num_layers=4, num_heads=4,
                 mult_class_weights=None,
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
                                num_layers=num_layers, num_heads=num_heads)
        self.smiles_decoder = NMR2MolDecoder(smiles_vocab_size,
                                             d_model=d_model, d_ff=d_ff, 
                                             decoder_head=decoder_head, 
                                             N_decoder_layer=N_decoder_layer, 
                                             dropout=dropout)
        
        self.d_model = d_model
        self.warm_up_step = warm_up_step
        self.lr = lr

    def _get_src(self, batch, node_embeddings):
        # 使用to_dense_batch将稀疏的node_embeddings转换为密集的batch格式
        # src shape: (batch_size, max_nodes, embedding_dim)
        # src_mask shape: (batch_size, 1, max_nodes)
        from torch_geometric.utils import to_dense_batch
        
        print(node_embeddings.shape)
        src, src_mask = to_dense_batch(node_embeddings, batch)
        print(src.shape)
        
        # 验证梯度流 - 如果node_embeddings需要梯度，src也应该需要梯度
        if node_embeddings.requires_grad:
            print(f"node_embeddings requires_grad: {node_embeddings.requires_grad}")
            print(f"src requires_grad: {src.requires_grad}")
            print(f"src.grad_fn: {src.grad_fn}")
        
        # src_mask需要调整维度以匹配transformer的期望格式
        src_mask = src_mask.unsqueeze(1)  # (batch_size, 1, max_nodes)
        
        return src, src_mask
    
    def forward(self, data, padding_smiles_ids, padding_smiles_masks):
        node_embeddings  = self.graph_encoder.encode(data)
        src, src_mask = self._get_src(data.batch, node_embeddings)
        logits = self.smiles_decoder(padding_smiles_ids, padding_smiles_masks, src, src_mask)
        return logits
    
    def _cal_loss(self, logits, tgt, norm):
        """
        SimpleLossCompute
        logits: output of self.lm_head
        tgt
        norm: ntokens
        """
        preds = torch.argmax(logits, dim=-1)
        sloss = (
            self.criterion(
                logits.contiguous().view(-1, logits.size(-1)), tgt.contiguous().view(-1)
            )
            / norm
        )
        return sloss, preds
    
    
    def _postprocess_smiles(self, decoded_str):
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
                return -1
            if pred == tgt: correct_pred += 1

        # print(correct_pred/len(preds)) 
        return correct_pred/len(preds)
    
    def training_step(self, batch, batch_idx):
        
        batch_size = len(batch.id)
        padding_smiles_ids, padding_smiles_masks = pad_smiles_ids(batch.smiles_ids, batch.smiles_len)
        logits = self(batch, padding_smiles_ids, padding_smiles_masks)

        smiles_pred_loss, _ = self._cal_loss(logits, self.tgt_y, norm=self.ntokens)

        self.log("train_loss", smiles_pred_loss, batch_size=batch_size)
        return smiles_pred_loss
    
    def validation_step(self, batch, batch_idx):
        batch_size = len(batch.id)
        padding_smiles_ids, padding_smiles_masks = pad_smiles_ids(batch.smiles_ids, batch.smiles_len)
        logits = self(batch, padding_smiles_ids, padding_smiles_masks)

        smiles_pred_loss, preds = self._cal_loss(logits, self.tgt_y, norm=self.ntokens)
        acc = self._cal_mol_acc(preds, self.tgt_y)

        # 提取目标：batch.masked_node_target 是 dict of tensors
        self.log("val_loss", smiles_pred_loss, batch_size=batch_size)
        self.log("val_acc", acc, batch_size=batch_size)
        return smiles_pred_loss
    
    def test_step(self, batch, batch_idx):
        batch_size = len(batch.id)
        padding_smiles_ids, padding_smiles_masks = pad_smiles_ids(batch.smiles_ids, batch.smiles_len)
        logits = self(batch, padding_smiles_ids, padding_smiles_masks)

        smiles_pred_loss, preds = self._cal_loss(logits, self.tgt_y, norm=self.ntokens)
        acc = self._cal_mol_acc(preds, self.tgt_y)

        # 提取目标：batch.masked_node_target 是 dict of tensors
        self.log("val_loss", smiles_pred_loss, batch_size=batch_size)
        self.log("val_acc", acc, batch_size=batch_size)
        return smiles_pred_loss

    
    
    def configure_optimizers(self):
        # 使用 AdamW 优化器
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            amsgrad=True,
            weight_decay=1e-12
        )

        def rate(step):
           
            if step == 0:
                step = 1
            lr_scale = 1 * (
                self.d_model ** (-0.5) * min(step ** (-0.5), step * self.warm_up_step ** (-1.5))
            )

            return lr_scale

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=rate
        )

        # 返回优化器和调度器
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
    
    
        
        
        