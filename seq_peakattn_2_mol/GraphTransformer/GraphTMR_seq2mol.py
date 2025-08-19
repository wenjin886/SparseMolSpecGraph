import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.utils import to_dense_batch

from transformers import (PreTrainedTokenizerFast, GenerationMixin,
                          GenerationConfig, PretrainedConfig,
                          BeamSearchScorer)
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

import copy
import rdkit.Chem as Chem

from tmr_utils import (BertEmbeddings, BertLayer,
                       BertSelfAttention,
                       relative_position_bucket)

from tf_utils import (Embeddings, PositionalEncoding, MultiHeadedAttention, 
                      PositionwiseFeedForward, LabelSmoothing, 
                      Encoder, Decoder, EncoderLayer, DecoderLayer)

class GraphAggregation(BertSelfAttention):
    def __init__(self, hidden_size, num_attention_heads, output_attentions, attention_probs_dropout_prob):
        super(GraphAggregation, self).__init__(hidden_size, num_attention_heads, output_attentions, attention_probs_dropout_prob)
        self.output_attentions = False

    def forward(self, hidden_states, attention_mask=None, rel_pos=None):
        # query = self.query(hidden_states[:, :1])  # B 1 D
        query = self.query(hidden_states)  # (B, N_peaks, D)
        
        key = self.key(hidden_states)
        value = self.value(hidden_states)
        station_embed = self.multi_head_attention(query=query,
                                                  key=key,
                                                  value=value,
                                                  attention_mask=attention_mask,
                                                  rel_pos=rel_pos)[0]  # B 1 D
        # station_embed = station_embed.squeeze(1)

        return station_embed

class GraphBertEncoder(nn.Module):
    def __init__(self, output_attentions, output_hidden_states, 
                 num_hidden_layers, hidden_size, num_attention_heads, 
                 attention_probs_dropout_prob,
                 intermediate_size, layer_norm_eps, hidden_dropout_prob
                 ):
        super().__init__()

        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        c = copy.deepcopy
        bert_layer = BertLayer(hidden_size, num_attention_heads, output_attentions, attention_probs_dropout_prob,
                               intermediate_size, layer_norm_eps, hidden_dropout_prob)
        self.layer = nn.ModuleList([c(bert_layer) for _ in range(num_hidden_layers)])

        self.graph_attention = GraphAggregation(hidden_size, num_attention_heads, output_attentions, attention_probs_dropout_prob)

    def forward(self,
                hidden_states,
                attention_mask,
                node_mask=None,
                node_rel_pos=None,
                rel_pos=None):
        """
        hidden_states: (B*N_peaks, 1+L_peak, D)
        attention_mask: (B*N_peaks, 1+L_peak)
        node_mask: (B, N_peaks)
        
        """
        peak_attention_mask = attention_mask.unsqueeze(1) # (B*N_peaks, 1, 1+L_peak)

        all_hidden_states = ()
        all_attentions = ()

        all_nodes_num, seq_length, emb_dim = hidden_states.shape # seq_length=1+L_peak
        # batch_size, _, _, subgraph_node_num = node_mask.shape
        # print("node_mask", node_mask.shape)
        batch_size,  subgraph_node_num = node_mask.shape
        # print(node_mask)

        

        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if i > 0:

                hidden_states = hidden_states.view(batch_size, subgraph_node_num, seq_length, emb_dim)  # B SN L D
                cls_emb = hidden_states[:, :, 1].clone()  # B SN D (i.e. (B, N_peaks, D))（每个peak的第一个token是[CLS], 在tokenzier中已经加好了; 每个节点序列自己的汇总向量，用来表征该节点的“局部/自身”信息）
                # print("cls_emb", cls_emb.shape)
                station_emb = self.graph_attention(hidden_states=cls_emb, attention_mask=node_mask.unsqueeze(1),
                                                   rel_pos=node_rel_pos)  # B D (station（位置0）: 额外插入的“聚合”槽位，不代表某个具体token；它用邻接掩码和相对位置从子图邻居里聚合信息，然后把聚合结果写回到每个子图的“主节点”的位置0，作为后续层的上下文注入)

                # update the station in the query/key
                # print("updated hidden shape", hidden_states[:, :, 0, :].shape, "station_emb", station_emb.shape)
                # hidden_states[:, 0, 0] = station_emb
                hidden_states[:, :, 0, :] = station_emb
                hidden_states = hidden_states.view(all_nodes_num, seq_length, emb_dim)

                layer_outputs = layer_module(hidden_states, attention_mask=peak_attention_mask, rel_pos=rel_pos)

            else:
                # temp_attention_mask = attention_mask.clone()
                # temp_attention_mask[::subgraph_node_num, :, :, 0] = -10000.0
                # layer_outputs = layer_module(hidden_states, attention_mask=temp_attention_mask, rel_pos=rel_pos)
                # print("hidden_states", hidden_states.shape)
                # print("peak_attention_mask", peak_attention_mask.shape)
                # print(peak_attention_mask)
                layer_outputs = layer_module(hidden_states, attention_mask=peak_attention_mask, rel_pos=rel_pos)
                

            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class GraphFormers(nn.Module):
    def __init__(self, vocab_size, rel_pos_bins, max_rel_pos,
                 hidden_size=512, fix_word_embedding=False, max_position_embeddings=5000, type_vocab_size=0, 
                 layer_norm_eps=1e-6, hidden_dropout_prob=0.1,
                 output_attentions=False, output_hidden_states=False,  
                 num_attention_heads=8, num_hidden_layers=4,
                 attention_probs_dropout_prob=0.1,
                 intermediate_size=2048,
                 ):
        """
        output_attentions, output_hidden_states: True when debug; False otherwise to save memory
        """
        super(GraphFormers, self).__init__()
        self.embeddings = BertEmbeddings(vocab_size, hidden_size, 
                                         fix_word_embedding,
                                         max_position_embeddings, type_vocab_size, 
                                         layer_norm_eps, hidden_dropout_prob)
        self.encoder = GraphBertEncoder(output_attentions, output_hidden_states, 
                                        num_hidden_layers, hidden_size, num_attention_heads, 
                                        attention_probs_dropout_prob,
                                        intermediate_size, layer_norm_eps, hidden_dropout_prob
                                        )
        self.rel_pos_bins = rel_pos_bins
        self.max_rel_pos = max_rel_pos
        if self.rel_pos_bins > 0:
            self.rel_pos_bias = nn.Linear(self.rel_pos_bins + 2,
                                          num_attention_heads,
                                          bias=False)
        else:
            self.rel_pos_bias = None

    def forward(self,
                input_ids,
                attention_mask
                ):
                # neighbor_mask=None):
        """
        input_ids: (B, N_peaks, L_peak) --> all_nodes_num: B*N_peaks, subgraph_node_num:N_peaks, seq_length: L_peak
        attention_mask: (B, N_peaks, L_peak)
        """
        # all_nodes_num, seq_length = input_ids.shape
        # batch_size, subgraph_node_num = neighbor_mask.shape

        batch_size, subgraph_node_num, seq_length = input_ids.shape
        # print("input_ids", input_ids.shape)
        # print(input_ids)
        all_nodes_num = batch_size * subgraph_node_num
        is_all_zero_row = torch.all(input_ids == 0, dim=-1)
        node_mask = (~is_all_zero_row).int() # (B, N_peaks)

        input_ids = input_ids.reshape(all_nodes_num, seq_length)
        embedding_output, position_ids = self.embeddings(input_ids=input_ids)
        # print("attention_mask", attention_mask.shape)
        # print(attention_mask)
        attention_mask = attention_mask.reshape(all_nodes_num, seq_length)

        # Add station attention mask
        # station_mask = torch.zeros((all_nodes_num, 1), dtype=attention_mask.dtype, device=attention_mask.device) # 0: mask, 1: no mask
        station_mask = torch.ones((all_nodes_num, 1), dtype=attention_mask.dtype, device=attention_mask.device) # 0: mask, 1: no mask
        attention_mask = torch.cat([station_mask, attention_mask], dim=-1)  # N 1+L (i.e. ((B*N_peaks, L_peak+1)))
        
        # attention_mask[::(subgraph_node_num), 0] = 1.0  # only use the station for main nodes

        # node_mask = (1.0 - neighbor_mask[:, None, None, :]) * -10000.0
        # extended_attention_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        

        if self.rel_pos_bins > 0:
            rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
            rel_pos = relative_position_bucket(rel_pos_mat, num_buckets=self.rel_pos_bins,
                                               max_distance=self.max_rel_pos)

            # rel_pos: (N,L,L) -> (N,1+L,L)
            temp_pos = torch.zeros(all_nodes_num, 1, seq_length, dtype=rel_pos.dtype, device=rel_pos.device)
            rel_pos = torch.cat([temp_pos, rel_pos], dim=1)
            # rel_pos: (N,1+L,L) -> (N,1+L,1+L)
            station_relpos = torch.full((all_nodes_num, seq_length + 1, 1), self.rel_pos_bins,
                                        dtype=rel_pos.dtype, device=rel_pos.device)
            rel_pos = torch.cat([station_relpos, rel_pos], dim=-1)

            # node_rel_pos:(B:batch_size, Head_num, neighbor_num+1)
            node_pos = self.rel_pos_bins + 1
            node_rel_pos = torch.full((batch_size, subgraph_node_num), node_pos, dtype=rel_pos.dtype,
                                      device=rel_pos.device)
            node_rel_pos[:, 0] = 0
            node_rel_pos = F.one_hot(node_rel_pos,
                                     num_classes=self.rel_pos_bins + 2).type_as(
                embedding_output)
            node_rel_pos = self.rel_pos_bias(node_rel_pos).permute(0, 2, 1)  # B head_num, neighbor_num
            node_rel_pos = node_rel_pos.unsqueeze(2)  # B head_num 1 neighbor_num

            # rel_pos: (N,Head_num,1+L,1+L)
            rel_pos = F.one_hot(rel_pos, num_classes=self.rel_pos_bins + 2).type_as(
                embedding_output)
            rel_pos = self.rel_pos_bias(rel_pos).permute(0, 3, 1, 2)

        else:
            node_rel_pos = None
            rel_pos = None

        # Add station_placeholder
        station_placeholder = torch.zeros(all_nodes_num, 1, embedding_output.size(-1)).type(
            embedding_output.dtype).to(embedding_output.device)
        embedding_output = torch.cat([station_placeholder, embedding_output], dim=1)  # N 1+L D

        encoder_outputs = self.encoder(
            embedding_output,
            # attention_mask=extended_attention_mask,
            attention_mask=attention_mask,
            node_mask=node_mask,
            node_rel_pos=node_rel_pos,
            rel_pos=rel_pos)

        return encoder_outputs, node_mask


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
        self.is_encoder_decoder = False
        self.vocab_size = smiles_vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        
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


    def forward(self, input_ids, tgt_attn_mask, encoder_hidden_states, encoder_attention_mask, 
                use_cache=False, past_key_values=None, **kwargs):
        """
        input_ids: (B, T) smiles ids
        encoder_hidden_states: src
        encoder_attention_mask: src_mask
        """
        # smiles_att_mask = torch.where(input_ids == 0, False, True).float()
        for_generation = (use_cache or (past_key_values is not None)) or (input_ids.shape[1] == 1)
        tgt_mask = self._get_tgt_mask(input_ids, tgt_attn_mask, for_generation=for_generation)
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
  
class NMRSeq2MolGenerator(pl.LightningModule):
    def __init__(self, 
                 smiles_tokenizer_path,
                 formula_vocab_size,
                 nmr_vocab_size,
                 rel_pos_bins=32,
                 max_rel_pos=32,
                 spec_encoder_layer=4,
                 spec_encoder_head=8,
                 spec_formula_encoder_head=8, 
                 spec_formula_encoder_layer=4,
                 peak_attn_encoder_layer=4,
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
        
        self.spec_encoder = GraphFormers(vocab_size=nmr_vocab_size,
                                         rel_pos_bins=rel_pos_bins, 
                                         max_rel_pos=max_rel_pos,
                                         hidden_size=d_model, 
                                         max_position_embeddings=5000, 
                                         type_vocab_size=0, 
                                         layer_norm_eps=1e-12, 
                                         hidden_dropout_prob=dropout,
                                         output_attentions=False, 
                                         output_hidden_states=False, 
                                         num_hidden_layers=spec_encoder_layer,
                                         num_attention_heads=spec_encoder_head, 
                                         attention_probs_dropout_prob=dropout)

        # formula
        c = copy.deepcopy
        multi_att = MultiHeadedAttention(h=decoder_head, d_model=d_model, dropout=dropout)
        ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.formula_embeddings = Embeddings(d_model=d_model, vocab=formula_vocab_size)       
        self.formula_spec_encoder = Encoder(layer=EncoderLayer(d_model, c(multi_att), c(ff), dropout), N=N_decoder_layer)
       
        # tgt
        self.smiles_decoder = NMR2MolDecoder(smiles_vocab_size,
                                             d_model=d_model, d_ff=d_ff, 
                                             decoder_head=decoder_head, 
                                             N_decoder_layer=N_decoder_layer, 
                                             dropout=dropout)
        
        self.d_model = d_model
        self.warm_up_step = warm_up_step
        self.lr = lr
        self.criterion = LabelSmoothing(size=smiles_vocab_size, padding_idx=0, smoothing=0.1)

    
    
    def forward(self, formula_ids, formula_mask, peaks_ids, peaks_mask, tgt_ids, tgt_mask):
        """
        formula_ids: (B, L_formula)
        formula_mask: (B, L_formula)
        peaks_ids: (B, N_peaks, L_peak)
        peaks_ids: (B, N_peaks, L_peak)
        tgt_ids: (B, L_smiles)
        tgt_mask: (B, L_smiles)
        """
        B, N_peaks, L_peak = peaks_ids.shape
        enc_out, node_mask = self.spec_encoder(peaks_ids, peaks_mask) # (B*N_peaks, 1+L, D); (B, N_peaks)
        spec_embed = enc_out[0]
        spec_node_embed = spec_embed[:, 1, :].view(B, N_peaks, -1) # [CLS]; (B*N_peaks, 1, D) --> (B, N_peaks, D)

        formula_embed = self.formula_embeddings(formula_ids)

        # print("formula", formula_embed.shape, formula_mask.shape)
        # print("spec", spec_node_embed.shape, node_mask.shape)
        src_embed = torch.cat([formula_embed, spec_node_embed], dim=1) # (B, N_formula+N_peaks, D)
        src_mask = torch.cat([formula_mask, node_mask], dim=-1).unsqueeze(1) # (B, 1, N_formula+N_peaks)
        # print("src_embed", src_embed.shape)
        # print("src_mask", src_mask.shape)
        src_embed = self.formula_spec_encoder(src_embed, src_mask)
  
        decoder_output = self.smiles_decoder(input_ids=tgt_ids, tgt_attn_mask=tgt_mask, encoder_hidden_states=src_embed, encoder_attention_mask=src_mask)
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
        logits = self(
            
                      formula_ids=batch["formula_ids"], 
                      formula_mask=batch["formula_mask"], 
                      peaks_ids=batch["peaks_ids"], 
                      peaks_mask=batch["peaks_mask"], 
                      tgt_ids=batch["tgt_ids"], 
                      tgt_mask=batch["tgt_mask"]                      
                    )
        return logits

    def training_step(self, batch, batch_idx):
        logits = self._step(batch, batch_idx)

        smiles_pred_loss, _ = self._cal_loss(logits, self.smiles_decoder.tgt_y, norm=self.smiles_decoder.ntokens)

        batch_size = batch["tgt_ids"].shape[0]
        self.log("train_loss", smiles_pred_loss, batch_size=batch_size)
        return smiles_pred_loss
    
    def validation_step(self, batch, batch_idx):
        logits = self._step(batch, batch_idx)

        smiles_pred_loss, preds = self._cal_loss(logits, self.smiles_decoder.tgt_y, norm=self.smiles_decoder.ntokens)
        token_acc = self._cal_token_acc(logits, self.smiles_decoder.tgt_y)
        acc = self._cal_mol_acc(preds, self.smiles_decoder.tgt_y)

        batch_size = batch["tgt_ids"].shape[0]

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

        batch_size = batch["tgt_ids"].shape[0]

        # 提取目标：batch.masked_node_target 是 dict of tensors
        self.log("test_loss", smiles_pred_loss, batch_size=batch_size)
        self.log("test_token_acc", token_acc, batch_size=batch_size)
        self.log("test_acc", acc, batch_size=batch_size)
        return smiles_pred_loss 
    
    def configure_optimizers(self):
        # 使用 AdamW 优化器
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            amsgrad=True,
            weight_decay=1e-12
        )
        # optimizer = torch.optim.Adam(
        #     self.parameters(),
        #     lr=self.lr,
        #     betas=(0.9, 0.998),
        #     eps=1e-9
        # )

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
    
   
        
        
        

# def pad_str_ids(str_ids_list, str_len, pad_token=0):
#     """
#     将一个list中的str_ids补齐成tensor batch，同时返回padding mask
#     """
#     max_length = max(str_len)
    
#     padded_str_ids = []
#     padding_masks = []
#     for ids in str_ids_list:
#         pad_len = max_length - len(ids)
#         padded_ids = ids + [pad_token] * pad_len
#         padded_str_ids.append(padded_ids)
        
#         # 创建padding mask: 1表示真实token，0表示padding token
#         mask = [1] * len(ids) + [0] * pad_len
#         padding_masks.append(mask)

#     return torch.tensor(padded_str_ids, dtype=torch.long), torch.tensor(padding_masks, dtype=torch.bool)

# class NMRPeakGraphTMREncoderLayer(nn.Module):
#     def __init__(self, 
#                  src_vocab_size,
#                  d_model=512, d_ff=2048, 
#                  encoder_head=8, encoder_layer=4,
#                  dropout=0.1, 
#                  ):
#         super().__init__()
        
#         self.src_vocab_size = src_vocab_size
        
        
#         c = copy.deepcopy
#         pe = PositionalEncoding(d_model, dropout=dropout)
#         self.src_embed = nn.Sequential(Embeddings(d_model=d_model, vocab=self.src_vocab_size))
#         self.peak_feature_pe = c(pe)
#         self.src_pe = c(pe)
        
#         encoder_attn = MultiHeadedAttention(h=encoder_head,d_model=d_model)
#         ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
#         self.peak_local_encoder = Encoder(EncoderLayer(d_model, c(encoder_attn),c(ff), dropout), N=1)
#         self.peaks_global_encoder = Encoder(EncoderLayer(d_model, c(encoder_attn),c(ff), dropout), N=1)
      
#         self.__init_weights__()
    
#     def __init_weights__(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
    
#     def forward(self, peaks_embed, peaks_mask, peak_tokens, global_peaks_mask):
#         """
        
#         peaks_embed: (B, N_peaks, L_peak, D)
#         peaks_mask: (B, N_peaks, L_peak) 
#         """
#         B, N_peaks, _, D = peaks_embed.shape
        

#         peaks_embed_with_peak_token = torch.cat([peak_tokens, peaks_embed], dim=1) # (B*N_peaks, 1+L_peak, D)
#         peaks_mask_with_peak_token = torch.cat(
#             [torch.ones(peaks_embed.shape[0], 1, device=peaks_mask.device), 
#              peaks_mask.view(-1, peaks_mask.shape[-1])], 
#              dim=1).unsqueeze(1) # (B*N_peaks, 1, L_peak+1)
        
#         peaks_embed_with_peak_token = self.peak_encoder(peaks_embed_with_peak_token, peaks_mask_with_peak_token) # (B*N_peaks, L_peak+1, D)
#         peak_tokens_updated = peaks_embed_with_peak_token[:, 0, :] # (B*N_peaks, D)
#         peaks_embed_updated = peaks_embed_with_peak_token[:, 1:, :] # (B*N_peaks, L_peak, D)
        
#         peak_tokens_updated = peak_tokens_updated.reshape(B, N_peaks, peak_tokens_updated.shape[-1]) # (B, N_peaks, D)

#         # position encoding
#         peaks_embed_updated = self.peaks_global_encoder(peak_tokens_updated, global_peaks_mask)

        
#         return peaks_embed_updated, peaks_embed_updated

# class NMRPeakGraphTMREncoder(nn.Module):
#     def __init__(self, 
#                  src_vocab_size,
#                  d_model=512, d_ff=2048, 
#                  encoder_head=8, encoder_layer=4,
#                  dropout=0.1, 
#                  ):
#         super().__init__()
        
#         self.src_vocab_size = src_vocab_size
#         self.peak_token = nn.Parameter(torch.randn(1, 1, d_model))


#         self.__init_weights__()
    
#     def __init_weights__(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
    
#     def forward(self, peaks_ids, peaks_mask):
#         """
#         peaks_ids: (B, N_peaks, L_peak) 
#         peaks_mask: (B, N_peaks, L_peak) 
#         """
#         peaks_embed_updated, peaks_embed_updated = self.peak_local_encoder(peaks_embed, peaks_mask, peak_tokens, global_peaks_mask)
#         peaks_embed_updated, peaks_embed_updated = self.peaks_global_encoder(peaks_embed_updated, peaks_embed_updated, peak_tokens, global_peaks_mask)
        
#         return peaks_embed_updated, peaks_embed_updated

# class NMRPeakPatchFormulaEncoder(nn.Module):
#     def __init__(self, 
#                  src_vocab_size,
#                  d_model=512, d_ff=2048, 
#                  peak_attn_layer=4,
#                  encoder_head=8, encoder_layer=4,
#                  dropout=0.1, 
#                  ):
#         super().__init__()

#         self.src_vocab_size = src_vocab_size
        
#         c = copy.deepcopy
#         pe = PositionalEncoding(d_model, dropout=dropout)
#         self.src_embed = nn.Sequential(Embeddings(d_model=d_model, vocab=self.src_vocab_size))
#         self.peak_feature_pe = c(pe)
#         self.src_pe = c(pe)

#         encoder_attn = MultiHeadedAttention(h=encoder_head,d_model=d_model)
#         ff = PositionwiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        
#         self.peak_token = nn.Parameter(torch.randn(1, 1, d_model))
#         self.spec_formula_encoder = Encoder(EncoderLayer(d_model, c(encoder_attn),c(ff), dropout), N=encoder_layer)

#         self.__init_weights__()
    
#     def __init_weights__(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Embedding):
#                 nn.init.xavier_normal_(m.weight)
    
#     def forward(self, formula_ids, formula_mask, peaks_ids, peaks_mask):
#         """
#         formula_ids: (B, L_formula) formula ids
#         formula_mask: (B, L_formula) formula mask
#         peaks_ids: (B, N_peaks, L_peak) peaks ids
#         peaks_mask: (B, N_peaks, L_peak) peaks mask
#         """
#         formula_embed = self.src_embed(formula_ids)
#         if len(formula_mask.shape) == 2:
#             formula_mask = formula_mask.unsqueeze(1)
#         # print("formula_embed", formula_embed.shape)
        
#         peaks_ = peaks_ids.reshape(-1, peaks_ids.shape[-1]) # (B*N_peaks, L_peak)
#         peaks_embed = self.peak_feature_pe(self.src_embed(peaks_)) # (B*N_peaks, L_peak, D)
        
#         peak_tokens = self.peak_token.expand(peaks_embed.shape[0], -1, -1)
#         # print("peak_tokens", peak_tokens.shape)
#         peaks_embed_with_peak_token = torch.cat([peak_tokens, peaks_embed], dim=1) # (B*N_peaks, L_peak+1, D)
#         # print("peaks_embed_with_peak_token", peaks_embed_with_peak_token.shape)
#         peaks_mask_with_peak_token = torch.cat(
#             [torch.ones(peaks_embed.shape[0], 1, device=peaks_mask.device), 
#              peaks_mask.view(-1, peaks_mask.shape[-1])], 
#              dim=1).unsqueeze(1) # (B*N_peaks, 1, L_peak+1)

#         peaks_embed_with_peak_token = self.peak_encoder(peaks_embed_with_peak_token, peaks_mask_with_peak_token) # (B*N_peaks, L_peak+1, D)
#         peaks_embed = peaks_embed_with_peak_token[:, 0, :] # (B*N_peaks, D)
#         peaks_embed = peaks_embed.reshape(formula_embed.shape[0], -1, peaks_embed.shape[-1]) # (B, N_peaks, D)
#         is_all_zero_row = torch.all(peaks_ids == 0, dim=-1)
#         global_peaks_mask = (~is_all_zero_row).int().unsqueeze(1) # (B, 1, N_peaks)
#         # global_peaks_mask = torch.all(peaks_mask == 0, dim=-1).unsqueeze(1) # (B, 1, N_peaks)
#         # print("peaks_ids")
#         # print(peaks_ids)
#         # print("global_peaks_mask")
#         # print(global_peaks_mask)

#         src_embed = self.src_pe(torch.cat([formula_embed, peaks_embed], dim=1)) # (B, L_formula+N_peaks, D)
#         src_mask = torch.cat([formula_mask, global_peaks_mask], dim=2) # (B, 1,  L_formula+N_peaks)
#         # print("src_embed", src_embed.shape, "src_mask", src_mask.shape)

#         src_embed = self.spec_formula_encoder(src_embed, src_mask)

#         return src_embed, src_mask
        
        
        
