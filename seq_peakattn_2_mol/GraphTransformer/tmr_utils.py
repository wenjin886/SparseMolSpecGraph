# https://huggingface.co/transformers/v4.5.1/_modules/transformers/models/bert/modeling_bert.html
# https://github.com/microsoft/GraphFormers/tree/main?tab=readme-ov-file
import math
import os

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import copy

BertLayerNorm = torch.nn.LayerNorm
class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, vocab_size, hidden_size, 
                 fix_word_embedding,
                 max_position_embeddings, type_vocab_size, 
                 layer_norm_eps, hidden_dropout_prob):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if fix_word_embedding:
            self.word_embeddings.weight.requires_grad = False
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        if type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        else:
            self.token_type_embeddings = None

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + position_embeddings

        if self.token_type_embeddings:
            embeddings = embeddings + self.token_type_embeddings(token_type_ids)

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, position_ids


class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, output_attentions, attention_probs_dropout_prob):
        super(BertSelfAttention, self).__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.output_attentions = output_attentions

        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def multi_head_attention(self, query, key, value, attention_mask, rel_pos):
        query_layer = self.transpose_for_scores(query)
        key_layer = self.transpose_for_scores(key)
        value_layer = self.transpose_for_scores(value)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask
        if rel_pos is not None:
            attention_scores = attention_scores + rel_pos

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return (context_layer, attention_probs) if self.output_attentions else (context_layer,)

    def forward(self, hidden_states, attention_mask=None, 
                encoder_hidden_states=None, 
                split_lengths=None, rel_pos=None):
        mixed_query_layer = self.query(hidden_states)
        if split_lengths:
            assert not self.output_attentions

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        if split_lengths:
            query_parts = torch.split(mixed_query_layer, split_lengths, dim=1)
            key_parts = torch.split(mixed_key_layer, split_lengths, dim=1)
            value_parts = torch.split(mixed_value_layer, split_lengths, dim=1)

            key = None
            value = None
            outputs = []
            sum_length = 0
            for (query, _key, _value, part_length) in zip(query_parts, key_parts, value_parts, split_lengths):
                key = _key if key is None else torch.cat((key, _key), dim=1)
                value = _value if value is None else torch.cat((value, _value), dim=1)
                sum_length += part_length
                outputs.append(self.multi_head_attention(
                    query, key, value, attention_mask[:, :, sum_length - part_length: sum_length, :sum_length], 
                    rel_pos=None if rel_pos is None else rel_pos[:, :, sum_length - part_length: sum_length, :sum_length], 
                )[0])
            outputs = (torch.cat(outputs, dim=1), )
        else:
            outputs = self.multi_head_attention(
                mixed_query_layer, mixed_key_layer, mixed_value_layer, 
                attention_mask, rel_pos=rel_pos)
        return outputs

class BertIntermediate(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.ReLU()
        

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, intermediate_size, hidden_size, layer_norm_eps, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertSelfOutput(nn.Module):
    """
    NO intermediate_size (compare to BertOutput)
    """
    def __init__(self, hidden_size, layer_norm_eps, hidden_dropout_prob):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, output_attentions, attention_probs_dropout_prob):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(hidden_size, num_attention_heads, output_attentions, attention_probs_dropout_prob)
        self.output = BertSelfOutput(hidden_size, num_attention_heads, output_attentions, attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, 
                split_lengths=None, rel_pos=None):
        self_outputs = self.self(
            hidden_states, attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states, 
            split_lengths=split_lengths, rel_pos=rel_pos)
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, output_attentions, attention_probs_dropout_prob):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(hidden_size, num_attention_heads, output_attentions, attention_probs_dropout_prob)
        # self.intermediate = BertIntermediate(hidden_size, num_attention_heads, output_attentions, attention_probs_dropout_prob)
        # self.output = BertOutput(hidden_size, num_attention_heads, output_attentions, attention_probs_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, split_lengths=None, rel_pos=None):
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, 
            split_lengths=split_lengths, rel_pos=rel_pos)
        attention_output = self_attention_outputs[0]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + self_attention_outputs[1:]
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_hidden_layers, output_attentions, output_hidden_states, attention_probs_dropout_prob):
        super(BertEncoder, self).__init__()
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        bert_layer = BertLayer(hidden_size, num_attention_heads, output_attentions, attention_probs_dropout_prob)
        c = copy.deepcopy
        self.layer = nn.ModuleList([c(bert_layer) for _ in range(num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, split_lengths=None, rel_pos=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, 
                split_lengths=split_lengths, rel_pos=rel_pos)
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


def relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
    """
    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593
    """
    ret = 0
    if bidirectional:
        num_buckets //= 2
        # mtf.to_int32(mtf.less(n, 0)) * num_buckets
        ret += (relative_position > 0).long() * num_buckets
        n = torch.abs(relative_position)
    else:
        n = torch.max(-relative_position, torch.zeros_like(relative_position))
    # now n is in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2
    is_small = n < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
    val_if_large = max_exact + (
        torch.log(n.float() / max_exact) / math.log(max_distance /
                                                    max_exact) * (num_buckets - max_exact)
    ).to(torch.long)
    val_if_large = torch.min(
        val_if_large, torch.full_like(val_if_large, num_buckets - 1))

    ret += torch.where(is_small, n, val_if_large)
    return ret
