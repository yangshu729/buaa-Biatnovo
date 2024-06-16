import logging
import math
import numpy as np
import torch
import torch.nn as nn

import deepnovo_config

logger = logging.getLogger(__name__)

# most of the code is from https://nlp.seas.harvard.edu/annotated-transformer/

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_hid, n_position=200):
#         super(PositionalEncoding, self).__init__()

#         # Not a parameter
#         self.register_buffer("pos_table", self._get_sinusoid_encoding_table(n_position, d_hid))

#     def _get_sinusoid_encoding_table(self, n_position, d_hid):
#         """Sinusoid position encoding table"""
#         # TODO: make it with torch instead of numpy

#         def get_position_angle_vec(position):
#             return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

#         sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
#         sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
#         sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

#         return torch.FloatTensor(sinusoid_table).unsqueeze(0)

#     def forward(self, x):
#         return x + self.pos_table[:, : x.size(1)].clone().detach()

class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=deepnovo_config.PAD_ID)
        self.emb_size = emb_size

    def forward(self, tokens: torch.Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, attn_dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, query, key, value, mode = True, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        # query: (batchsize, n_head, len_q, emb_dim)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = scores.softmax(dim=-1)
        p_attn = self.dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        # we use d_k = d_v = d_model / n_head
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, d_model)
        self.w_ks = nn.Linear(d_model, d_model)
        self.w_vs = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention(attn_dropout=dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)  # (batchsize, len_q, 8, 64)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)  # (batchsize, len_k, 8, 64)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)  # (batchsize, len_v, 8, 64)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.
        x, attn = self.attention(q, k, v, mask=mask)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        x = x.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        return self.fc(x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, d_model, d_inner, d_k, d_v, n_head, dropout):
        super(DecoderLayer, self).__init__()
        self.size = d_model
        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.src_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(self.size, dropout) for _ in range(3)])
        
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class TransformerDecoder(nn.Module):
    def __init__(self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, pad_idx, n_position=200, dropout_keep={}):
        super(TransformerDecoder, self).__init__()
        logger.info("Transformer Parameters - n_trg_vocab: %d, d_word_vec: %d, n_layers: %d, n_head: %d, d_k: %d, d_v: %d, d_model: %d, d_inner: %d, dropout: %.4f" % 
                (n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout_keep["transformer"]))
        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout_keep["transformer"])
        self.layer_stack = nn.ModuleList(
            [DecoderLayer(d_model, d_inner, d_k, d_v, n_head, dropout=dropout_keep["transformer"]) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trg_seq, trg_mask, enc_output, src_mask):
        dec_output = self.trg_word_emb(trg_seq)
        dec_output = self.dropout(self.position_enc(dec_output))
        for dec_layer in self.layer_stack:
            dec_output = dec_layer(
                dec_output,
                enc_output,
                src_mask = src_mask,
                tgt_mask = trg_mask
            )
        return self.norm(dec_output)
        
class TransformerDecoderFormal(nn.Module):
    def __init__(self, feature_size, num_decoder_layers, num_heads, hidden_dim, dropout=0.1):
        super(TransformerDecoderFormal, self).__init__()
        self.tgt_tok_emb = TokenEmbedding(deepnovo_config.vocab_size, deepnovo_config.embedding_size)
        self.pos_encoder = PositionalEncoding(emb_size=feature_size, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout,
                                                   batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None):
        x = self.pos_encoder(self.tgt_tok_emb(x))
        output = self.transformer_decoder(x, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return output

class TransformerEncoderDecoder(nn.Module):
    def __init__(self):
        super(TransformerEncoderDecoder, self).__init__()
        self.transformer = nn.Transformer(d_model=deepnovo_config.embedding_size,
                                           nhead=deepnovo_config.n_head, 
                                           num_encoder_layers=deepnovo_config.n_layers,
                                           num_decoder_layers=deepnovo_config.n_layers,
                                           dim_feedforward=deepnovo_config.d_inner,
                                           dropout=deepnovo_config.dropout_keep["transformer"],)
        self.tgt_tok_emb = TokenEmbedding(deepnovo_config.vocab_size, deepnovo_config.embedding_size)
        self.pos_encoder = PositionalEncoding(emb_size=deepnovo_config.embedding_size, dropout=deepnovo_config.dropout_keep["transformer"])
    
    def forward(self, src_emb, tgt, tgt_mask=None, tgt_key_padding_mask=None):
        tgt_emb = self.pos_encoder(self.tgt_tok_emb(tgt))
        output = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        return output

