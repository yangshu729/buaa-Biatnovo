import logging
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepnovo_config 

logger = logging.getLogger(__name__)

# most of the code is from https://nlp.seas.harvard.edu/annotated-transformer/


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].clone().detach()
        return self.dropout(x)

# class PositionalEncoding(nn.Module):
#     def __init__(self,
#                  emb_size: int,
#                  dropout: float,
#                  maxlen: int = 5000):
#         super(PositionalEncoding, self).__init__()
#         den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
#         pos = torch.arange(0, maxlen).reshape(maxlen, 1)
#         pos_embedding = torch.zeros((maxlen, emb_size))
#         pos_embedding[:, 0::2] = torch.sin(pos * den)
#         pos_embedding[:, 1::2] = torch.cos(pos * den)
#         pos_embedding = pos_embedding.unsqueeze(-2)

#         self.dropout = nn.Dropout(dropout)
#         self.register_buffer('pos_embedding', pos_embedding)

#     def forward(self, token_embedding: torch.Tensor):
#         token_embedding =  token_embedding.permute(1, 0, 2)
#         x =  self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
#         return x.permute(1, 0, 2)

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

    def forward(self, query, key, value, mode, mask=None):
        "Compute 'Scaled Dot Product Attention'"
        # query: (batchsize, n_head, len_q, emb_dim)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mode:
            scores = scores.masked_fill(mask == False, -1e9)
        p_attn = scores.softmax(dim=-1)
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
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mode, mask=None):
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
        x, attn = self.attention(q, k, v, mode, mask=mask)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        x = x.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        x = self.dropout(self.fc(x))
        x += residual
        # (batchsize, len_q, 256)
        x = self.layer_norm(x)
        return x


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_ff, eps=1e-6)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(self.dropout1(F.relu(self.w_1(x))))
        x = self.dropout2(x)
        x += residual
        x = self.layer_norm(x)
        return x
        

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, d_model, d_inner, n_head, dropout):
        super(DecoderLayer, self).__init__()
        self.size = d_model  
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.src_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.feed_forward = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
     
    def forward(self, dec_input_q, dec_input_kv, enc_output, attn_mask=None, key_padding_mask = None):
        # mode = True: mask scores before softmax
        residual = dec_input_q
        dec_output = self.self_attn(dec_input_q, dec_input_kv, dec_input_kv, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        dec_output = self.dropout1(dec_output)
        dec_output = self.norm1(residual + dec_output)

        residual = dec_output
        dec_output = self.src_attn(dec_output, enc_output, enc_output, attn_mask=None, key_padding_mask = None, need_weights=False)[0]
        dec_output = self.dropout2(dec_output)
        dec_output = self.norm2(residual + dec_output)

        dec_output = self.feed_forward(dec_output)
        return dec_output

class TransformerDecoder(nn.Module):
    def __init__(self, n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout_keep={}):
        super(TransformerDecoder, self).__init__()
        logger.info("Transformer Parameters - n_trg_vocab: %d, d_word_vec: %d, n_layers: %d, n_head: %d, d_k: %d, d_v: %d, d_model: %d, d_inner: %d, dropout: %.4f" % 
                (n_trg_vocab, d_word_vec, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout_keep["transformer"]))
        self.trg_word_emb = TokenEmbedding(n_trg_vocab, d_word_vec)
        self.position_enc = PositionalEncoding(d_word_vec, dropout_keep["transformer"])
        self.dropout = nn.Dropout(p=dropout_keep["transformer"])
        self.forward_layer_stacks = nn.ModuleList(
            [DecoderLayer(d_model, d_inner, n_head, dropout=dropout_keep["transformer"]) for _ in range(n_layers)]
        )
        self.backward_layer_stacks = nn.ModuleList(
            [DecoderLayer(d_model, d_inner, n_head, dropout=dropout_keep["transformer"]) for _ in range(n_layers)]
        )
        
    def forward(self, trg_seq_l2r, trg_seq_r2l, enc_output, trg_seq_l2r_predict = None, 
                trg_seq_r2l_predict = None, attn_mask=None, key_padding_mask=None):
        dec_output_l2r = self.trg_word_emb(trg_seq_l2r)
        dec_output_l2r = self.position_enc(dec_output_l2r) 
        # dec_output_l2r = self.norm(dec_output_l2r)
        dec_output_r2l = self.trg_word_emb(trg_seq_r2l)
        dec_output_r2l = self.position_enc(dec_output_r2l)
        if deepnovo_config.is_sb:
            dec_output_l2r_predict = self.trg_word_emb(trg_seq_l2r_predict)
            dec_output_l2r_predict = self.position_enc(dec_output_l2r_predict)
            dec_output_r2l_predict = self.trg_word_emb(trg_seq_r2l_predict)
            dec_output_r2l_predict = self.position_enc(dec_output_r2l_predict)
        # dec_output_r2l = self.norm(dec_output_r2l)
        if deepnovo_config.is_sb:
            # 交互式双向
            for forward_dec_layer, backward_dec_layer in zip(self.forward_layer_stacks, self.backward_layer_stacks):
                # (batchsize * beamsize, len_q, 256)
                dec_output_l2r_1 = forward_dec_layer(
                    dec_output_l2r, dec_output_l2r, enc_output, attn_mask = attn_mask, key_padding_mask = key_padding_mask
                )

                dec_output_l2r_2 = forward_dec_layer(
                    dec_output_l2r, dec_output_r2l_predict, enc_output, attn_mask = attn_mask, key_padding_mask = key_padding_mask
                )

                # forward state
                dec_output_l2r = dec_output_l2r_1 + F.relu(dec_output_l2r_2) * 0.1
                # dec_output_l2r = dec_output_l2r_1

                dec_output_r2l_1 = backward_dec_layer(
                    dec_output_r2l, dec_output_r2l, enc_output, attn_mask = attn_mask, key_padding_mask = key_padding_mask
                )

                dec_output_r2l_2 = backward_dec_layer(
                    dec_output_r2l, dec_output_l2r_predict, enc_output, attn_mask = attn_mask, key_padding_mask = key_padding_mask
                )
                #backward state
                dec_output_r2l = dec_output_r2l_1 + F.relu(dec_output_r2l_2) * 0.1
                # dec_output_r2l = dec_output_r2l_1
        else:
            # 独立双向
            for forward_dec_layer in self.forward_layer_stacks:
                dec_output_l2r = forward_dec_layer(
                    dec_output_l2r, dec_output_l2r, enc_output, attn_mask = attn_mask, key_padding_mask = key_padding_mask
                )
            for backward_dec_layer in self.backward_layer_stacks:
                dec_output_r2l = backward_dec_layer(
                    dec_output_r2l, dec_output_r2l, enc_output, attn_mask = attn_mask, key_padding_mask = key_padding_mask
                )
        return dec_output_l2r, dec_output_r2l

    # def forward(self, trg_seq_l2r, trg_seq_r2l, enc_output, attn_mask=None, key_padding_mask=None):
    #     dec_output_l2r = self.trg_word_emb(trg_seq_l2r)
    #     dec_output_l2r = self.position_enc(dec_output_l2r)
    #     # dec_output_l2r = self.norm(dec_output_l2r)
    #     dec_output_r2l = self.trg_word_emb(trg_seq_r2l)
    #     dec_output_r2l = self.position_enc(dec_output_r2l)
    #     # dec_output_r2l = self.norm(dec_output_r2l)
    #     if deepnovo_config.is_sb:
    #         # 交互式双向
    #         for forward_dec_layer, backward_dec_layer in zip(self.forward_layer_stacks, self.backward_layer_stacks):
    #             # (batchsize * beamsize, len_q, 256)
    #             dec_output_l2r_1 = forward_dec_layer(
    #                 dec_output_l2r, dec_output_l2r, enc_output, attn_mask = attn_mask, key_padding_mask = key_padding_mask
    #             )

    #             dec_output_l2r_2 = forward_dec_layer(
    #                 dec_output_l2r, dec_output_r2l, enc_output, attn_mask = attn_mask, key_padding_mask = key_padding_mask
    #             )

    #             # forward state
    #             dec_output_l2r = dec_output_l2r_1 + self.relu(dec_output_l2r_2) * 0.1

    #             dec_output_r2l_1 = backward_dec_layer(
    #                 dec_output_r2l, dec_output_r2l, enc_output, attn_mask = attn_mask, key_padding_mask = key_padding_mask
    #             )

    #             dec_output_r2l_2 = backward_dec_layer(
    #                 dec_output_r2l, dec_output_l2r, enc_output, attn_mask = attn_mask, key_padding_mask = key_padding_mask
    #             )
    #             # backward state
    #             dec_output_r2l = dec_output_r2l_1 + self.relu(dec_output_r2l_2) * 0.1
    #     else:
    #         # 独立双向
    #         for forward_dec_layer in self.forward_layer_stacks:
    #             dec_output_l2r = forward_dec_layer(
    #                 dec_output_l2r, dec_output_l2r, enc_output, attn_mask = attn_mask, key_padding_mask = key_padding_mask
    #             )
    #         for backward_dec_layer in self.backward_layer_stacks:
    #             dec_output_r2l = backward_dec_layer(
    #                 dec_output_r2l, dec_output_r2l, enc_output, attn_mask = attn_mask, key_padding_mask = key_padding_mask
    #             )
    #     return dec_output_l2r, dec_output_r2l

        
class TransformerDecoderFormal(nn.Module):
    def __init__(self, feature_size, num_decoder_layers, num_heads, hidden_dim, dropout=0.1):
        super(TransformerDecoderFormal, self).__init__()
        self.tgt_tok_emb = TokenEmbedding(deepnovo_config.vocab_size, deepnovo_config.embedding_size)
        self.pos_encoder = PositionalEncoding(feature_size, dropout=dropout)
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_size, nhead=num_heads, dim_feedforward=hidden_dim, dropout=dropout,
                                                   batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

    def forward(self, x, memory, tgt_mask=None, tgt_key_padding_mask=None):
        token_emb = self.tgt_tok_emb(x)
        pos_emb = self.pos_encoder(token_emb)
        output = self.transformer_decoder(pos_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)
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

