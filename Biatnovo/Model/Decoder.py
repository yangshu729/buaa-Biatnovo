import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mode=False, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        # if self attention, False -1e9
        if mode:
            attn = attn.masked_fill(mask == False, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.attention = ScaledDotProductAttention(temperature=d_k**0.5, attn_dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mode=False, mask=None):
        # print("q shape:", q.shape) # (batchsize, len_q, 512)
        # print("k shape:", k.shape) # (batchsize, len_k, 512)
        # print("v shape:", v.shape) # (batchsize, len_v, 512)
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q
        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)  # (batchsize, len_q, 8, 64)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)  # (batchsize, len_k, 8, 64)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)  # (batchsize, len_v, 8, 64)
        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.
        if mode:
            q, attn = self.attention(q, k, v, mode=True, mask=mask)
        else:
            q, attn = self.attention(q, k, v, mode=False, mask=mask)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        # (batchsize, len_q, 512)
        q += residual
        q = self.layer_norm(q)
        # (batchsize, len_q, 512)
        return q, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual
        x = self.layer_norm(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer("pos_table", self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table"""
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, : x.size(1)].clone().detach()


class DecoderLayer(nn.Module):
    """Compose with three layers"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input_q, dec_input_kv, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input_q, dec_input_kv, dec_input_kv, mode=True, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(dec_output, enc_output, enc_output, mode=False, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)

        return dec_output, dec_slf_attn, dec_enc_attn


class Transformer(nn.Module):
    def __init__(
        self,
        n_trg_vocab,
        d_word_vec,
        n_layers,
        n_head,
        d_k,
        d_v,
        d_model,
        d_inner,
        pad_idx,
        n_position=200,
        dropout=0.1,
        scale_emb=False,
    ):
        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList(
            [DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, trg_seq_l2r, trg_seq_r2l, trg_mask, enc_output, src_mask, return_attns=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Forward
        dec_output_l2r = self.trg_word_emb(trg_seq_l2r)
        # (batchsize, seq len, 512)
        if self.scale_emb:
            dec_output_l2r *= self.d_model**0.5
        dec_output_l2r = self.dropout(self.position_enc(dec_output_l2r))
        dec_output_l2r = self.layer_norm(dec_output_l2r)

        # backward
        dec_output_r2l = self.trg_word_emb(trg_seq_r2l)
        if self.scale_emb:
            dec_output_r2l *= self.d_model**0.5
        dec_output_r2l = self.dropout(self.position_enc(dec_output_r2l))
        dec_output_r2l = self.layer_norm(dec_output_r2l)
        # (batchsize, seq len, 512)

        for dec_layer in self.layer_stack:
            dec_output_l2r_1, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output_l2r, dec_output_l2r, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask
            )

            dec_output_l2r_2, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output_l2r, dec_output_r2l, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask
            )

            dec_output_r2l_1, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output_r2l, dec_output_r2l, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask
            )

            dec_output_r2l_2, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output_r2l, dec_output_l2r, enc_output, slf_attn_mask=trg_mask, dec_enc_attn_mask=src_mask
            )

            dec_output_l2r = dec_output_l2r_1 + self.relu(dec_output_l2r_2) * 0.1
            dec_output_r2l = dec_output_r2l_1 + self.relu(dec_output_r2l_2) * 0.1
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []
        if return_attns:
            return dec_output_l2r, dec_output_r2l, dec_slf_attn_list, dec_enc_attn_list
        return dec_output_l2r, dec_output_r2l
