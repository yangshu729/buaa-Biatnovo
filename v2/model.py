import logging
import numpy as np
import torch
import torch.nn as nn
import deepnovo_config
import torch.nn.functional as F
logger = logging.getLogger(__name__)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mode=False, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
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
        if mode:
            q, attn = self.attention(q, k, v, mode=True, mask=mask)
        else:
            q, attn = self.attention(q, k, v, mode=False, mask=mask)
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual
        q = self.layer_norm(q)
        return q, attn

class SpectrumCNN(nn.Module):
    def __init__(self, dropout_keep: dict):
        super().__init__()
        self.maxpool1 = nn.MaxPool2d((1, 50))
        self.conv1 = nn.Conv2d(1, 4, (5, 5), stride=(5, 1), padding=(0, 2))
        self.conv2 = nn.Conv2d(4, 16, (1, 5), stride=(1, 1), padding=(0, 2))
        self.maxpool2 = nn.MaxPool2d((1, 6), stride=(1, 4), padding=(0, 1))
        # self.fc = nn.Linear(750, 512)
        self.fc = nn.Linear(750, 256)
        self.dropout1 = nn.Dropout(p=dropout_keep["conv"])
        self.dropout2 = nn.Dropout(p=dropout_keep["dense"])

    def forward(self, spectrum_holder):
        spectrum_holder = spectrum_holder.view(-1, deepnovo_config.neighbor_size, deepnovo_config.MZ_SIZE, 1)
        spectrum_holder = spectrum_holder.permute(0, 3, 1, 2)
        # (batchsize, 1, 5, 150000)
        output = self.maxpool1(spectrum_holder)
        # (batchsize, 1, 5, 3000)
        output = F.relu(self.conv1(output))
        # (batchsize, 4, 1, 3000)
        output = F.relu(self.conv2(output))
        # (batchsize, 16, 1, 3000)
        output = self.maxpool2(output)
        # (batchsize, 16, 1, 750)
        #output = F.dropout(output, p=self.dropout_keep["conv"], training=True)
        output = self.dropout1(output)
        output = output.view(-1, 16, 1 * (deepnovo_config.MZ_SIZE // deepnovo_config.SPECTRUM_RESOLUTION // (4)))
        output = F.relu(self.fc(output))
        output = self.dropout2(output)
        # (batchsize, 16, 256)
        return output
    
class IonCNN(nn.Module):
    def __init__(self, dropout_keep: dict):
        super(IonCNN, self).__init__()
        self.conv1 = nn.Conv3d(26, 64, (1, 3, 3), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(64, 64, (1, 3, 3), padding=(0, 1, 1))
        self.conv3 = nn.Conv3d(64, 64, (1, 3, 3), padding=(0, 1, 1))
        self.maxpool = nn.MaxPool3d((1, 2, 2), padding=(0, 1, 0), stride=(1, 2, 2))
        self.fc = nn.Linear(7680, 512)
        self.dropout1 = nn.Dropout(p=dropout_keep["conv"])
        self.dropout2 = nn.Dropout(p=dropout_keep["dense"])

    def forward(self, input_intensity):
        # (batchsize, 26, 40, 10)
        input_intensity = input_intensity.view(
            -1,
            deepnovo_config.vocab_size,
            deepnovo_config.num_ion,
            deepnovo_config.neighbor_size,
            deepnovo_config.WINDOW_SIZE,
        )
        # (batchsize, 26, 8, 5, 10)
        output = F.relu(self.conv1(input_intensity))
        # (batchsize, 64, 8, 5, 10)
        output = F.relu(self.conv2(output))
        # (batchsize, 64, 8, 5, 10)
        output = F.relu(self.conv3(output))
        # (batchsize, 64, 8, 3, 5)
        output = self.maxpool(output)
        output = self.dropout1(output)
        # (batchsize, 7680)
        output = output.view(
            -1,
            deepnovo_config.num_ion
            * (deepnovo_config.neighbor_size // 2 + 1)
            * (deepnovo_config.WINDOW_SIZE // 2)
            * 64,
        )
        # (batchsize, 512)
        output = self.fc(output)
        output = self.dropout2(output)
        return output

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

class DecoderLayer(nn.Module):
    """Compose with three layers"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(dec_input, dec_input, dec_input, mode=True, mask=slf_attn_mask)
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
        #dropout=0.1,
        dropout_keep = {},
        scale_emb=False,
    ):
        super().__init__()

        self.trg_word_emb = nn.Embedding(n_trg_vocab, d_word_vec, padding_idx=pad_idx)
        self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout_keep["transformer"])
        self.layer_stack = nn.ModuleList(
            [DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout_keep["transformer"]) for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, trg_seq, trg_mask, enc_output, src_mask, return_attns=False):
        dec_slf_attn_list, dec_enc_attn_list = [], []
        dec_output = self.trg_word_emb(trg_seq)
        if self.scale_emb:
            dec_output *= self.d_model**0.5
        dec_output = self.dropout(self.position_enc(dec_output))
        dec_output = self.layer_norm(dec_output)
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output,
                enc_output,
                slf_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask,
            )
            dec_slf_attn_list += [dec_slf_attn] if return_attns else []
            dec_enc_attn_list += [dec_enc_attn] if return_attns else []
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output

class DeepNovoAttion(nn.Module):
    def __init__(self, dropout_keep: dict):
        super(DeepNovoAttion, self).__init__()
        self.ion_cnn = IonCNN(dropout_keep=dropout_keep)
        self.spectrum_cnn = SpectrumCNN(dropout_keep=dropout_keep)  
        self.transformer = Transformer(26, 256, 6, 8, 32, 32, 256, 256, 0, dropout_keep=dropout_keep)
        self.word_emb = nn.Embedding(
            deepnovo_config.vocab_size, deepnovo_config.embedding_size, padding_idx=deepnovo_config.PAD_ID
        )
        # self.trg_word_prj = nn.Linear(1024, deepnovo_config.vocab_size, bias=False)
        self.trg_word_prj = nn.Linear(768, deepnovo_config.vocab_size, bias=False)
        self.dropout_keep = dropout_keep

    def get_src_mask(self, spectrum_cnn_output):
        sz_b, len_q = spectrum_cnn_output.size(0), spectrum_cnn_output.size(1)
        return torch.ones((sz_b, len_q), dtype=torch.bool)

    def get_subsequent_mask(self, seq):
        """For masking out the subsequent info."""
        sz_b, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
        return subsequent_mask    
    
    def get_pad_mask(self, seq, pad_idx):
        return (seq != pad_idx).unsqueeze(-2)

    def forward(self, 
                spectrum_holder, # shape=(batchsize, 5, 150000)
                intensity_inputs_forward, # shape = (seq_len, batchsize, 26, 40, 10)
                intensity_inputs_backward,
                decoder_inputs_forward,  # shape=(seq_len - 1, batch_size)
                decoder_inputs_backward):
            spectrum_cnn_outputs = self.spectrum_cnn(spectrum_holder) # (batchsize, 16, 256)
            decoder_inputs_forward_emb_ion = self.word_emb(decoder_inputs_forward)
            decoder_inputs_backward_emb_ion = self.word_emb(decoder_inputs_backward)
            # (batchsize, seq_len, embedding_size)
            decoder_inputs_forward_trans = decoder_inputs_forward.permute(1, 0)
            decoder_inputs_backward_trans = decoder_inputs_backward.permute(1, 0)
            src_mask = self.get_src_mask(spectrum_cnn_outputs)  # 全1,不mask
            # (batchsize, seq_len,seq_len)
            trg_mask = self.get_pad_mask(decoder_inputs_forward_trans, 0) & self.get_subsequent_mask(
                decoder_inputs_forward_trans
            )
            # (batchsize, seq len, ebmedding_size)
            output_transformer_forward = self.transformer(
                decoder_inputs_forward_trans, trg_mask, spectrum_cnn_outputs, src_mask
            )
            output_transformer_backward = self.transformer(
                decoder_inputs_backward_trans, trg_mask, spectrum_cnn_outputs, src_mask
            )
            output_forward = []
            output_backward = []
            for direction, intensity_inputs, decoder_inputs_emb, outputs in zip(
                ["forward", "backward"],
                [intensity_inputs_forward, intensity_inputs_backward],
                [decoder_inputs_forward_emb_ion, decoder_inputs_backward_emb_ion],
                [output_forward, output_backward],
            ):
                for i, AA_2 in enumerate(decoder_inputs_emb):
                    input_intensity = torch.tensor(intensity_inputs[i]).cuda()
                    output = self.ion_cnn(input_intensity)
                    output = output.unsqueeze_(0)  # (1, batchsize, 512)
                    outputs.append(output)
            output_forward = torch.cat(output_forward, dim=0).permute(1, 0, 2)
            output_backward = torch.cat(output_backward, dim=0).permute(1, 0, 2)
             # (batchsize, seq len, 768)
            output_forward = torch.cat([output_transformer_forward, output_forward], dim=2)
            # (batchsize, seq len, 768)
            output_backward = torch.cat([output_transformer_backward, output_backward], dim=2)
            # (batchsize, seq len, 26)
            logit_forward = self.trg_word_prj(output_forward)
            logit_backward = self.trg_word_prj(output_backward)
            # (batchsize , seq len, 26)
            #return logit_forward.view(-1, logit_forward.size(2)), logit_backward.view(-1, logit_backward.size(2))
            return logit_forward, logit_backward
    
class InferenceModelWrapper(object):
    def __init__(self, model : DeepNovoAttion):
        self.model = model
        self.device = device
        # make sure model in eval mode
        self.model.eval()

    def init_spectrum_cnn(self, spectrum_holder: torch.Tensor):
        with torch.no_grad():
            spectrum_holder = spectrum_holder.to(self.device)
            return self.model.spectrum_cnn(spectrum_holder)

    def inference(self, spectrum_cnn_outputs, candidate_intensity, decoder_inputs):
        with torch.no_grad():
            # spectrum_cnn_outputs = self.spectrum_cnn(spectrum_holder, self.dropout_keep)
            output_ion_cnn = self.model.ion_cnn(candidate_intensity) # candidate_intensity shape=(batchsize, 26, 40, 10)
            # (batchsize, embedding_size)
            src_mask = self.model.get_src_mask(spectrum_cnn_outputs) # spectrum_cnn_outputs shape=(batchsize, 16, 256), src_mask shape=(batchsize, 16)
            decoder_inputs_trans = decoder_inputs.permute(1, 0) # decoder_inputs shape=(seq_len, batchsize), decode_inputs_trans shape=(batchsize, seq_len)
            # (1, 当前步序列)
            trg_mask = self.model.get_subsequent_mask(decoder_inputs_trans) 
            output_transformer_forward = self.model.transformer( # (batchsize, seq len, ebmedding_size)
                decoder_inputs_trans, trg_mask, spectrum_cnn_outputs, src_mask
            )
            output_transformer_forward = output_transformer_forward[:, -1, :]
            # (batchsize, embedding_size)
            output_forward = torch.cat([output_transformer_forward, output_ion_cnn], dim=1)
            # output_forward = output_transformer_forward
            logit_forward = self.model.trg_word_prj(output_forward)
            # (batchsize, embedding_size)
            return logit_forward

    