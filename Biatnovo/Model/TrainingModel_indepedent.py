import torch
import numpy as np
import torch.nn as nn
import deepnovo_config
from Biatnovo import deepnovo_config_dda
from six.moves import xrange  # pylint: disable=redefined-builtin
from Biatnovo.Model.Encoder import Spectrum_cnn, Ion_cnn, Spectrum_cnn_DDA, Ion_cnn_DDA
from Biatnovo.Model.Decoder_indepedent import Transformer


class TrainingModel(nn.Module):
    """TODO(wusiyu): docstring."""

    def __init__(self, opt, training_mode):  # TODO(wusiyu): init
        """TODO(wusiyu): docstring."""
        super().__init__()
        print("TrainingModel: __init__()")
        self.dropout_keep = {}
        if training_mode:
            self.dropout_keep["conv"] = deepnovo_config.keep_conv
            self.dropout_keep["dense"] = deepnovo_config.keep_dense
        else:
            self.dropout_keep["conv"] = 0.0
            self.dropout_keep["dense"] = 0.0
        self.spectrum_cnn = Spectrum_cnn()
        self.ion_cnn = Ion_cnn()
        if training_mode:
            self.transformer = Transformer(26, 256, 6, 8, 32, 32, 256, 256, 0, dropout=0.0)
        else:
            self.transformer = Transformer(26, 256, 6, 8, 32, 32, 256, 256, 0, dropout=0.0)
        self.word_emb = nn.Embedding(
            deepnovo_config.vocab_size, deepnovo_config.embedding_size, padding_idx=deepnovo_config.PAD_ID
        )
        # self.trg_word_prj = nn.Linear(1024, deepnovo_config.vocab_size, bias=False)
        self.trg_word_prj = nn.Linear(768, deepnovo_config.vocab_size, bias=False)
        # self.trg_word_prj = nn.Linear(256, deepnovo_config.vocab_size, bias=False)

    def get_pad_mask(self, seq, pad_idx):
        return (seq != pad_idx).unsqueeze(-2)

    def get_src_mask(self, spectrum_cnn_output):
        sz_b, len_q = spectrum_cnn_output.size(0), spectrum_cnn_output.size(1)
        return torch.ones((sz_b, len_q), dtype=torch.bool)

    def get_subsequent_mask(self, seq):
        """For masking out the subsequent info."""
        sz_b, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
        return subsequent_mask

    def forward(
        self,
        opt,
        spectrum_holder, # shape=(batchsize, 5, 150000)
        intensity_inputs_forward, # (12, batchsize, 26, 40, 10)
        intensity_inputs_backward,
        decoder_inputs_forward, # shape=(seq_len, batchsize)
        decoder_inputs_backward,
        training_mode,
    ):
        if training_mode:
            spectrum_cnn_outputs = self.spectrum_cnn(spectrum_holder, self.dropout_keep)
            # decoder_inputs_forward_emb_ion 没用到？
            decoder_inputs_forward_emb_ion = self.word_emb(decoder_inputs_forward) # shape=(seq_len, batchsize, embedding_size)
            decoder_inputs_backward_emb_ion = self.word_emb(decoder_inputs_backward)
            decoder_inputs_forward_trans = decoder_inputs_forward.permute(1, 0)
            decoder_inputs_backward_trans = decoder_inputs_backward.permute(1, 0)
            src_mask = self.get_src_mask(spectrum_cnn_outputs)
            # (batchsize, seq_len, seq_len), true是有效位置（非遮盖位），只有既是非填充值，也是当前或过去值的位置才是true
            trg_mask = self.get_pad_mask(decoder_inputs_forward_trans, 0) & self.get_subsequent_mask(
                decoder_inputs_forward_trans
            )
            # output_transformer_forward shape=(batchsize, seq len, ebmedding_size)
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
                    # (batchsize, 26, 40, 10)
                    input_intensity = torch.tensor(intensity_inputs[i]).cuda()
                    output = self.ion_cnn(input_intensity, self.dropout_keep)
                    output = output.unsqueeze_(0)  # (1, batchsize, 512)
                    outputs.append(output)
            # ion_cnn_forward_output shape=(batchsize, seq len, 512)
            output_forward = torch.cat(output_forward, dim=0).permute(1, 0, 2)
            output_backward = torch.cat(output_backward, dim=0).permute(1, 0, 2)
            output_forward = torch.cat([output_transformer_forward, output_forward], dim=2)
            # (batchsize, seq len, 1024)
            output_backward = torch.cat([output_transformer_backward, output_backward], dim=2)
            # (batchsize, seq len, 1024)

            logit_forward = self.trg_word_prj(output_forward)  # (batchsize, seq len, 26)
            logit_backward = self.trg_word_prj(output_backward)
            # (batchsize x seq len, 26)
            return logit_forward.view(-1, logit_forward.size(2)), logit_backward.view(-1, logit_backward.size(2))

    def Inference(self, spectrum_cnn_outputs, candidate_intensity, decoder_inputs, direction_id):
        if direction_id == 0:
            output_ion_cnn = self.ion_cnn(candidate_intensity, self.dropout_keep) # candidate_intensity shape=(batchsize, 26, 40, 10)
            # (1, 512)
            src_mask = self.get_src_mask(spectrum_cnn_outputs) # spectrum_cnn_outputs shape=(batchsize, 16, 256), src_mask shape=(batchsize, 16)
            decoder_inputs_trans = decoder_inputs.permute(1, 0) # decoder_inputs shape=(seq_len, batchsize), decode_inputs_trans shape=(batchsize, seq_len)
            # (1, 当前步序列)
            trg_mask = self.get_subsequent_mask(decoder_inputs_trans) 
            output_transformer_forward = self.transformer(
                decoder_inputs_trans, trg_mask, spectrum_cnn_outputs, src_mask
            )
            output_transformer_forward = output_transformer_forward[:, -1, :]
            # (1, 512)
            output_forward = torch.cat([output_transformer_forward, output_ion_cnn], dim=1)
            # output_forward = output_transformer_forward
            logit_forward = self.trg_word_prj(output_forward)
            # (1, 26)
            return logit_forward
            # (1, 26)
        if direction_id == 1:
            output_ion_cnn = self.ion_cnn(candidate_intensity, self.dropout_keep)
            # (1, 512)
            src_mask = self.get_src_mask(spectrum_cnn_outputs)
            decoder_inputs_trans = decoder_inputs.permute(1, 0)
            # (1, 当前步序列)
            trg_mask = self.get_subsequent_mask(decoder_inputs_trans)
            output_transformer_backward = self.transformer(
                decoder_inputs_trans, trg_mask, spectrum_cnn_outputs, src_mask
            )
            output_transformer_backward = output_transformer_backward[:, -1, :]
            # (1, 512)
            output_backward = torch.cat([output_transformer_backward, output_ion_cnn], dim=1)
            # output_backward = output_transformer_backward
            logit_backward = self.trg_word_prj(output_backward)
            # (1, 26)
            return logit_backward

    def Spectrum_output_inference(self, spectrum_holder):
        spectrum_cnn_outputs = self.spectrum_cnn(spectrum_holder, self.dropout_keep)
        return spectrum_cnn_outputs


class TrainingModel_DDA(nn.Module):
    """TODO(wusiyu): docstring."""

    def __init__(self, opt, training_mode):  # TODO(wusiyu): init
        """TODO(wusiyu): docstring."""
        super().__init__()
        print("TrainingModel: __init__()")
        self.dropout_keep = {}
        if training_mode:
            self.dropout_keep["conv"] = deepnovo_config_dda.keep_conv
            self.dropout_keep["dense"] = deepnovo_config_dda.keep_dense
        else:
            self.dropout_keep["conv"] = 0.0
            self.dropout_keep["dense"] = 0.0

        self.spectrum_cnn = Spectrum_cnn_DDA()
        self.ion_cnn = Ion_cnn_DDA()
        if training_mode:
            self.transformer = Transformer(26, 256, 6, 8, 32, 32, 256, 256, 0, dropout=0.2)
        else:
            self.transformer = Transformer(26, 256, 6, 8, 32, 32, 256, 256, 0, dropout=0.0)
        self.word_emb = nn.Embedding(
            deepnovo_config_dda.vocab_size, deepnovo_config_dda.embedding_size, padding_idx=deepnovo_config_dda.PAD_ID
        )
        self.trg_word_prj = nn.Linear(768, deepnovo_config_dda.vocab_size, bias=True)
        # self.trg_word_prj = nn.Linear(256, deepnovo_config_dda.vocab_size, bias=True)

    def get_pad_mask(self, seq, pad_idx):
        return (seq != pad_idx).unsqueeze(-2)

    def get_src_mask(self, spectrum_cnn_output):
        sz_b, len_q = spectrum_cnn_output.size(0), spectrum_cnn_output.size(1)
        return torch.ones((sz_b, len_q), dtype=torch.bool)

    def get_subsequent_mask(self, seq):
        """For masking out the subsequent info."""
        sz_b, len_s = seq.size()
        subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
        return subsequent_mask

    def forward(
        self,
        opt,
        spectrum_holder,
        intensity_inputs_forward,
        intensity_inputs_backward,
        decoder_inputs_forward,
        decoder_inputs_backward,
        training_mode,
    ):
        if training_mode:
            spectrum_cnn_outputs = self.spectrum_cnn(spectrum_holder, self.dropout_keep)
            decoder_inputs_forward_emb_ion = self.word_emb(decoder_inputs_forward)
            decoder_inputs_backward_emb_ion = self.word_emb(decoder_inputs_backward)
            decoder_inputs_forward_trans = decoder_inputs_forward.permute(1, 0)
            decoder_inputs_backward_trans = decoder_inputs_backward.permute(1, 0)
            src_mask = self.get_src_mask(spectrum_cnn_outputs)
            trg_mask = self.get_pad_mask(decoder_inputs_forward_trans, 0) & self.get_subsequent_mask(
                decoder_inputs_forward_trans
            )
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
                    output = self.ion_cnn(input_intensity, self.dropout_keep)
                    output = output.unsqueeze_(0)  # (1, batchsize, 512)
                    outputs.append(output)
            output_forward = torch.cat(output_forward, dim=0).permute(1, 0, 2)
            output_backward = torch.cat(output_backward, dim=0).permute(1, 0, 2)
            output_forward = torch.cat([output_transformer_forward, output_forward], dim=2)
            output_backward = torch.cat([output_transformer_backward, output_backward], dim=2)
            # output_forward = output_transformer_forward
            # output_backward = output_transformer_backward
            logit_forward = self.trg_word_prj(output_forward)
            logit_backward = self.trg_word_prj(output_backward)
            return logit_forward.view(-1, logit_forward.size(2)), logit_backward.view(-1, logit_backward.size(2))

    def Inference(self, spectrum_cnn_outputs, candidate_intensity, decoder_inputs, direction_id):
        if direction_id == 0:
            output_ion_cnn = self.ion_cnn(candidate_intensity, self.dropout_keep)
            src_mask = self.get_src_mask(spectrum_cnn_outputs)
            decoder_inputs_trans = decoder_inputs.permute(1, 0)
            trg_mask = self.get_subsequent_mask(decoder_inputs_trans)
            output_transformer_forward = self.transformer(
                decoder_inputs_trans, trg_mask, spectrum_cnn_outputs, src_mask
            )
            output_transformer_forward = output_transformer_forward[:, -1, :]
            output_forward = torch.cat([output_transformer_forward, output_ion_cnn], dim=1)
            logit_forward = self.trg_word_prj(output_forward)
            return logit_forward
        if direction_id == 1:
            output_ion_cnn = self.ion_cnn(candidate_intensity, self.dropout_keep)
            src_mask = self.get_src_mask(spectrum_cnn_outputs)
            decoder_inputs_trans = decoder_inputs.permute(1, 0)
            trg_mask = self.get_subsequent_mask(decoder_inputs_trans)
            output_transformer_backward = self.transformer(
                decoder_inputs_trans, trg_mask, spectrum_cnn_outputs, src_mask
            )
            output_transformer_backward = output_transformer_backward[:, -1, :]
            output_backward = torch.cat([output_transformer_backward, output_ion_cnn], dim=1)
            logit_backward = self.trg_word_prj(output_backward)
            return logit_backward

    def Spectrum_output_inference(self, spectrum_holder):
        spectrum_cnn_outputs = self.spectrum_cnn(spectrum_holder, self.dropout_keep)
        return spectrum_cnn_outputs