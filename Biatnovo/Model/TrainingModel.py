import torch
import numpy as np
import torch.nn as nn
import deepnovo_config
from six.moves import xrange  # pylint: disable=redefined-builtin
from Model.Encoder import Spectrum_cnn, Ion_cnn, Spectrum_cnn_DDA, Ion_cnn_DDA
from Model.Decoder import Transformer


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
        self.trg_word_prj = nn.Linear(768, deepnovo_config.vocab_size, bias=False)

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
            output_transformer_forward, output_transformer_backward = self.transformer(
                decoder_inputs_forward_trans, decoder_inputs_backward_trans, trg_mask, spectrum_cnn_outputs, src_mask
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

    def Inference_SB(
        self,
        spectrum_cnn_outputs,
        candidate_intensity_l2r,
        candidate_intensity_r2l,
        decoder_inputs_l2r,
        decoder_inputs_r2l,
    ):
        output_ion_cnn_l2r = self.ion_cnn(candidate_intensity_l2r, self.dropout_keep)
        output_ion_cnn_r2l = self.ion_cnn(candidate_intensity_r2l, self.dropout_keep)
        src_mask = self.get_src_mask(spectrum_cnn_outputs)
        decoder_inputs_trans_l2r = decoder_inputs_l2r.permute(1, 0)
        decoder_inputs_trans_r2l = decoder_inputs_r2l.permute(1, 0)
        trg_mask = self.get_subsequent_mask(decoder_inputs_trans_l2r)
        output_transformer_forward, output_transformer_backward = self.transformer(
            decoder_inputs_trans_l2r, decoder_inputs_trans_r2l, trg_mask, spectrum_cnn_outputs, src_mask
        )
        output_transformer_forward = output_transformer_forward[:, -1, :]
        output_transformer_backward = output_transformer_backward[:, -1, :]
        output_forward = torch.cat([output_transformer_forward, output_ion_cnn_l2r], dim=1)
        logit_forward = self.trg_word_prj(output_forward)
        output_backward = torch.cat([output_transformer_backward, output_ion_cnn_r2l], dim=1)
        logit_backward = self.trg_word_prj(output_backward)
        return logit_forward, logit_backward

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
            self.dropout_keep["conv"] = deepnovo_config.keep_conv
            self.dropout_keep["dense"] = deepnovo_config.keep_dense
        else:
            self.dropout_keep["conv"] = 0.0
            self.dropout_keep["dense"] = 0.0

        self.spectrum_cnn = Spectrum_cnn_DDA()
        self.ion_cnn = Ion_cnn_DDA()
        if training_mode:
            self.transformer = Transformer(26, 256, 6, 8, 32, 32, 256, 256, 0, dropout=0.5)
        else:
            self.transformer = Transformer(26, 256, 6, 8, 32, 32, 256, 256, 0, dropout=0.5)
        self.word_emb = nn.Embedding(
            deepnovo_config.vocab_size, deepnovo_config.embedding_size, padding_idx=deepnovo_config.PAD_ID
        )
        self.trg_word_prj = nn.Linear(768, deepnovo_config.vocab_size, bias=True)

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
            output_transformer_forward, output_transformer_backward = self.transformer(
                decoder_inputs_forward_trans, decoder_inputs_backward_trans, trg_mask, spectrum_cnn_outputs, src_mask
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

    def Inference_SB(
        self,
        spectrum_cnn_outputs,
        candidate_intensity_l2r,
        candidate_intensity_r2l,
        decoder_inputs_l2r,
        decoder_inputs_r2l,
    ):
        output_ion_cnn_l2r = self.ion_cnn(candidate_intensity_l2r, self.dropout_keep)
        output_ion_cnn_r2l = self.ion_cnn(candidate_intensity_r2l, self.dropout_keep)
        src_mask = self.get_src_mask(spectrum_cnn_outputs)
        decoder_inputs_trans_l2r = decoder_inputs_l2r.permute(1, 0)
        decoder_inputs_trans_r2l = decoder_inputs_r2l.permute(1, 0)
        trg_mask = self.get_subsequent_mask(decoder_inputs_trans_l2r)
        output_transformer_forward, output_transformer_backward = self.transformer(
            decoder_inputs_trans_l2r, decoder_inputs_trans_r2l, trg_mask, spectrum_cnn_outputs, src_mask
        )
        output_transformer_forward = output_transformer_forward[:, -1, :]
        output_transformer_backward = output_transformer_backward[:, -1, :]
        output_forward = torch.cat([output_transformer_forward, output_ion_cnn_l2r], dim=1)
        logit_forward = self.trg_word_prj(output_forward)
        output_backward = torch.cat([output_transformer_backward, output_ion_cnn_r2l], dim=1)
        logit_backward = self.trg_word_prj(output_backward)
        return logit_forward, logit_backward

    def Spectrum_output_inference(self, spectrum_holder):
        spectrum_cnn_outputs = self.spectrum_cnn(spectrum_holder, self.dropout_keep)
        return spectrum_cnn_outputs
