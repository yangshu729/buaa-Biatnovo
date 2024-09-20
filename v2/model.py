import logging
import math
import numpy as np
import torch
import torch.nn as nn
import deepnovo_config
import torch.nn.functional as F

from transformer_decoder import TransformerDecoder, TransformerDecoderFormal
from v2.custom_encoder import CustomConv3D, CustomLinear, CustomLinearNoReLU, constant_initializer, uniform_unit_scaling_initializer, variance_scaling_initializer

torch.autograd.set_detect_anomaly(True)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SpectrumCNN2(nn.Module):
    '''
        完全按照deepnovo-dia的模型结构版本
        used to init lstm
    '''
    def __init__(self):
        super().__init__()
        self.neighbor_size = deepnovo_config.neighbor_size
        self.mz_size = deepnovo_config.MZ_SIZE
        self.spectrum_resolution = deepnovo_config.SPECTRUM_RESOLUTION
        #  Total Padding=(W−1)×S+F−W   general formula
        # stride = 1 , p = (F - 1) / 2

        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, self.spectrum_resolution), stride=(1, self.spectrum_resolution), padding=(0, 0))
        # 定义第一层卷积：输入通道1，输出通道4，卷积核大小(neighbor_size, 5)
        self.conv1 = nn.Conv2d(1, 4, (deepnovo_config.neighbor_size, 5), stride=(deepnovo_config.neighbor_size, 1), padding=(0, 2))
        # 定义第二层卷积：输入输出通道均为4，卷积核大小(1, 4)
        self.conv2 = nn.Conv2d(4, 4, (1, 5), stride=(1, 1), padding=(0, 2))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 6), stride=(1, 4), padding=(0, 1))
        # Dropout层
        self.dropout = nn.Dropout(p=(1 - deepnovo_config.dropout_keep["conv"]))
        # 计算全连接层输入尺寸
        self.reduced_size = deepnovo_config.MZ_SIZE // self.spectrum_resolution
        # 计算全连接层输入尺寸
        self.dense1_input_size = 1 * (self.reduced_size // 4) * 4
        # 创建全连接层
        self.dense1 = nn.Linear(self.dense1_input_size, deepnovo_config.num_units)
        self.dropout2 = nn.Dropout(p=(1 -deepnovo_config.dropout_keep["dense"]))
        #self.output = nn.Linear(deepnovo_config.num_units, 2 * deepnovo_config.num_units)

        # For the SAME padding, the output height and width are computed as:
        #  out_height = ceil(float(in_height) / float(strides[1]))
        #  out_width  = ceil(float(in_width) / float(strides[2]))

    def forward(self, spectrum_holder):
        # 改变张量形状以适应PyTorch的卷积层和池化层格式，这里维度顺序是(batch_size, channels, height, width)
        # (batchsize, 5, 150000) -> (batchsize, 1, 5, 150000)
        layer0 = spectrum_holder.view(-1, 1, self.neighbor_size, self.mz_size)
        # 这表示池化层将沿着宽度方向上的self.spectrum_resolution个像素进行合并
        # (batchsize, 1, 5, 150000) -> (batchsize, 1, 5, 3000)
        layer0 = self.maxpool1(layer0)

        # 第一层卷积和ReLU激活
        # x=(batchsize, 4, 1, 3000)
        x = F.relu(self.conv1(layer0))

        # 第二层卷积和ReLU激活
        # x=[batch_size, 4, 1, 3000]
        x = F.relu(self.conv2(x))

        # 应用第二次最大池化
        # x= [batch_size, 4, 1, 750]
        x = self.maxpool2(x)

        # 应用Dropout
        # 维度不变，仍为: [batch_size, 4, 1, 750]
        x = self.dropout(x)
        # x=[batch_size, 3000]
        x = x.view(-1, self.dense1_input_size)  # 注意：确保x在重塑前的形状是正确的
        # 应用全连接层和ReLU激活函数
        # x=[batch_size, 512]
        x = F.relu(self.dense1(x))
        # 应用Dropout
        x = self.dropout2(x)
        x = x.unsqueeze(0)  # [1, batch_size, num_units]
        #x = x.expand(deepnovo_config.lstm_layers, -1, -1)
        #h0, c0 = torch.split(x, deepnovo_config.num_units, dim=2)
        return x

class SpectrumCNN(nn.Module):
    def __init__(self, dropout_keep: dict):
        '''
        use for init transformer
        '''
        super().__init__()
        self.maxpool1 = nn.MaxPool2d((1, 50))
        self.conv1 = nn.Conv2d(1, 4, (5, 5), stride=(5, 1), padding=(0, 2))  # (3000 * 1 + 5 - 3000) / 2 = 2
        self.conv2 = nn.Conv2d(4, 16, (1, 5), stride=(1, 1), padding=(0, 2))
        self.maxpool2 = nn.MaxPool2d((1, 6), stride=(1, 4), padding=(0, 1))
        # self.fc = nn.Linear(750, 512)
        self.fc = nn.Linear(750, 256)
        self.dropout1 = nn.Dropout(p= (1 - dropout_keep["conv"]))
        self.dropout2 = nn.Dropout(p= (1 - dropout_keep["dense"]))

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
        #self.conv1 = nn.Conv3d(26, 64, (1, 3, 3), padding=(0, 1, 1))   # (w - F + 2p) / s + 1
        self.conv1 = CustomConv3D(26, 64, (1, 3, 3), (1, 1, 1), padding=(0, 1, 1))
        self.conv2 = CustomConv3D(64, 64, (1, 3, 3), (1, 1, 1), padding=(0, 1, 1))  #
        self.conv3 = CustomConv3D(64, 64, (1, 3, 3), (1, 1, 1), padding=(0, 1, 1))
        self.maxpool = nn.MaxPool3d((1, 2, 2), padding=(0, 1, 0), stride=(1, 2, 2))
        #self.fc = nn.Linear(7680, 512)
        self.dense1 = CustomLinear(7680, 512, init_weight=lambda x: variance_scaling_initializer(x, 1.43),
                                init_bias=lambda x: constant_initializer(x, 0.1))
        #self.fc = CustomLinear(512, 26, init_weight=lambda x: uniform_unit_scaling_initializer(x, 1.43),
        #                         init_bias=lambda x: constant_initializer(x, 0.1))
        self.fc = CustomLinearNoReLU(512, 26, init_weight=lambda x: variance_scaling_initializer(x, 1.43),
                                    init_bias=lambda x: constant_initializer(x, value=0.1))
        #self.fc = nn.Linear(512, deepnovo_config.vocab_size)
        self.dropout1 = nn.Dropout(p=(1- dropout_keep["conv"]))
        self.dropout2 = nn.Dropout(p=(1 -dropout_keep["dense"]))

    def forward(self, input_intensity):
        # (batchsize, 26, 40, 10)
        input_intensity = input_intensity.view(
            -1,
            deepnovo_config.vocab_size,
            deepnovo_config.num_ion,
            deepnovo_config.neighbor_size,
            deepnovo_config.WINDOW_SIZE,
        )
        # (batchsize, 26, 8, 5, 10) (N, C_{in}, D, H, W)
        conv1 = F.relu(self.conv1(input_intensity))
        # (batchsize, 64, 8, 5, 10)
        conv2 = F.relu(self.conv2(conv1))
        # (batchsize, 64, 8, 5, 10)
        conv3 = F.relu(self.conv3(conv2))
        # (batchsize, 64, 8, 3, 5)
        pool1 = self.maxpool(conv3)
        dropout1 = self.dropout1(pool1)
        # (batchsize, 7680)
        dropout1 = dropout1.view(
            -1,
            deepnovo_config.num_ion
            * (deepnovo_config.neighbor_size // 2 + 1)
            * (deepnovo_config.WINDOW_SIZE // 2)
            * 64,
        )
        # (batchsize, 512)
        dense1 = self.dense1(dropout1)
        dropout2 = self.dropout2(dense1)
        ion_cnn_feature = dropout2

        # Output layer
        ion_cnn_logit = self.fc(dropout2)
        return ion_cnn_feature, ion_cnn_logit

class DeepNovoAttion(nn.Module):
    def __init__(self, dropout_keep: dict):
        super(DeepNovoAttion, self).__init__()
        self.ion_cnn_forward = IonCNN(dropout_keep=dropout_keep)
        self.ion_cnn_backward = IonCNN(dropout_keep=dropout_keep)
        #self.spectrum_cnn = SpectrumCNN2()
        #self.spectrum_cnn = SpectrumCNN(dropout_keep)
        # self.lstm = nn.LSTM(deepnovo_config.embedding_size, deepnovo_config.num_units,
        #                             num_layers=deepnovo_config.lstm_layers,
        #                             batch_first=True)
        self.word_emb = nn.Embedding(deepnovo_config.vocab_size, deepnovo_config.embedding_size, padding_idx=deepnovo_config.PAD_ID)
        self.d_k = deepnovo_config.embedding_size // deepnovo_config.n_head
        self.d_v = self.d_k
       
        self.transformer = TransformerDecoder(deepnovo_config.vocab_size, deepnovo_config.embedding_size, deepnovo_config.n_layers,
                                              deepnovo_config.n_head, self.d_k, self.d_v, deepnovo_config.d_model, deepnovo_config.d_inner, dropout_keep=dropout_keep)
        # else:    
        # self.transformer_forward = TransformerDecoderFormal(deepnovo_config.embedding_size, deepnovo_config.n_layers, deepnovo_config.n_head, deepnovo_config.d_inner, dropout=dropout_keep["transformer"])
        #  # 初始化transformer模型参数
        # for p in self.transformer_forward.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        # self.transformer_backward = TransformerDecoderFormal(deepnovo_config.embedding_size, deepnovo_config.n_layers, deepnovo_config.n_head, deepnovo_config.d_inner, dropout=dropout_keep["transformer"])
        # for p in self.transformer_backward.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)
        # self.transformer_forward.tgt_tok_emb.embedding.weight = self.transformer_backward.tgt_tok_emb.embedding.weight
       
        # self.trg_word_prj = nn.Linear(1024, deepnovo_config.vocab_size, bias=False)
        self.linear = nn.Linear(512, deepnovo_config.vocab_size)
        self.combine_feature_dense1_forward = CustomLinear(768, 512, init_weight=lambda x: variance_scaling_initializer(x, 1.43),
                                init_bias=lambda x: constant_initializer(x, 0.1))
        self.combine_feature_dropout1_forward = nn.Dropout(p=(1 - dropout_keep["dense"]))
        self.combine_feature_dense2_forward = CustomLinearNoReLU(512, deepnovo_config.vocab_size, init_weight=lambda x: variance_scaling_initializer(x, 1.43),
                                    init_bias=lambda x: constant_initializer(x, 0.1))
        self.combine_feature_dense1_backward = CustomLinear(768, 512, init_weight=lambda x: variance_scaling_initializer(x, 1.43),
                                init_bias=lambda x: constant_initializer(x, 0.1))
        self.combine_feature_dropout1_backward = nn.Dropout(p=(1 - dropout_keep["dense"]))
        self.combine_feature_dense2_backward = CustomLinearNoReLU(512, deepnovo_config.vocab_size, init_weight=lambda x: variance_scaling_initializer(x, 1.43),
                                    init_bias=lambda x: constant_initializer(x, 0.1))
        #self.trg_word_prj = nn.Linear(768, deepnovo_config.vocab_size, bias=False)
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
        """最终的 mask 是一个形状为 (batch_size, 1, seq_length) 的布尔张量"""
        # seq shape = (batch_size, seq_len), 如果seq中的元素不等于pad_idx, 则返回True
        return (seq != pad_idx).unsqueeze(-2)

    def combine_feature(self, transformer_output : torch.Tensor, ion_cnn_output):
        output = torch.cat([transformer_output, ion_cnn_output], dim=2)
        # (batchsize, seq len, 768)
        output = self.combine_feature_dense1(output)
        output = self.combine_feature_dropout(output)
        output = self.combine_feature_dense2(output)
        # linear transform to logit [128, 26]
        return output

    def generate_square_subsequent_mask(self, sz):
        """
        Args:
            seq_len: int, the length of the sequence
        """
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self,
                spectrum_cnn_outputs, # (batchsize, 1, 256)
                intensity_inputs_forward,
                intensity_inputs_backward,
                decoder_inputs_forward, # shape=(seq_len - 1, batch_size)
                decoder_inputs_backward):
        decoder_inputs_forward_emb_ion = self.word_emb(decoder_inputs_forward)
        decoder_inputs_backward_emb_ion = self.word_emb(decoder_inputs_backward)
        decoder_inputs_forward_trans = decoder_inputs_forward.permute(1, 0)
        decoder_inputs_backward_trans = decoder_inputs_backward.permute(1, 0)
        src_mask = self.get_src_mask(spectrum_cnn_outputs)
        tgt_padding_mask = (decoder_inputs_forward_trans == 0)
        seq_len = decoder_inputs_forward_trans.size(1)
        tgt_mask = self.generate_square_subsequent_mask(seq_len)
        
        # true : not mask, false : mask
        # trg_mask = self.get_pad_mask(decoder_inputs_forward_trans, 0) & self.get_subsequent_mask(
        #     decoder_inputs_forward_trans)
        
        output_transformer_forward, output_transformer_backward = self.transformer(
            decoder_inputs_forward_trans, decoder_inputs_backward_trans, spectrum_cnn_outputs, attn_mask=tgt_mask, key_padding_mask=tgt_padding_mask)
        
                                                                                   
        # output_transformer_forward = self.transformer_forward(decoder_inputs_forward_trans, spectrum_cnn_outputs, 
        #                                         tgt_mask = tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        
        # output_transformer_backward = self.transformer_backward(decoder_inputs_backward_trans, spectrum_cnn_outputs, 
        #                                         tgt_mask = tgt_mask, tgt_key_padding_mask=tgt_padding_mask)
        # part 2: ion cnn
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
                if direction == "forward":
                    output, output_logit = self.ion_cnn_forward(input_intensity)
                elif direction == "backward":
                    output, output_logit = self.ion_cnn_backward(input_intensity)
                else:
                    raise ValueError("direction must be forward or backward")
                output = output.unsqueeze_(0)  # (1, batchsize, 512)
                outputs.append(output)
        # (seq_len, batchsize, 512) -> (batchsize, seq_len, 512)
        ion_outputs_forward = torch.cat(output_forward, dim=0).permute(1, 0, 2)
        ion_outputs_backward = torch.cat(output_backward, dim=0).permute(1, 0, 2)
        # part3. combine spectrum_cnn + transformer + ion_cnn
        # logit_forward = self.combine_feature(output_transformer_forward, ion_outputs_forward)
        # logit_backward = self.combine_feature(output_transformer_backward, ion_outputs_backward)
        output_forward = torch.cat([output_transformer_forward, ion_outputs_forward], dim=2)
        output_backward = torch.cat([output_transformer_backward, ion_outputs_backward], dim=2)
        logit_forward = self.combine_feature_dense1_forward(output_forward)
        logit_forward = self.combine_feature_dropout1_forward(logit_forward)
        logit_forward = self.combine_feature_dense2_forward(logit_forward)
        logit_backward = self.combine_feature_dense1_backward(output_backward)
        logit_backward = self.combine_feature_dropout1_backward(logit_backward)
        logit_backward = self.combine_feature_dense2_backward(logit_backward)
        return logit_forward, logit_backward


class InferenceModelWrapper(object):

    def __init__(self, forward_model : DeepNovoAttion = None, backward_model : DeepNovoAttion = None, 
                 sbatt_model : DeepNovoAttion = None, spectrum_cnn : SpectrumCNN2 = None):
        self.forward_model = forward_model
        self.backward_model = backward_model
        self.spectrum_cnn = spectrum_cnn
        self.sbatt_model = sbatt_model
        # make sure model in eval mode
        
        self.sbatt_model.eval()
        # else:
        #     self.forward_model.eval()
        #     self.backward_model.eval()
        self.spectrum_cnn.eval()

    def init_spectrum_cnn(self, spectrum_holder: torch.Tensor):
        with torch.no_grad():
            spectrum_holder = spectrum_holder.to(device)
            return self.spectrum_cnn(spectrum_holder).permute(1, 0, 2)
        
    def inference(self, spectrum_cnn_outputs, candidate_intensity_forward,
                     candidate_intensity_backward, decoder_inputs_forward, decoder_inputs_backward):
        with torch.no_grad():
            # (batchsize * beamsize, 512)
            output_ion_cnn_forward, _ = self.sbatt_model.ion_cnn_forward(candidate_intensity_forward)
            output_ion_cnn_backward, _ = self.sbatt_model.ion_cnn_backward(candidate_intensity_backward)
            
            # src_mask = self.sbatt_model.get_src_mask(spectrum_cnn_outputs)
            decoder_inputs_forward_trans = decoder_inputs_forward.permute(1, 0)
            decoder_inputs_backward_trans = decoder_inputs_backward.permute(1, 0)
            tgt_padding_mask = (decoder_inputs_forward_trans == 0)
            seq_len = decoder_inputs_forward_trans.size(1)
            tgt_mask = self.sbatt_model.generate_square_subsequent_mask(seq_len)
            # true : not mask, false : mask
            if deepnovo_config.is_sb:
                output_transformer_forward, output_transformer_backward = self.sbatt_model.transformer(
                decoder_inputs_forward_trans, decoder_inputs_backward_trans, spectrum_cnn_outputs,
                decoder_inputs_forward_trans, decoder_inputs_backward_trans, attn_mask=tgt_mask, key_padding_mask=tgt_padding_mask)
            else:
                output_transformer_forward, output_transformer_backward = self.sbatt_model.transformer(
                    decoder_inputs_forward_trans, decoder_inputs_backward_trans, 
                    spectrum_cnn_outputs, attn_mask=tgt_mask, key_padding_mask=tgt_padding_mask)
        
            output_transformer_forward = output_transformer_forward[:, -1, :]
            output_transformer_backward = output_transformer_backward[:, -1, :]
            output_forward = torch.cat([output_transformer_forward, output_ion_cnn_forward], dim=1)
            output_backward = torch.cat([output_transformer_backward, output_ion_cnn_backward], dim=1)

            logit_forward = self.sbatt_model.combine_feature_dense1_forward(output_forward)
            logit_forward = self.sbatt_model.combine_feature_dropout1_forward(logit_forward)
            logit_forward = self.sbatt_model.combine_feature_dense2_forward(logit_forward)
            logit_backward = self.sbatt_model.combine_feature_dense1_backward(output_backward)
            logit_backward = self.sbatt_model.combine_feature_dropout1_backward(logit_backward)
            logit_backward = self.sbatt_model.combine_feature_dense2_backward(logit_backward)
            return logit_forward, logit_backward

    # def inference(self, spectrum_cnn_outputs, candidate_intensity, decoder_inputs, direction_id):
    #     if direction_id == 0:
    #         self.model = self.forward_model
    #     elif direction_id == 1:
    #         self.model = self.backward_model
    #     else:
    #         raise ValueError("direction must be forward or backward")
    #     with torch.no_grad():
    #         # spectrum_cnn_outputs = self.spectrum_cnn(spectrum_holder, self.dropout_keep)
    #         output_ion_cnn, _ = self.model.ion_cnn(candidate_intensity)
    #         # (batchsize, embedding_size)
    #         src_mask = self.model.get_src_mask(spectrum_cnn_outputs) # spectrum_cnn_outputs shape=(batchsize, 16, 256), src_mask shape=(batchsize, 16)
    #         seq_size = decoder_inputs.size(0)
    #         decoder_inputs_trans = decoder_inputs.permute(1, 0) # decoder_inputs shape=(seq_len, batchsize), decode_inputs_trans shape=(batchsize, seq_len)
    #         # (1, 当前步序列)
    #         trg_mask = self.model.generate_square_subsequent_mask(seq_size)
    #         tgt_padding_mask = (decoder_inputs_trans == 0)
    #         output_transformer_forward = self.model.transformer( # (batchsize, seq len, ebmedding_size)
    #             decoder_inputs_trans, spectrum_cnn_outputs,  tgt_mask=trg_mask, tgt_key_padding_mask=tgt_padding_mask
    #         )
    #         output_transformer_forward = output_transformer_forward[:, -1, :]
    #         # (batchsize, embedding_size)
    #         output_forward = torch.cat([output_transformer_forward, output_ion_cnn], dim=1)
    #         # output_forward = output_transformer_forward
    #         logit_forward = self.model.combine_feature_dense1(output_forward)
    #         logit_forward = self.model.combine_feature_dropout(logit_forward)
    #         logit_forward = self.model.combine_feature_dense2(logit_forward)
    #         # (batchsize, embedding_size)
    #         return logit_forward

