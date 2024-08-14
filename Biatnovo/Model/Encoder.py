import torch
import torch.nn as nn
import torch.nn.functional as F
import deepnovo_config
from Biatnovo import deepnovo_config_dda


class Spectrum_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = nn.MaxPool2d((1, 50))
        self.conv1 = nn.Conv2d(1, 4, (5, 5), stride=(5, 1), padding=(0, 2))
        self.conv2 = nn.Conv2d(4, 16, (1, 5), stride=(1, 1), padding=(0, 2))
        self.maxpool2 = nn.MaxPool2d((1, 6), stride=(1, 4), padding=(0, 1))
        # self.fc = nn.Linear(750, 512)
        self.fc = nn.Linear(750, 256)

    def forward(self, spectrum_holder, dropout_keep):
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
        output = F.dropout(output, p=dropout_keep["conv"], training=True)
        output = output.view(-1, 16, 1 * (deepnovo_config.MZ_SIZE // deepnovo_config.SPECTRUM_RESOLUTION // (4)))
        output = F.relu(self.fc(output))
        output = F.dropout(output, p=dropout_keep["dense"], training=True)
        # (batchsize, 16, 256)
        return output


class Spectrum_cnn_DDA(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = nn.MaxPool2d((1, 50))
        self.conv1 = nn.Conv2d(1, 4, (1, 5), stride=(5, 1), padding=(0, 2))
        self.conv2 = nn.Conv2d(4, 16, (1, 5), stride=(1, 1), padding=(0, 2))
        self.maxpool2 = nn.MaxPool2d((1, 6), stride=(1, 4), padding=(0, 1))
        self.fc = nn.Linear(750, 256)

    def forward(self, spectrum_holder, dropout_keep):
        spectrum_holder = spectrum_holder.view(-1, deepnovo_config_dda.neighbor_size, deepnovo_config_dda.MZ_SIZE, 1)
        spectrum_holder = spectrum_holder.permute(0, 3, 1, 2)
        output = self.maxpool1(spectrum_holder)
        output = F.relu(self.conv1(output))
        output = F.relu(self.conv2(output))
        output = self.maxpool2(output)
        # (batchsize, 16, 1, 750)
        # dropout
        output = F.dropout(output, p=dropout_keep["conv"], training=True)
        output = output.view(
            -1, 16, 1 * (deepnovo_config_dda.MZ_SIZE // deepnovo_config_dda.SPECTRUM_RESOLUTION // (4))
        )
        output = F.relu(self.fc(output))
        output = F.dropout(output, p=dropout_keep["dense"], training=True)
        # (batchsize, 16, 512)
        return output


class Ion_cnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(26, 64, (1, 3, 3), padding=(0, 1, 1))
        self.conv2 = nn.Conv3d(64, 64, (1, 3, 3), padding=(0, 1, 1))
        self.conv3 = nn.Conv3d(64, 64, (1, 3, 3), padding=(0, 1, 1))
        self.maxpool = nn.MaxPool3d((1, 2, 2), padding=(0, 1, 0), stride=(1, 2, 2))
        self.fc = nn.Linear(7680, 512)

    def forward(self, input_intensity, dropout_keep):
        # (batchsize, 26, 40, 10) 将input_intensity张量重塑成一个五维张量
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
        output = F.dropout(output, p=dropout_keep["conv"])
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
        output = F.dropout(output, p=dropout_keep["dense"])
        return output


class Ion_cnn_DDA(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(26, 64, (1, 1, 3), padding=(0, 0, 1))
        self.conv2 = nn.Conv3d(64, 64, (1, 1, 3), padding=(0, 0, 1))
        self.conv3 = nn.Conv3d(64, 64, (1, 1, 3), padding=(0, 0, 1))
        self.maxpool = nn.MaxPool3d((1, 1, 3), padding=(0, 0, 1), stride=(1, 1, 2))
        # self.fc = nn.Linear(3840, 512)
        self.fc = nn.Linear(2560, 512)

    def forward(self, input_intensity, dropout_keep):
        # (batchsize, 26, 8, 10)
        input_intensity = input_intensity.view(
            -1,
            deepnovo_config_dda.vocab_size,
            deepnovo_config_dda.num_ion,
            deepnovo_config_dda.neighbor_size,
            deepnovo_config_dda.WINDOW_SIZE,
        )
        # (batchsize, 26, 8, 1, 10)
        output = F.relu(self.conv1(input_intensity))
        # (batchsize, 64, 8, 1, 10)
        output = F.relu(self.conv2(output))
        # (batchsize, 64, 8, 1, 10)
        output = F.relu(self.conv3(output))
        # (batchsize, 64, 8, 1, 5)
        output = self.maxpool(output)
        output = F.dropout(output, p=dropout_keep["conv"])
        # (batchsize, 2560)
        output = output.view(
            -1,
            deepnovo_config_dda.num_ion
            * (deepnovo_config_dda.neighbor_size // 2 + 1)
            * (deepnovo_config_dda.WINDOW_SIZE // 2)
            * 64,
        )
        # (batchsize, 512)
        output = F.relu(self.fc(output))
        output = F.dropout(output, p=dropout_keep["dense"])
        return output
