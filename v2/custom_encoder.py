import torch
import torch.nn as nn
import torch.nn.init as init

class CustomLinear(nn.Module):
    def __init__(self, input_size, output_size, init_weight, init_bias):
        super(CustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        self.init_weight = init_weight
        self.init_bias = init_bias
        self.reset_parameters()
    
    def reset_parameters(self):
        self.init_weight(self.weight)
        self.init_bias(self.bias)
    
    def forward(self, input):
        return torch.relu(torch.matmul(input, self.weight.t()) + self.bias)

class CustomLinearNoReLU(nn.Module):
    def __init__(self, input_size, output_size, init_weight, init_bias):
        super(CustomLinearNoReLU, self).__init__()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.bias = nn.Parameter(torch.empty(output_size))
        self.init_weight = init_weight
        self.init_bias = init_bias
        self.reset_parameters()
    
    def reset_parameters(self):
        self.init_weight(self.weight)
        self.init_bias(self.bias)
    
    def forward(self, input):
        return torch.matmul(input, self.weight.t()) + self.bias

def uniform_unit_scaling_initializer(tensor, scale=1.43):
    fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
    bound = scale * torch.sqrt(torch.tensor(3.0 / fan_in))
    init.uniform_(tensor, -bound, bound)

def variance_scaling_initializer(tensor, scale=1.43, mode='fan_avg', distribution='uniform'):
    fan_in, fan_out = init._calculate_fan_in_and_fan_out(tensor)
    
    if mode == 'fan_in':
        scale /= fan_in
    elif mode == 'fan_out':
        scale /= fan_out
    elif mode == 'fan_avg':
        scale /= (fan_in + fan_out) / 2.0
    
    if distribution == 'uniform':
        bound = torch.sqrt(torch.tensor(3.0 * scale))
        init.uniform_(tensor, -bound, bound)
    elif distribution == 'normal':
        std = torch.sqrt(torch.tensor(scale))
        init.normal_(tensor, 0.0, std)

def constant_initializer(tensor, value=0.1):
    init.constant_(tensor, value)

class CustomConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, strides, padding):
        super(CustomConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=strides, padding=padding)
        self.initialize_weights()

    def initialize_weights(self):
        variance_scaling_initializer(self.conv.weight, scale=1.43, mode='fan_avg', distribution='uniform')
        constant_initializer(self.conv.bias, value=0.1)

    def forward(self, x):
        return torch.relu(self.conv(x))

# # 使用示例
# conv1 = CustomConv3D(
#     in_channels=3,
#     out_channels=64,
#     kernel_size=(1, 3, 3),
#     strides=(1, 1, 1),
#     padding=(0, 1, 1)  # PyTorch 中 'same' 等效的计算方式
# )
