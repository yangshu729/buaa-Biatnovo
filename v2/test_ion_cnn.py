import os
import torch

from v2 import deepnovo_config
from v2.model import DeepNovoAttion
from v2.test_accuracy import cal_dia_focal_loss

# from v2.model import IonCNN
# dropout_keep = {"conv": 0.5, "dense": 0.5}
# model = IonCNN(dropout_keep)
# # 启用异常检测
# torch.autograd.set_detect_anomaly(True)
# input_tensor = torch.randn(1, 26, 40, 10, requires_grad=True)
# output = model(input_tensor)

# # 计算损失并进行反向传播
# loss = output.mean()
# loss.backward()

# print("======")
# batch_size = 2
# decoder_size = 4
# num_classes = 3

# # 手动初始化假数据
# pred_forward = torch.tensor([[0.1, 0.2, 0.7],
#                             [0.2, 0.3, 0.5],
#                             [0.4, 0.4, 0.2],
#                             [0.3, 0.3, 0.4],
#                             [0.6, 0.2, 0.2],
#                             [0.5, 0.3, 0.2]], requires_grad=True)

# pred_backward = torch.tensor([[0.2, 0.3, 0.5],
#                                     [0.1, 0.5, 0.4],
#                                     [0.3, 0.3, 0.4],
#                                     [0.4, 0.4, 0.2],
#                                     [0.5, 0.2, 0.3],
#                                     [0.3, 0.3, 0.4]], requires_grad=True)

# gold_forward = torch.tensor([2, 1, 0, 2, 1, 0])
# gold_backward = torch.tensor([1, 2, 1, 0, 2, 1])
# loss = cal_dia_focal_loss(pred_forward, pred_backward, gold_forward, gold_backward, batch_size)
# print(loss)
checkpoint = torch.load(os.path.join("/root/v2/only_ion_cnn/", "translate.ckpt"))
model = DeepNovoAttion(deepnovo_config.dropout_keep)
model.load_state_dict(checkpoint["model"])
for name, param in model.named_parameters():
    if name.startswith("ion_cnn"):
        print(name, param.mean())
