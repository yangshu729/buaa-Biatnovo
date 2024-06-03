import torch

from v2.model import IonCNN
dropout_keep = {"conv": 0.5, "dense": 0.5}
model = IonCNN(dropout_keep)
# 启用异常检测
torch.autograd.set_detect_anomaly(True)
input_tensor = torch.randn(1, 26, 40, 10, requires_grad=True)
output = model(input_tensor)

# 计算损失并进行反向传播
loss = output.mean()
loss.backward()

print("======")
