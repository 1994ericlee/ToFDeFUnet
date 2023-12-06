import torch
import torch.nn as nn

# 假设 output 是模型的输出，target 是相应的目标
output = torch.randn(20, 256, 256)
target = torch.randn(20, 256, 256)

# 计算均方误差损失
a = nn.MSELoss(reduction='mean')
loss = a(output, target)

print("损失值:", loss.item())