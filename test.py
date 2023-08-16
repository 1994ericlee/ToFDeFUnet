# import torch

# # 创建两个矩阵
# matrix1 = torch.tensor([[1.0, 2, 3],
#                         [4, 5, 6]])
# matrix2 = torch.tensor([[7.0, 8, 9],
#                         [10, 11, 12]])

# # 将两个矩阵合并成一个两通道的矩阵
# combined_matrix = torch.stack((matrix1, matrix2), dim=-1)

# y = torch.view_as_complex(combined_matrix)
# y = y.unsqueeze(0)
# print("Combined Matrix:\n", combined_matrix)
# print("Shape of Combined Matrix:", y.shape)
# print("Combined Matrix:\n", y)


# class father:
#     def __init__(self):
#         self.name = 'father'
#         self.age = 40

#     def forward(self, x):
#         print(self.name)

#     def print_age(self):
#         print(self.age)
        
# class son(father):
#     def __init__(self, name, age):
#         super().__init__()
#         # self.name = 'son'
#         self.age = 20
        
#     def forward(self):
#         print(self.name)
        
#     def __call__(self, x):
#         print(x)
#         self.forward()
        
# a = son("abc", 30)
# a("222")

import numpy as np

# 假设损失值 tensor 的形状为 (4, 1, 320, 320)
loss_tensor = np.random.random((4, 1, 320, 320))  # 这里使用随机生成的示例数据

# 计算每个 batch 的平均损失值
batch_losses = []

for batch_idx in range(loss_tensor.shape[0]):
    # 获取当前 batch 的损失值张量切片
    batch_loss_slice = loss_tensor[batch_idx, 0, :, :]
    
    # 计算损失值的平均值
    batch_loss = np.mean(batch_loss_slice)
    batch_losses.append(batch_loss)

# 打印每个 batch 的平均损失值
for batch_idx, loss in enumerate(batch_losses):
    print(f"Batch {batch_idx + 1}: Average Loss = {loss}")