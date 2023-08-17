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
A = 2
pha = np.pi*2

s = A * np.exp(1j * pha)
b = A * np.cos(pha) + 1j * A * np.sin(pha)
x = np.cos(2*np.pi)
x = np.sin
print(s)
print(x)