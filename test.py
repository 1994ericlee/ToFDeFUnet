import torch

# 创建两个矩阵
matrix1 = torch.tensor([[1, 2, 3],
                        [4, 5, 6]])
matrix2 = torch.tensor([[7, 8, 9],
                        [10, 11, 12]])

# 将两个矩阵合并成一个两通道的矩阵
combined_matrix = torch.stack((matrix1, matrix2), dim=0)

print("Combined Matrix:\n", combined_matrix)
print("Shape of Combined Matrix:", combined_matrix.shape)