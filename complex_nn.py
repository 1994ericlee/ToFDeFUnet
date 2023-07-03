import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ComplexConv2d(nn.module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self, x):
        
    