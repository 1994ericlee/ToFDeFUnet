import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)

class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        
    def forward(self, x):
        #W ∗h = (A∗x−B ∗y)+i(B ∗x+A∗y). 
        # real = self.conv_real(x[:,0,:,:]) - self.conv_imag(x[:,1,:,:])
        # imaginary = self.conv_imag(x[:,1,:,:]) + self.conv_real(x[:,0,:,:])
        # output = torch.stack([real, imaginary], dim=0)
        
class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1):
        super().__init__()
        
        self.conv_real = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
        self.conv_imag = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
    
    def forward(self, x):
        
        real = self.conv_real(x[..., 0]) - self.conv_imag(x[..., 1])
        imaginary = self.conv_imag(x[..., 0]) + self.conv_real(x[..., 1])
        output = torch.stack([real, imaginary], dim=-1)
        
class ComplexBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        
        self.bn_real = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        self.bn_imag = nn.BatchNorm2d(num_features, eps, momentum, affine, track_running_stats)
        
    def forward(self, x):
        real = self.bn_real(x[..., 0])
        imaginary = self.bn_imag(x[..., 1]) 
        output = torch.stack([real, imaginary], dim=-1)
        
        return output
    
class ComplexReLU(nn.Module):
    
    def __init__(self, inplace=False):
        super().__init__()
        
        self.relu = nn.ReLU(inplace)
    def forward(self, x):
        return self.relu(x.real).type(torch.complex64) + 1j * self.relu(x.imag).type(torch.complex64)
    
    