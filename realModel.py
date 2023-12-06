import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import complexLayers as cl

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1, bias=True, dilation=1, groups=1, stride=1),
            nn.BatchNorm2d(mid_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=1, bias=True, dilation=1, groups=1, stride=1),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)
    
class Down_DoubleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, Conv2d_kernel_size=3, Pooling_kernel_size=2):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=Pooling_kernel_size, stride=None, padding=0,
                         ceil_mode=False,count_include_pad=True, divisor_override=None),
            DoubleConv(in_channels, out_channels, mid_channels= None, kernel_size=Conv2d_kernel_size)
        )

    def forward(self, x):
        return self.pool_conv(x)

class DoubleTransConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, mid_channels, kernel_size=kernel_size, stride=1, padding=1,  groups=1, bias=True, dilation=1),
            nn.BatchNorm2d(mid_channels,eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1,  groups=1, bias=True, dilation=1),
            nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU()
        )
    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, Conv2d_kernel_size=3, ConvTrans_kernel_size=2):
        super().__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.Tconv = DoubleTransConv(in_channels, out_channels, mid_channels=in_channels // 2, kernel_size=Conv2d_kernel_size)
        else: 
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=ConvTrans_kernel_size, stride=2, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')
            self.Tconv = DoubleTransConv(in_channels, out_channels, mid_channels=None, kernel_size=Conv2d_kernel_size)
            
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY //2])
        x = torch.cat([x2, x1], dim=1)
        return self.Tconv(x)
            
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1):
        super(OutConv, self).__init__()
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
    def forward(self, x):
        return self.out(x)
                
class RealToFDeFNet(nn.Module):
    def __init__(self, n_in_channels, n_out_channels, bilinear = False):
        super(RealToFDeFNet, self).__init__()
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.bilinear = bilinear
    
        self.inc1 = DoubleConv(in_channels=n_in_channels, out_channels=16, mid_channels=None, kernel_size=3)
        self.down1 = Down_DoubleConv2d(in_channels=16, out_channels=32, Conv2d_kernel_size=3, Pooling_kernel_size=2)
        self.down2 = Down_DoubleConv2d(in_channels=32, out_channels=64, Conv2d_kernel_size=3, Pooling_kernel_size=2)
        self.down3 = Down_DoubleConv2d(in_channels=64, out_channels=128, Conv2d_kernel_size=3, Pooling_kernel_size=2)
        
        factor = 2 if bilinear else 1
        self.down4 = Down_DoubleConv2d(in_channels=128, out_channels=256 // factor, Conv2d_kernel_size=3, Pooling_kernel_size=2)
        
        self.feature = DoubleConv(in_channels=256 // factor, out_channels=256 // factor, mid_channels=None, kernel_size=3)
        
        self.up1 = Up(in_channels=256, out_channels=128 // factor, bilinear=self.bilinear, Conv2d_kernel_size=3, ConvTrans_kernel_size=2)
        self.up2 = Up(in_channels=128, out_channels=64 // factor, bilinear=self.bilinear, Conv2d_kernel_size=3, ConvTrans_kernel_size=2)
        self.up3 = Up(in_channels=64, out_channels=32 // factor, bilinear=self.bilinear, Conv2d_kernel_size=3, ConvTrans_kernel_size=2)
        self.up4 = Up(in_channels=32, out_channels=16, bilinear=self.bilinear, Conv2d_kernel_size=3, ConvTrans_kernel_size=2)
        self.outc1 = OutConv(in_channels=16, out_channels=n_out_channels, kernel_size=1)
    
    def forward(self, x):
        i1 = self.inc1(x)
        d1 = self.down1(i1)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        f = self.feature(d4)
        
        u1 = self.up1(f, d3)
        u2 = self.up2(u1, d2)
        u3 = self.up3(u2, d1)
        u4 = self.up4(u3, i1)
        out = self.outc1(u4)
        
        return out