import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import complexLayers as cl

class Encoder(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size, stride, padding = None, complex = False, padding_mode="zeors", dilation=1, groups=1, bias=True):
        super().__init__()
              
        conv = cl.ComplexConv2d
        bn = cl.ComplexBatchNorm2d
        relu = cl.ComplexReLU
       
            
        self.conv = conv(input_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.bn = bn(out_channels)
        self.relu = relu()
        self.conv2 = conv(out_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.conv_down = conv(out_channels, out_channels, kernel_size, (2,2), padding, dilation, groups, bias,  padding_mode)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv_down(x)
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size, stride, padding=(0,0), complex =False):
        super().__init__()
        
        if complex:
            tconv = cl.ComplexConvTranspose2d
            bn = cl.ComplexBatchNorm2d
            relu = cl.ComplexReLU
        else:
            tconv = nn.ConvTranspose2d
            bn = nn.BatchNorm2d
            relu = nn.ReLU
            
        self.tconv = tconv(input_channels, out_channels, kernel_size, stride, padding)
        self.bn = bn(out_channels)
        self.relu = relu()
    
    def forward(self, x):
        x = self.tconv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x
    
class Unet(nn.Module):
    def __init__(self, input_channels=1, complex=False, base_channel=64, model_depth=8, padding_mode="zeros"):
        super().__init__()
        
        # if complex:
        #     model_complexity = int(model_complexity / 1.414) 
        base_channel = 64
        self.complex = complex
        self.set_size(base_channel, model_depth, input_channels)
        self.encoders = []
        self.model_length = model_depth //2
        
        
        for i in range(self.model_length):
            module = Encoder(input_channels = self.encoder_channels[i],
                             out_channels = self.encoder_channels[i + 1],
                             kernel_size=self.encoder_kernel_sizes[i],
                             stride=self.encoder_strides[i],
                             padding=self.encoder_paddings[i],
                             complex=complex,
                             padding_mode=padding_mode)
            self.add_module("encoder_{}".format(i), module)
            self.encoders.append(module)
            
        self.decoders = []
        
        for i in range(self.model_length):
            module = Decoder(input_channels = self.decoder_channels[i],
                             out_channels = self.decoder_channels[i + 1],
                             kernel_size=self.decoder_kernel_sizes[i],
                             stride=self.decoder_strides[i],
                             padding=self.decoder_paddings[i],
                             complex=complex)
            self.add_module("decoder_{}".format(i), module)
            self.decoders.append(module)
            
        if complex:
            conv = cl.ComplexConv2d
        else:
            conv = nn.Conv2d
            
        self.decoders = nn.ModuleList(self.decoders)
        self.encoders = nn.ModuleList(self.encoders)
           
    def forward(self, x):
        xs = []
        for i, encoder in enumerate(self.encoders):
            xs.append(x)
            print("model.py:x{}".format(i), x.shape)
            x = encoder(x)
        print(x.shape)
        
        p = x
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if i == self.model_length - 1:
                break
            
            p = torch.cat([p, xs[self.model_length - 1 -i]], dim=1) 
            
        return p
                    
    def set_size(self, base_channel, model_depth=8, input_channels=2):
        if model_depth == 8:
            self.encoder_channels = [input_channels,
                                     base_channel,
                                     base_channel * 2,
                                     base_channel * 2,
                                     base_channel * 2]
            self.encoder_kernel_sizes = [(3,3),
                                         (3,3),
                                         (3,3),
                                         (3,3)]
            self.encoder_strides = [(1,1),
                                    (1,1),
                                    (1,1),
                                    (1,1)]
                                    
            self.encoder_paddings = [(1,1),
                                     (1,1),
                                     (1,1),
                                     (1,1)]
            self.decoder_channels = [512,
                                     256,
                                     128,
                                     64,
                                     1]
            self.decoder_kernel_sizes = [(3,3),
                                         (3,3),
                                         (3,3),
                                         (3,3),
                                         (3,3),]
            self.decoder_strides = [(1,1),
                                    (1,1),
                                    (1,1),
                                    (1,1),
                                    (1,1)]
            self.decoder_paddings = [(1,1),
                                     (1,1),
                                     (1,1),
                                     (1,1),
                                     (1,1)]
                                     