import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import complex_nn as complex_nn

class Encoder(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size, stride, padding = None, complex = False, padding_mode="zeors"):
        super().__init__()
        if padding is None:
            padding = [(k - 1) // 2 for k in kernel_size] #???
            
        if complex:
            conv = complex_nn.ComplexConv2d
            bn = complex_nn.ComplexBatchNorm2d
            relu = complex_nn.ComplexReLU
        else:
            conv = nn.Conv2d
            bn = nn.BatchNorm2d
            relu = nn.ReLU
            
        self.conv = conv(input_channels, out_channels, kernel_size, stride, padding, padding_mode)
        self.bn = bn(out_channels)
        self.relu = relu(inplace=True)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size, stride, padding=(0,0), complex =False):
        super().__init__()
        
        if complex:
            tconv = complex_nn.ComplexConvTranspose2d
            bn = complex_nn.ComplexBatchNorm2d
            relu = complex_nn.ComplexReLU
        else:
            tconv = nn.ConvTranspose2d
            bn = nn.BatchNorm2d
            relu = nn.ReLU
            
        self.tconv = tconv(input_channels, out_channels, kernel_size, stride, padding)
        self.bn = bn(out_channels)
        self.relu = relu(inplace=True)
    
    def forward(self, x):
        x = self.tconv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x
    
class Unet(nn.Module):
    def __init__(self, input_channels=1, complex=False, model_complexity=45, model_depth=10, padding_mode="zeros"):
        super().__init__()
        
        if complex:
            model_complexity = int(model_complexity / 1.414) 
        
        self.complex = complex
        self.set_size(model_complexity, model_depth, input_channels)
        self.encoders = []
        self.model_length = model_depth //2
        
        
        for i in range(self.model_length):
            module = Encoder(input_channels = self.encoder_channels[i],
                             out_channels = self.encoder_channels[i + 1],
                             kernel_size=self.encoder_kernel_sizes[i],
                             stride=self.encoder_strides,
                             padding=self.encoder_paddings[i],
                             complex=complex,
                             padding_mode=padding_mode)
            self.add_module("encoder_{}".format(i), module)
            self.encoders.append(module)
            
        self.decoders = []
        
        for i in range(self.model_length):
            module = Decoder(input_channels = self.decoder_channels[i] + self.encoder_channels[self.model_length - i],
                             out_channels = self.decoder_channels[i + 1],
                             kernel_size=self.decoder_kernel_sizes[i],
                             stride=self.decoder_strides,
                             padding=self.decoder_paddings[i],
                             complex=complex)
            self.add_module("decoder_{}".format(i), module)
            self.decoders.append(module)
            
        if complex:
            conv = complex_nn.ComplexConv2d
        else:
            conv = nn.Conv2d
            
        self.decoders = nn.ModuleList(self.decoders)
        self.encoders = nn.ModuleList(self.encoders)
           
    def forward(self, x):
        xs = []
        for i, encoder in enumerate(self.encoders):
            xs.append(x)
            print("x{}".format(i), x.shape)
            x = encoder(x)
        print(x.shape)
        
        p = x
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if i == self.model_length - 1:
                break
            
            p = torch.cat([p, xs[self.model_length - 1 -i]], dim=1) 
            
        return p
                    
    def set_size(self, model_complexity, model_depth=20, input_channels=1):
        if model_depth == 10:
            self.encoder_channels = [input_channels,
                                     model_complexity,
                                     model_complexity * 2,
                                     model_complexity * 2,
                                     model_complexity * 2,
                                     model_complexity * 2,]
            self.encoder_kernel_sizes = [(3,3),
                                         (3,3),
                                         (3,3),
                                         (3,3),
                                         (3,3),]
            self.encoder_strides = [(2,2),
                                    (2,2),
                                    (2,2),
                                    (2,2),
                                    (2,2)]
                                    
            self.encoder_paddings = [(1,1),
                                     None,
                                     None,
                                     None,
                                     None]
            self.decoder_channels = [0,
                                     model_complexity * 2,
                                     model_complexity * 2,
                                     model_complexity * 2,
                                     model_complexity * 2,
                                     model_complexity * 2,]
            self.decoder_kernel_sizes = [(3,3),
                                         (3,3),
                                         (3,3),
                                         (3,3),
                                         (3,3),]
            self.decoder_strides = [(2,2),
                                    (2,2),
                                    (2,2),
                                    (2,2),
                                    (2,2)]
            self.decoder_paddings = [(1,1),
                                     (1,1),
                                     (1,1),
                                     (1,1),
                                     (1,1)]
                                     