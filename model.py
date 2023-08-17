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
        self.conv2 = conv(out_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.bn = bn(out_channels) #????
        self.relu = relu()      
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)

        return x
    
class Down(Encoder):
    def __init__(self, input_channels, out_channels, kernel_size, stride, padding = None, complex = False, padding_mode="zeors", dilation=1, groups=1, bias=True, flag: bool = True):
        super().__init__(input_channels, out_channels, kernel_size, stride, padding, complex, padding_mode, dilation, groups, bias)
        self.flag = flag
        
    def forward(self, x):
        if self.flag:
            x = cl.complex_max_pool2d(x, (2,2), (2,2))
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x
        
    
class Decoder(nn.Module):
    def __init__(self, input_channels, mid_channels, out_channels, kernel_size, stride, padding=(0,0), complex =False):
        super().__init__()
        
        tconv = cl.ComplexConvTranspose2d
        bn = cl.ComplexBatchNorm2d
        relu = cl.ComplexReLU
        conv = cl.ComplexConv2d
        
        self.conv = conv(input_channels, mid_channels, kernel_size, stride, padding)  
        self.conv2 = conv(mid_channels, out_channels, kernel_size, stride, padding)  
        self.bn = bn(out_channels)
        self.relu = relu()
       
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x
    
class Up(Decoder):
    def __init__(self, input_channels, mid_channels, out_channels, kernel_size, stride, padding=(0,0), complex =False, flag: bool = True):
        super().__init__(input_channels, mid_channels, out_channels, kernel_size, stride, padding, complex) 
        self.flag = flag
        tconv = cl.ComplexConvTranspose2d
        bn = cl.ComplexBatchNorm2d
        relu = cl.ComplexReLU
        conv = cl.ComplexConv2d
        
        self.conv = conv(mid_channels, input_channels, kernel_size, stride, padding)  
        self.conv2 = conv(input_channels, out_channels, kernel_size, stride, padding)  
        self.bn1 = bn(input_channels)
        self.bn2 = bn(out_channels)
        
        self.tconv = tconv(input_channels, input_channels, kernel_size, (2,2), (1,1), (1,1))
        
    def forward(self, x1, x2):
        if self.flag:
            x1 = self.tconv(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
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
        
        
        # for i in range(self.model_length):
        #     module = Encoder(input_channels = self.encoder_channels[i],
        #                      out_channels = self.encoder_channels[i + 1],
        #                      kernel_size=self.encoder_kernel_sizes[i],
        #                      stride=self.encoder_strides[i],
        #                      padding=self.encoder_paddings[i],
        #                      complex=complex,
        #                      padding_mode=padding_mode)
        #     self.add_module("encoder_{}".format(i), module)
        #     self.encoders.append(module)
            
        left_layer1 = Encoder(1,64,3,1,1,True,padding_mode,1,1,True)
        self.add_module("left_layer1", left_layer1)
        self.encoders.append(left_layer1)
        
        left_layer2 = Down(64,128,3,1,1,True,padding_mode,1,1,True)
        self.add_module("left_layer2", left_layer2)
        self.encoders.append(left_layer2)
        
        left_layer3 = Down(128,256,3,1,1,True,padding_mode,1,1,True)
        self.add_module("left_layer3", left_layer3)
        self.encoders.append(left_layer3)
        
        left_layer4 = Down(256,512,3,1,1,True,padding_mode,1,1,True)
        self.add_module("left_layer4", left_layer4)
        self.encoders.append(left_layer4)
            
            
        bottom_layer = Down(512,512,3,1,1,True,padding_mode,1,1,True)
        self.add_module("bottom_layer", bottom_layer)
        self.encoders.append(bottom_layer)
        
        
        self.decoders = []
        
        # for i in range(self.model_length):
        #     module = Decoder(input_channels = self.decoder_in_channels[i],
        #                      mid_channels = self.decoder_mid_channels[i],
        #                      out_channels = self.decoder_out_channels[i],
        #                      kernel_size=self.decoder_kernel_sizes[i],
        #                      stride=self.decoder_strides[i],
        #                      padding=self.decoder_paddings[i],
        #                      complex=complex)
        #     self.add_module("decoder_{}".format(i), module)
        #     self.decoders.append(module)
            
        right_layer4 = Up(512,1024,256,3,1,1,True,True)
        self.add_module("right_layer4", right_layer4)
        self.decoders.append(right_layer4)
        
        right_layer3 = Up(256,512,128,3,1,1,True,True)
        self.add_module("right_layer3", right_layer3)
        self.decoders.append(right_layer3)
        
        right_layer2 = Up(128,256,64,3,1,1,True,True)
        self.add_module("right_layer2", right_layer2)
        self.decoders.append(right_layer2)
        
        right_layer1 = Up(64,128,1,3,1,1,True,True)
        self.add_module("right_layer1", right_layer1)
        self.decoders.append(right_layer1)
            
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
            # print("model.py:x{}".format(i), x.shape)
            x = encoder(x)
        # print(x.shape)
        
        p = x
        
 
        for i, decoder in enumerate(self.decoders):
            p = decoder(p, xs[4 -i])
            # print("model.py:p{}".format(i), p.shape)
            if i == self.model_length - 1:
                break
            
            # p = torch.cat([p, xs[self.model_length - 1 -i]], dim=1) 
        
            
        return p
                    
    def set_size(self, base_channel, model_depth=8, input_channels=2):
        if model_depth == 8:
            self.encoder_channels = [input_channels,
                                     base_channel,
                                     base_channel * 2,
                                     base_channel * 4,
                                     base_channel * 8]
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
            self.decoder_in_channels = [512,
                                     1024,
                                     512,
                                     256,
                                     128]
            self.decoder_mid_channels = [512,
                                        512,
                                        256,
                                        128,
                                        64] 
            self.decoder_out_channels = [512,
                                        256,
                                        128,
                                        64,
                                        64]   
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
                                     
                                     
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.conv1 = cl.ComplexConv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = cl.ComplexConv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        y = self.conv1(x)
        y = cl.complex_relu(y)
        y = self.conv2(y)
        return cl.complex_relu(x + y)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()

        conv = cl.ComplexConv2d
        bn = cl.ComplexBatchNorm2d   
        tconv = cl.ComplexConvTranspose2d
        
        self.conv1 = conv(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = bn(32, track_running_stats=False)
        self.conv2 = conv(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = bn(64, track_running_stats=False)
        self.tconv1 = tconv(64, 32, kernel_size=4, stride=2, padding=1)
        self.tconv2 = tconv(32, 1, kernel_size=4, stride=2, padding=1)
        
        self.rblock1 = ResidualBlock(32)
        self.rblock2 = ResidualBlock(64)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = cl.complex_relu(x)
        x = cl.complex_max_pool2d(x, kernel_size=2, stride=2)
        x = self.rblock1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = cl.complex_relu(x)
        x = cl.complex_max_pool2d(x, kernel_size=2, stride=2)
        x = self.rblock2(x)
        
        x = self.tconv1(x)
        x = cl.complex_relu(x)
        x = self.tconv2(x)
        x = cl.complex_relu(x)
        
        return x
    
         
        
                                          