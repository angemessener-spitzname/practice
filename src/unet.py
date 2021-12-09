import torch
import torch.nn as nn
from torchsummary import summary

def conv3x3(in_channels, out_channels, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding='same')


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution without padding"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

def up_conv2x2(in_channels,out_channels):
    """2x2 deconvolution with padding"""
    return nn.ConvTranspose2d(in_channels,out_channels,kernel_size=2,stride=2)

def max_pool2x2():
    return nn.MaxPool2d(kernel_size=2,stride=2)

class UNet(nn.Module):
    def __init__(self,class_num=2):
        super(UNet, self).__init__()
        # downsample stage
        self.conv_1 = nn.Sequential(conv3x3(1,64),conv3x3(64,64))
        self.conv_2 = nn.Sequential(conv3x3(64,128),conv3x3(128,128))
        self.conv_3 = nn.Sequential(conv3x3(128,256),conv3x3(256,256))
        self.conv_4 = nn.Sequential(conv3x3(256,512),conv3x3(512,512))
        self.conv_5 = nn.Sequential(conv3x3(512,1024),conv3x3(1024,1024))
        self.maxpool = max_pool2x2()
        
        # upsample stage
        self.up_conv_4 = nn.Sequential(up_conv2x2(1024,512))
        self.conv_6 = nn.Sequential(conv3x3(1024,512),conv3x3(512,512))
        self.up_conv_3 = nn.Sequential(up_conv2x2(512,256))
        self.conv_7 = nn.Sequential(conv3x3(512,256),conv3x3(256,256))
        self.up_conv_2 = nn.Sequential(up_conv2x2(256,128))
        self.conv_8 = nn.Sequential(conv3x3(256,128),conv3x3(128,128))
        self.up_conv_1 = nn.Sequential(up_conv2x2(128,64))
        self.conv_9 = nn.Sequential(conv3x3(128,64),conv3x3(64,64))
        self.result = conv1x1(64,2)
    
    def _concat(self,tensor1,tensor2):
        tensor1,tensor2 = (tensor1,tensor2) if tensor1.size()[3]>=tensor2.size()[3] else (tensor2,tensor1)
        crop_val = int((tensor1.size()[3]-tensor2.size()[3])/2)
        tensor1 = tensor1[:, :, crop_val:tensor1.size()[3]-crop_val
                      , crop_val:tensor1.size()[3]-crop_val]
        return torch.cat((tensor1,tensor2),1)

    def forward(self,x):
        c1 = self.conv_1(x)
        c2 = self.conv_2(self.maxpool(c1))
        c3 = self.conv_3(self.maxpool(c2))
        c4 = self.conv_4(self.maxpool(c3))

        c5 = self.conv_5(self.maxpool(c4))
        u6 = self.up_conv_4(c5)
        u6 = self._concat(u6,c4)

        c6 = self.conv_6(u6)
        u7 = self.up_conv_3(c6)
        u7 = self._concat(u7,c3)

        c7 = self.conv_7(u7)
        u8 = self.up_conv_2(c7)
        u8 = self._concat(u8,c2)

        c8 = self.conv_8(u8)
        u9 = self.up_conv_1(c8)
        u9 = self._concat(u9,c1)

        c9 = self.conv_9(u9)
        # result
        out = self.result(c9)
        return out


if __name__ == '__main__':
    ut = UNet(2)
    summary(ut,(1,256,256))