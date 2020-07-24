import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
from basic_modules import *
from att_unet import AttU_Net2, AttU_Net
from aspp_att_unet import ASPPAttU_Net2, ASPPAttU_Net

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, bias=True):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, bias=bias, padding=kernel_size//2)

    def forward(self, x):
        x = self.conv(x)
        return x

class SomeModel (nn.Module):
    def __init__(self, last_feat_ch, backbone, out_ch):
        super(SomeModel,self).__init__()
        self.name = backbone.name
        self.backbone=backbone

    def forward (self, x):
        x = self.backbone (x)
        return x

def to_numpy (tensor):
    return tensor.cpu ().numpy ().squeeze ()

def debug (tensor):
    tensor_np = to_numpy (tensor)
    shape = tensor_np.shape
    for i in range (shape [0]):
        print (tensor_np [i])

def get_model (name, input_shape, features, num_actions, split, atrous_rates=[6, 12, 18]):
    if name == "AttUNet":
        model = SomeModel (features [0], AttU_Net (input_shape [0], features, num_actions, split=split), out_ch=num_actions)
    if name == "AttUNet2":
        model = SomeModel (features [0], AttU_Net2 (input_shape [0], features, num_actions, split=split), out_ch=num_actions)
    if name == "ASPPAttUNet2":
        model = SomeModel (features [0] * (len (atrous_rates) + 1), 
                                ASPPAttU_Net2 (input_shape [0], features, num_actions, split=split, atrous_rates=atrous_rates), out_ch=num_actions)
    if name == "ASPPAttUNet":
        model = SomeModel (features [0] * (len (atrous_rates) + 1), 
                                ASPPAttU_Net (input_shape [0], features, num_actions, split=split, atrous_rates=atrous_rates), out_ch=num_actions)
    if name == "DeepLab":
        model = SomeModel (features [0], DeepLab(backbone='xception', input_stride=input_shape [0], output_stride=16, split=split, num_classes=num_actions), out_ch=num_actions)
    return model


def test_models ():
    FEATURES = [32, 64, 128, 256, 512]
    model = get_model ("ASPPAttUNet2", (5,256,256), features=FEATURES, num_actions=2, split=3)
    x = torch.randn ((1,5,256,256), dtype=torch.float32)
    print (x.shape)    
    y = model (x)
    print (y.shape)

test_models ()
    
