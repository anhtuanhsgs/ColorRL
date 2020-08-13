from .basic_modules import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .conv_lstm import ConvLSTMCell
from .conv_gru import ConvGRUCell
from .att_unet import AttU_Net, AttU_Net2

class ASPPAttU_Net (nn.Module):
    def __init__(self,in_ch, features, out_ch, atrous_rates = [6, 12, 18], split=1):
        super(ASPPAttU_Net,self).__init__()
        self.name = "ASPPAttU_Net"
        self.att_unet = AttU_Net (in_ch, features, out_ch, split=split)
        self.aspp = ASPP (features[0], features[0], atrous_rates)

    def forward(self,x):
        # encoding path
        x = self.att_unet (x)
        aspp = self.aspp (x)

        return aspp

class ASPPAttU_Net2 (nn.Module):
    def __init__(self,in_ch, features, out_ch, atrous_rates = [6, 12, 18], split=1):
        super(ASPPAttU_Net2,self).__init__()
        self.name = "ASPPAttU_Net2"
        self.att_unet2 = AttU_Net2 (in_ch, features, out_ch, split=split)
        self.aspp = ASPP (features[0], features[0], atrous_rates, conv="ConvInRelu")

    def forward(self,x):
        # encoding path
        x = self.att_unet2 (x)
        aspp = self.aspp (x)

        return aspp
