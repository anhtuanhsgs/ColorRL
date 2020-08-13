from .basic_modules import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .conv_lstm import ConvLSTMCell
from .conv_gru import ConvGRUCell


class UNet3D (nn.Module):
    def __init__(self, in_ch, features, out_ch, split=1, level=0, multi=1, is3D=False):
        super(UNet3D, self).__init__()
        self.multi = multi
        self.name = "UNet3D"

        self.Maxpool = nn.MaxPool3d (kernel_size=(2,2,2), stride=(2,2,2))

        if is3D:
            self.fuse_in = nn.Sequential (
                nn.Conv3d (in_channels=in_ch, out_channels=features[0], kernel_size=3, padding=1),
                nn.ReLU (),
                )
        else:
            self.fuse_in = FuseIn2(in_ch, features[0] // 2, split=split, is3D=is3D)

        kernel_size = 3

        if is3D:
            kernel_size = (3, 3, 3)

        self.Conv1 = Residual_Conv(in_ch=features[0], out_ch=features[0], kernel_size=kernel_size, is3D=is3D)
        self.Conv2 = Residual_Conv(in_ch=features[0], out_ch=features[1], kernel_size=kernel_size, is3D=is3D)
        self.Conv3 = Residual_Conv(in_ch=features[1], out_ch=features[2], kernel_size=kernel_size, is3D=is3D)
        self.Conv4 = Residual_Conv(in_ch=features[2], out_ch=features[3], kernel_size=kernel_size, is3D=is3D)
        self.Conv5 = Residual_Conv(in_ch=features[3], out_ch=features[4], kernel_size=kernel_size, is3D=is3D)

        self.Up5 = UpConv(in_ch=features[4], out_ch=features[3], is3D=is3D)
        self.UpConv5 = Residual_Conv(in_ch=features[3] * 2, out_ch=features[3], is3D=is3D)

        self.Up4 = UpConv(in_ch=features[3], out_ch=features[2], is3D=is3D)
        self.UpConv4 = Residual_Conv(in_ch=features[2] * 2, out_ch=features[2], is3D=is3D)

        self.Up3 = UpConv(in_ch=features[2], out_ch=features[1], is3D=is3D)
        self.UpConv3 = Residual_Conv(in_ch=features[1] * 2, out_ch=features[1], kernel_size=kernel_size, is3D=is3D)

        self.Up2 = UpConv(in_ch=features[1], out_ch=features[0], is3D=is3D)
        self.UpConv2 = Residual_Conv(in_ch=features[0] * 2, out_ch=features[0], kernel_size=kernel_size, is3D=is3D)

    def forward(self, x):
        # encoding path
        x = self.fuse_in(x)
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # ------------------------------
        # decoding + concat path
        # ------------------------------

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.UpConv2(d2)

        _ret = [d5, d4, d3, d2]

        if self.multi==1:
            return d2
        return _ret [-self.multi:][::-1]

        if self.level==0:
            return d2
        if self.level==1:
            return d3;


class UNet2D (nn.Module):
    def __init__(self, in_ch, features, out_ch, split=1, level=0, multi=1):
        super(UNet2D, self).__init__()
        self.multi = multi
        self.name = "UNet2D"

        self.Maxpool = nn.MaxPool2d (kernel_size=(2,2), stride=(2,2))

        # self.fuse_in = nn.Sequential (
        #     nn.Conv2d (in_channels=in_ch, out_channels=features[0], kernel_size=3, padding=1),
        #     nn.ReLU (),
        #     )
        
        self.fuse_in = FuseIn2(in_ch, features[0] // 2, split=split, is3D=False)

        kernel_size = 3


        self.Conv1 = Residual_Conv(in_ch=features[0], out_ch=features[0], kernel_size=kernel_size, is3D=False)
        self.Conv2 = Residual_Conv(in_ch=features[0], out_ch=features[1], kernel_size=kernel_size, is3D=False)
        self.Conv3 = Residual_Conv(in_ch=features[1], out_ch=features[2], kernel_size=kernel_size, is3D=False)
        self.Conv4 = Residual_Conv(in_ch=features[2], out_ch=features[3], kernel_size=kernel_size, is3D=False)
        self.Conv5 = Residual_Conv(in_ch=features[3], out_ch=features[4], kernel_size=kernel_size, is3D=False)

        self.Up5 = UpConv(in_ch=features[4], out_ch=features[3], is3D=False)
        self.UpConv5 = Residual_Conv(in_ch=features[3] * 2, out_ch=features[3], is3D=False)

        self.Up4 = UpConv(in_ch=features[3], out_ch=features[2], is3D=False)
        self.UpConv4 = Residual_Conv(in_ch=features[2] * 2, out_ch=features[2], is3D=False)

        self.Up3 = UpConv(in_ch=features[2], out_ch=features[1], is3D=False)
        self.UpConv3 = Residual_Conv(in_ch=features[1] * 2, out_ch=features[1], kernel_size=kernel_size, is3D=False)

        self.Up2 = UpConv(in_ch=features[1], out_ch=features[0], is3D=False)
        self.UpConv2 = Residual_Conv(in_ch=features[0] * 2, out_ch=features[0], kernel_size=kernel_size, is3D=False)

    def forward(self, x):
        # encoding path
        x = self.fuse_in(x)
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # ------------------------------
        # decoding + concat path
        # ------------------------------

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.UpConv2(d2)

        _ret = [d5, d4, d3, d2]

        if self.multi==1:
            return d2
        return _ret [-self.multi:][::-1]

        if self.level==0:
            return d2
        if self.level==1:
            return d3;

def test_models ():
    FEATURES = [4, 8, 16, 32, 64]
    model = UNet3D (in_ch=6, features=FEATURES, out_ch=2, split=3, is3D=True)
    x = torch.zeros ((1,6,64,128,128), dtype=torch.float32)
    print (x.shape)
    x = model (x)
    print (x.shape)