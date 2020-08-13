from .basic_modules import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .conv_lstm import ConvLSTMCell
from .conv_gru import ConvGRUCell

class AttU_Net(nn.Module):
    def __init__(self,in_ch, features, out_ch, split=1):
        super(AttU_Net,self).__init__()
        self.name = "AttU_Net"
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fuse_in = FuseIn (in_ch, features[0] // 2, split=split)
        self.Conv1 = Residual_Conv (in_ch=features[0],out_ch=features[0])
        self.Conv2 = Residual_Conv (in_ch=features[0],out_ch=features[1])
        self.Conv3 = Residual_Conv (in_ch=features[1],out_ch=features[2])
        self.Conv4 = Residual_Conv (in_ch=features[2],out_ch=features[3])
        self.Conv5 = Residual_Conv (in_ch=features[3],out_ch=features[4])
        self.Conv6 = Residual_Conv (in_ch=features[4],out_ch=features[5])

        self.Up6 =  UpConv(in_ch=features[5],out_ch=features[4])
        self.Att6 = Attention_block(F_g=features[4],F_l=features[4],F_int=features[4]//2)
        self.UpConv6 = Residual_Conv (in_ch=features[5], out_ch=features[4])

        self.Up5 = UpConv(in_ch=features[4],out_ch=features[3])
        self.Att5 = Attention_block(F_g=features[3],F_l=features[3],F_int=features[3]//2)
        self.UpConv5 = Residual_Conv (in_ch=features[4], out_ch=features[3])

        self.Up4 = UpConv(in_ch=features[3],out_ch=features[2])
        self.Att4 = Attention_block(F_g=features[2],F_l=features[2],F_int=features[2]//2)
        self.UpConv4 = Residual_Conv (in_ch=features[3], out_ch=features[2])
        
        self.Up3 = UpConv(in_ch=features[2],out_ch=features[1])
        self.Att3 = Attention_block(F_g=features[1],F_l=features[1],F_int=features[1]//2)
        self.UpConv3 = Residual_Conv (in_ch=features[2], out_ch=features[1])
        
        self.Up2 = UpConv(in_ch=features[1],out_ch=features[0])
        self.Att2 = Attention_block(F_g=features[0],F_l=features[0],F_int=features[0]//2)
        self.UpConv2 = Residual_Conv (in_ch=features[1], out_ch=features[0])



    def forward(self,x):
        # encoding path
        x = self.fuse_in (x)
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        x6 = self.Maxpool(x5)
        x6 = self.Conv6 (x6)
        # decoding + concat path
        d6 = self.Up6 (x6);
        x5 = self.Att6 (g=d6, x=x5)
        d6 = torch.cat ((x5,d6),dim=1)
        d6 = self.UpConv6(d6)

        d5 = self.Up5(d6)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.UpConv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.UpConv2(d2)

        return d2

class AttU_Net3 (nn.Module):
    def __init__ (self, in_ch, features, out_ch, split=1, level=0, multi=1):
        super (AttU_Net3, self).__init__ ()
        self.name = "AttU_Net"
        self.level = level
        self.Maxpool = nn.MaxPool2d (kernel_size=2, stride=2)
        self.fuse_in = FuseIn2(in_ch, features[0] // 2, split=split)
        self.multi = multi

        self.encode_layeres = nn.ModuleList ([Residual_Conv(in_ch=features[0], out_ch=features[0])])
        for i in range (len (features) - 1):
            self.encode_layeres.append (
                Residual_Conv(in_ch=features[i], out_ch=features[i + 1])
            )

        self.decode_layers = nn.ModuleList ([])
        for i in range (len (features) -1, 0, -1):
            up_i = UpConv (in_ch=features[i], out_ch=features[i-1])
            att_i = Attention_block(F_g=features[i-1], F_l=features[i-1], F_int=features[i-1] // 2)
            upConv_i = Residual_Conv(in_ch=features[i-1] * 2, out_ch=features[i-1])
            self.decode_layers.extend ([up_i, att_i, upConv_i])

    def forward (self, x):
        x = self.fuse_in (x)
        xs = []
        xs.append (self.encode_layeres [0] (x))
        # Encode
        for i in range (1, len (self.encode_layeres)):
            x = self.encode_layeres [i] (self.Maxpool (xs[-1])) 
            xs.append ( x )

        # Decode 
        ds = []
        level = 0
        for i in range (0, len (self.decode_layers), 3):
            ds.append (self.decode_layers[i] (xs[-level-1]))
            xs[-level-1] = self.decode_layers[i+1] (g=ds[-1], x=xs[-level-2])
            ds[-1] = torch.cat ((xs[-level-1], ds[-1]), dim=1)
            ds[-1] = self.decode_layers [i+2] (ds[-1])
            level += 1
        
        ret = []
        for i in range (self.multi):
            ret += [ds [-self.level-i-1]]
        if self.multi == 1:
            return ds[-1]
        return ret


class AttU_Net2(nn.Module):
    def __init__(self, in_ch, features, out_ch, split=1, level=0, multi=1, is3D=False):
        super(AttU_Net2, self).__init__()
        self.multi = multi
        self.name = "AttU_Net"
        self.level = level
        if is3D:
            self.Maxpool = nn.MaxPool3d (kernel_size=(1,2,2), stride=(1,2,2))
        else:
            self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)


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
        self.Conv3 = Residual_Conv(in_ch=features[1], out_ch=features[2], is3D=is3D)
        self.Conv4 = Residual_Conv(in_ch=features[2], out_ch=features[3], is3D=is3D)
        self.Conv5 = Residual_Conv(in_ch=features[3], out_ch=features[4], is3D=is3D)

        self.Up5 = UpConv(in_ch=features[4], out_ch=features[3], is3D=is3D)
        self.Att5 = Attention_block(F_g=features[3], F_l=features[3], F_int=features[3] // 2, is3D=is3D)
        self.UpConv5 = Residual_Conv(in_ch=features[3] * 2, out_ch=features[3], is3D=is3D)

        self.Up4 = UpConv(in_ch=features[3], out_ch=features[2], is3D=is3D)
        self.Att4 = Attention_block(F_g=features[2], F_l=features[2], F_int=features[2] // 2, is3D=is3D)
        self.UpConv4 = Residual_Conv(in_ch=features[2] * 2, out_ch=features[2], is3D=is3D)

        self.Up3 = UpConv(in_ch=features[2], out_ch=features[1], is3D=is3D)
        self.Att3 = Attention_block(F_g=features[1], F_l=features[1], F_int=features[1] // 2, is3D=is3D)
        self.UpConv3 = Residual_Conv(in_ch=features[1] * 2, out_ch=features[1], kernel_size=kernel_size, is3D=is3D)

        self.Up2 = UpConv(in_ch=features[1], out_ch=features[0], is3D=is3D)
        self.Att2 = Attention_block(F_g=features[0], F_l=features[0], F_int=features[0] // 2, is3D=is3D)
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
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
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
    FEATURES = [16, 32, 64, 128, 256]
    model = AttU_Net2 (in_ch=5, features=FEATURES, out_ch=2, split=3, is3D=True)
    x = torch.zeros ((1,5,7,64,64), dtype=torch.float32)
    print (x.shape)
    x = model (x)
    print (x.shape)