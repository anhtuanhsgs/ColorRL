from basic_modules import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

        # decoding + concat path
        d5 = self.Up5(x5)
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

class AttU_Net2(nn.Module):
    def __init__(self,in_ch, features, out_ch, split=1):
        super(AttU_Net2,self).__init__()
        self.name = "AttU_Net"
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fuse_in = FuseIn2 (in_ch, features[0] // 2, split=split)
        self.Conv1 = Residual_Conv (in_ch=features[0],out_ch=features[0])
        self.Conv2 = Residual_Conv (in_ch=features[0],out_ch=features[1])
        self.Conv3 = Residual_Conv (in_ch=features[1],out_ch=features[2])
        self.Conv4 = Residual_Conv (in_ch=features[2],out_ch=features[3])
        self.Conv5 = Residual_Conv (in_ch=features[3],out_ch=features[4])

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

        # decoding + concat path
        d5 = self.Up5(x5)
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

def test_models ():
    FEATURES = [32, 64, 128, 256, 512]
    model = AttU_Net2 (in_ch=5, features=FEATURES, out_ch=2, split=3)
    x = torch.zeros ((1,5,176,176), dtype=torch.float32)
    print (x.shape)
    x = model (x)
    print (x.shape)
