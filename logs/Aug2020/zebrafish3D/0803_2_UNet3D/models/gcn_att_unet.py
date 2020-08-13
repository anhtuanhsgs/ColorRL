from .basic_modules import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .conv_lstm import ConvLSTMCell
from .conv_gru import ConvGRUCell

class GCN_AttU_Net (nn.Module):
    def __init__ (self, in_ch, features, out_ch, split=1, level=0, multi=1):
        super (GCN_AttU_Net, self).__init__ ()
        self.name = "AttU_Net"
        self.level = level
        self.Maxpool = nn.MaxPool2d (kernel_size=2, stride=2)
        self.fuse_in = FuseIn2(in_ch, features[0] // 2, split=split)
        self.multi = multi

        self.encode_layers = nn.ModuleList ([Residual_Conv(in_ch=features[0], out_ch=features[0])])
        base_size  = 31
        for i in range (len (features) - 1):
            layers = []
            layers.append (Residual_Conv(in_ch=features[i], out_ch=features[i + 1]))
            if (base_size > 3):
                layers.append (GCN (features[i+1], features[i+1], k=base_size))
                base_size = base_size // 2
            else:
                layers.append (nn.Identity ())
            self.encode_layers.extend (
                layers
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
        xs.append (self.encode_layers [0] (x))
        # Encode
        for i in range (1, len (self.encode_layers), 2):
            x = self.encode_layers [i] (self.Maxpool (xs[-1])) 
            x = self.encode_layers [i+1] (x)
            xs.append ( x )

        # Decode 
        ds = []
        level = 0
        for i in range (0, len (self.decode_layers), 3):
            ds.append (self.decode_layers[i] (xs[-level-1])) # UPi
            xs[-level-1] = self.decode_layers[i+1] (g=ds[-1], x=xs[-level-2]) # Atti
            ds[-1] = torch.cat ((xs[-level-1], ds[-1]), dim=1) # Cat
            ds[-1] = self.decode_layers [i+2] (ds[-1]) # Residuali
            level += 1
        
        ret = []
        for i in range (self.multi):
            ret += [ds [-self.level-i-1]]

        if self.multi == 1:
            return ds[-1]
        return ret

def test_models ():
    FEATURES = [32, 64, 64, 128]
    model = GCN_AttU_Net (in_ch=5, features=FEATURES, out_ch=2, split=3)
    x = torch.zeros ((1,5,256,256), dtype=torch.float32)
    print (x.shape)
    x = model (x)
    print (x [0].shape)