import torch
import torch.nn as nn
import torch.nn.functional as F

class INELU (nn.Module):
    def __init__ (self, out_ch):
        super (INELU, self).__init__ ()
        self.module = nn.Sequential (
                nn.InstanceNorm2d (out_ch),
                nn.ELU ()
            )

    def forward (self, x):
        x = self.module (x)
        return x

class Residual_Conv (nn.Module):
    def __init__ (self, in_ch, out_ch, bias=False):
        super (Residual_Conv, self).__init__ ()
        self.conv1 = nn.Sequential (
            nn.Conv2d (in_ch, out_ch, kernel_size=3, 
                padding=1, bias=bias),
            INELU (out_ch))
        self.conv2 = nn.Sequential (
            nn.Conv2d (out_ch, out_ch//2, kernel_size=3, 
                padding=1, bias=bias),
            INELU (out_ch))
        self.conv3 = nn.Conv2d (out_ch//2, out_ch, kernel_size=3,
            padding=1, bias=bias)

    def forward (self, x):
        _in = x
        x = self.conv1 (_in)
        x = self.conv2 (x)
        _out = self.conv3 (x)
        return _in + _out

class Residual_Conv (nn.Module):
    def __init__ (self, in_ch, out_ch, bias=False):
        super (Residual_Conv, self).__init__ ()
        self.conv1 = nn.Sequential (
            nn.Conv2d (in_ch, out_ch, kernel_size=3, 
                padding=1, bias=bias),
            INELU (out_ch))
        self.conv2 = nn.Sequential (
            nn.Conv2d (out_ch, out_ch//2, kernel_size=3, 
                padding=1, bias=bias),
            INELU (out_ch))
        self.conv3 = nn.Conv2d (out_ch//2, out_ch, kernel_size=3,
            padding=1, bias=bias)

    def forward (self, x):
        _in = x
        x = self.conv1 (_in)
        x = self.conv2 (x)
        _out = self.conv3 (x)
        return _in + _out

class FusionDown (nn.Module):
    def __init__ (self, in_ch, out_ch, bias=False, kernel_size=3):
        super (FusionDown, self).__init__ ()
        self.conv_in = nn.Sequential (
            nn.Conv2d (in_ch, out_ch, kernel_size=3, stride=2, 
                padding=1, bias=bias),
            INELU (out_ch))
        self.residual = Residual_Conv (out_ch, out_ch, bias=bias)
        self.conv_out = nn.Sequential (
            nn.Conv2d (out_ch, out_ch, kernel_size=kernel_size,
                padding=1, bias=bias),
            INELU (out_ch))

    def forward (self, x):
        x = self.conv_in (x)
        x = self.residual (x)
        x = self.conv_out (x)
        return x

class FusionUp (nn.Module):
    def __init__ (self, in_ch, out_ch, bias=False, kernel_size=3):
        super (FusionUp, self).__init__ ()
        self.conv_in = nn.Sequential (
            nn.Conv2d (in_ch, out_ch, kernel_size=kernel_size, padding=1, bias=bias), 
            INELU (out_ch))
        self.residual = Residual_Conv (out_ch, out_ch, bias=bias)
        self.deconv_out = nn.Sequential (
            nn.ConvTranspose2d (out_ch, out_ch, kernel_size=3, stride=2,
                bias=bias),
            INELU (out_ch))

    def forward (self, x):
        x = self.conv_in (x)
        x = self.residual (x)
        x = self.deconv_out (x)
        N, C, H, W = x.shape
        x = x [:,:,0:H-1, 0:W-1]
        return x

class FusionNet (nn.Module):

    def __init__ (self, in_ch, features, out_ch):
        super (FusionNet, self).__init__ ()
        self.first_layer = nn.Sequential (INELU (in_ch), 
            nn.Conv2d (in_ch, features[0], 3, bias=True, padding=1))
        self.down1 = FusionDown (features[0], features[0])
        self.down2 = FusionDown (features[0], features[1])
        self.down3 = FusionDown (features[1], features[2])
        self.down4 = FusionDown (features[2], features[3])
        self.middle = nn.Dropout (p=0.5)
        self.up4 = FusionUp (features[3], features[2])
        self.up3 = FusionUp (features[2], features[1])
        self.up2 = FusionUp (features[1], features[0])
        self.up1 = FusionUp (features[0], features[0])
        self.final = nn.Conv2d (features[0], out_ch, 3, padding=1, bias=True)


    def forward (self, x):
        x = self.first_layer (x)
        down1 = self.down1 (x)
        down2 = self.down2 (down1)
        down3 = self.down3 (down2)
        down4 = self.down4 (down3)
        middle = self.middle (down4)
        up4 = self.up4 (middle)
        up3 = self.up3 (up4 + down3)
        up2 = self.up2 (up3 + down2)
        up1 = self.up1 (up2 + down1)
        x = up1 + x
        x = self.final (x)
        return x

if __name__ == "__main__":
    FEATURES = [64, 64, 128, 128]
    model = FusionNet (5, FEATURES, 12)
    x = torch.zeros ((1,5,256,256), dtype=torch.float32)
    print (x.shape)
    y = model (x)
    print (y.shape)