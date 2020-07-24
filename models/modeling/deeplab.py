import torch
import torch.nn as nn
import torch.nn.functional as F
from ..modeling.aspp import build_aspp
from ..modeling.decoder import build_decoder

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

class ConvELU (nn.Module):
    def __init__ (self, in_ch, out_ch, kernel_size=3, bias=True):
        super (ConvELU, self).__init__ ()
        self.conv = nn.Sequential(
            nn.Conv2d (in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2, bias=bias),
            nn.ELU ()
        )

    def forward (self, x):
        return self.conv (x)

class ConvInELU (nn.Module):
    def __init__ (self, in_ch, out_ch, kernel_size=3, bias=True):
        super (ConvInELU, self).__init__ ()
        self.module = nn.Sequential (
            nn.Conv2d (in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2, bias=bias),
            INELU (out_ch))

    def forward (self, x):
        x = self.module (x)
        return x

from .backbone import xception

def build_backbone(backbone, input_stride, output_stride, BatchNorm):
    if backbone == 'resnet':
        return resnet.ResNet101(input_stride, output_stride, BatchNorm)
    elif backbone == 'xception':
        return xception.AlignedXception(input_stride, output_stride, BatchNorm)
    elif backbone == 'drn':
        return drn.drn_d_54(BatchNorm)
    elif backbone == 'mobilenet':
        return mobilenet.MobileNetV2(input_stride, output_stride, BatchNorm)
    else:
        raise NotImplementedError

class FuseIn (nn.Module):
    def __init__ (self, in_ch, out_ch, split=1):
        super (FuseIn, self).__init__ ()
        self.split = split
        self.local0 = ConvInELU (split, out_ch, kernel_size=3)
        self.local1 = ConvInELU (out_ch, out_ch, kernel_size=3)
        self.global0 = ConvELU (in_ch-split, out_ch, kernel_size=7)
        self.global1 = ConvELU (out_ch, out_ch, kernel_size=1)
        self.global2 = ConvInELU (out_ch, out_ch, kernel_size=3)

    def forward (self, x):
        x_raw = x [:, :self.split, :, :]
        x_lbl = x [:, self.split:, :, :]
        
        x_raw = self.local0 (x_raw)
        x_raw = self.local1 (x_raw)
        
        x_lbl = self.global0 (x_lbl)
        x_lbl = self.global1 (x_lbl)
        x_lbl = self.global2 (x_lbl)
        return torch.cat ([x_raw, x_lbl], dim=1)

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', input_stride=3, output_stride=16, num_classes=2,
                 freeze_bn=False, split=1, ):
        super(DeepLab, self).__init__()

        self.name = "deeplab"

        if backbone == 'drn':
            output_stride = 8

        
        BatchNorm = nn.GroupNorm
        self.fuseIn = FuseIn (input_stride, 16, split=split)
        self.backbone = build_backbone(backbone, 32, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        input = self.fuseIn (input)
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        critic, actor = self.decoder(x, low_level_feat)

        critic = F.interpolate(critic, size=input.size()[2:], mode='bilinear', align_corners=True)
        actor = F.interpolate(actor, size=input.size()[2:], mode='bilinear', align_corners=True)

        return critic, actor

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, InstanceNorm2d):
                m.eval()
            elif isinstance(m, nn.InstanceNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) \
                        or isinstance(m[1], nn.InstanceNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) \
                        or isinstance(m[1], nn.InstanceNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


def test ():
    model = DeepLab(backbone='xception', input_stride=5, output_stride=16, split=3)
    model.eval()
    input = torch.rand(1, 5, 256, 256)
    output = model(input)
    print(output.size())


