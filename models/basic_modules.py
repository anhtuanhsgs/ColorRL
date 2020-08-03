import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math, time
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable
from torch.nn.modules.utils import _single, _pair, _triple


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
    def __init__ (self, in_ch, out_ch, kernel_size=3, bias=True, is3D=False):
        super (ConvInELU, self).__init__ ()
        if is3D:

            if kernel_size == 3:
                kernel_size = (1,3,3)
                padding = (0, 1, 1)
            elif kernel_size == (3,3,3):
                padding = (1,1,1)

            self.module = nn.Sequential (
                nn.Conv3d (in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=bias),
                nn.InstanceNorm3d (out_ch),
                nn.ELU ())
        else:
            self.module = nn.Sequential (
                nn.Conv2d (in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size//2, bias=bias),
                INELU (out_ch))

    def forward (self, x):
        x = self.module (x)
        return x

class ConvInReLU (nn.Module):
    def __init__ (self, in_ch, out_ch, kernel_size, stride, padding, dilation):
        super (ConvInReLU, self).__init__ ()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False),
            nn.InstanceNorm2d (out_ch), 
            nn.ReLU(),
        )

    def forward (self, x):
        return self.conv (x)

class DoubleConv (nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            ConvInELU (in_ch, out_ch),
            ConvInELU (out_ch, out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Residual_Conv (nn.Module):
    def __init__ (self, in_ch, out_ch, kernel_size=3, bias=False, is3D=False):
        super (Residual_Conv, self).__init__ ()
        self.conv1 = ConvInELU (in_ch, out_ch, kernel_size=kernel_size, is3D=is3D)
        self.conv2 = ConvInELU (out_ch, out_ch//2, kernel_size=kernel_size, is3D=is3D)
        self.conv3 = ConvInELU (out_ch//2, out_ch, kernel_size=kernel_size, is3D=is3D)

    def forward (self, x):
        _in = x
        x1 = self.conv1 (_in)
        x2 = self.conv2 (x1)
        _out = self.conv3 (x2)
        return x1 + _out

class UpConv (nn.Module):
    def __init__ (self, in_ch, out_ch, kernel_size=2, stride=2, is3D=False):
        super (UpConv, self).__init__ ()
        if is3D:
            kernel_size = (2,2,2)
            stride = (2,2,2)
            self.up = nn.Sequential (
            nn.ConvTranspose3d (in_ch, in_ch, kernel_size=kernel_size, stride=stride),
            ConvInELU (in_ch, out_ch, is3D=is3D),
        )
        else:    
            self.up = nn.Sequential (
                nn.ConvTranspose2d (in_ch, in_ch, 2, stride=2),
                ConvInELU (in_ch, out_ch),
            )

    def forward (self, x):
        x = self.up (x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int, is3D=False):
        super(Attention_block,self).__init__()
        if is3D:
            self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm3d(F_int)
            )
            self.W_x = nn.Sequential(
                nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
                nn.InstanceNorm3d(F_int)
            )
            self.psi = nn.Sequential(
                nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
                nn.InstanceNorm3d(1),
                nn.Sigmoid()
            )
        else:
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
                nn.InstanceNorm2d(F_int)
            )
            self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
                nn.InstanceNorm2d(F_int)
            )
            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
                nn.InstanceNorm2d(1),
                nn.Sigmoid()
            )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class DilatedModule (nn.Module):
    def __init__ (self, in_ch, out_ch, depth, bias=False):
        super(DilatedModule, self).__init__()
        self.layers = nn.ModuleList ()
        self.depth = depth
        for i in range (depth):
            self.layers.append (
                nn.Sequential (
                    nn.Conv2d (out_ch, out_ch, kernel_size=3, dilation=2**i, padding=2**i, bias=bias), 
                    nn.InstanceNorm2d (out_ch), 
                    nn.ELU ()))
        self.last_layer = nn.Sequential (nn.InstanceNorm2d (out_ch), nn.ELU ())

    def forward (self, x):
        layer_rets = []
        for layer in self.layers:
            if len (layer_rets) > 0:
                prevs_sum = torch.sum (torch.cat (layer_rets, 0), 0)
                layer_ret = layer (prevs_sum)
            else:
                layer_ret = layer (x)
            layer_rets.append (layer_ret [None])
        return layer_rets [-1][0]

class ASPP (nn.Module):
    def __init__(self, in_ch, out_ch, rates, conv="ConvInRelu"):
        super(ASPP, self).__init__()
        self.stages = nn.Module()
        if conv == "ConvInRelu":
            self.stages.add_module("c0", ConvInReLU (in_ch, out_ch, 1, 1, 0, 1))
        else:
            self.stages.add_module("c0", ConvELU (in_ch, out_ch))
        for i, rate in enumerate(rates):
            if conv == "ConvInRelu":
                self.stages.add_module(
                    "c{}".format(i + 1),
                    ConvInReLU(in_ch, out_ch, 3, 1, padding=rate, dilation=rate),
                )
            else:
                self.stages.add_module(
                    "c{}".format(i + 1),
                    ConvELU(in_ch, out_ch),
                )

    def forward(self, x):
        return torch.cat([stage(x) for stage in self.stages.children()], dim=1)

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

class FuseIn3D (nn.Module):
    def __init__ (self, in_ch, out_ch, split=1):
        super (FuseIn3D, self).__init__ ()
        self.split = split
        self.raw_path0 = ConvInELU (split, out_ch // 2, kernel_size=(3,3,3), is3D=True)
        self.raw_path1 = ConvInELU (out_ch // 2, out_ch // 2, kernel_size=(3,3,3), is3D=True)
        self.lbl_path0 = ConvInELU (in_ch-split, out_ch // 2, kernel_size=(3,3,3), is3D=True)
        self.lbl_path1 = ConvInELU (out_ch // 2, out_ch // 2, kernel_size=(3,3,3), is3D=True)

    def forward (self, x):
        x_raw = x [:, :self.split, :, :]
        x_lbl = x [:, self.split:, :, :]
        
        x_raw = self.raw_path0 (x_raw)
        x_raw = self.raw_path1 (x_raw)
        
        x_lbl = self.lbl_path0 (x_lbl)
        x_lbl = self.lbl_path1 (x_lbl)

        return torch.cat ([x_raw, x_lbl], dim=1)


class FuseIn2 (nn.Module):
    # modified 01/20
    def __init__ (self, in_ch, out_ch, split=1, rates=[1,6,12,18], is3D=False):
        super (FuseIn2, self).__init__ ()
        self.split = split
        aspp_out_ch = 8
        self.raw_path = ASPP (split, aspp_out_ch, rates=rates, conv="ConvInReLU")
        self.raw_out = ConvInELU ((len (rates)+1) * aspp_out_ch, out_ch)
        self.lbl_path = ASPP (in_ch-split, aspp_out_ch, rates=rates, conv="ConvELU")
        self.lbl_out = ConvInELU ((len (rates)+1) * aspp_out_ch, out_ch)

    def forward (self, x):
        x_raw = x [:, :self.split, :, :]
        x_lbl = x [:, self.split:, :, :]
        
        x_raw = self.raw_path (x_raw)
        x_lbl = self.lbl_path (x_lbl)

        x_raw = self.raw_out (x_raw)
        x_lbl = self.lbl_out (x_lbl)

        return torch.cat ([x_raw, x_lbl], dim=1)


"""
A noisy convolution 2d for pytorch
Adapted from:
- https://raw.githubusercontent.com/Scitator/Run-Skeleton-Run/master/common/modules/NoisyLinear.py
- https://github.com/pytorch/pytorch/pull/2103/files#diff-531f4c06f42260d699f43dabdf741b6d
More details can be found in the paper `Noisy Networks for Exploration`
Original: https://gist.github.com/wassname/001aff274c7c8196055fabfc06cf80c5
"""

class NoisyConv2d(Module):
    """Applies a noisy conv2d transformation to the incoming data:
    More details can be found in the paper `Noisy Networks for Exploration` _ .
    Args:
        in_channels: size of each input sample
        out_channels: size of each output sample
        bias: If set to False, the layer will not learn an additive bias. Default: True
        factorised: whether or not to use factorised noise. Default: True
        std_init: initialization constant for standard deviation component of weights. If None,
            defaults to 0.017 for independent and 0.4 for factorised. Default: None
    Shape:
        - Input: :math:`(N, in_features)`
        - Output: :math:`(N, out_features)`
    Attributes:
        weight: the learnable weights of the module of shape (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
    Examples::
        >>> m = NoisyConv2d(4, 2, (3,1))
        >>> input = torch.autograd.Variable(torch.randn(1, 4, 51, 3))
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=True, stride=1, padding=1, dilation=1, groups=1, factorised=True, std_init=None, gpu_id=0):
        super(NoisyConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.factorised = factorised
        self.weight_mu = Parameter(torch.Tensor(out_channels, in_channels//groups, *kernel_size))
        self.weight_sigma = Parameter(torch.Tensor(out_channels, in_channels//groups, *kernel_size))
        self.gpu_id = gpu_id
        if bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_sigma = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        if not std_init:
            if self.factorised:
                self.std_init = 0.4
            else:
                self.std_init = 0.017
        else:
            self.std_init = std_init
        self.reset_parameters(bias)

    def reset_parameters(self, bias):
        if self.factorised:
            mu_range = 1. / math.sqrt(self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
            if bias:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))
        else:
            mu_range = math.sqrt(3. / self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-mu_range, mu_range)
            self.weight_sigma.data.fill_(self.std_init)
            if bias:
                self.bias_mu.data.uniform_(-mu_range, mu_range)
                self.bias_sigma.data.fill_(self.std_init)

    def scale_noise(self, size):
        with torch.cuda.device (self.gpu_id):
            x = torch.Tensor(size).normal_().cuda ()
            x = x.sign().mul(x.abs().sqrt())
        return x

    def forward(self, input):
        if self.factorised:
            epsilon = None
            for dim in self.weight_sigma.size():
                if epsilon is None:
                    epsilon = self.scale_noise(dim)
                else:
                    epsilon = epsilon.unsqueeze(-1)*self.scale_noise(dim)
            weight_epsilon = Variable(epsilon)
            bias_epsilon = Variable(self.scale_noise(self.out_channels))
        else:
            with torch.cuda.device (self.gpu_id):
                weight_epsilon = Variable(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size).normal_()).cuda ()
                bias_epsilon = Variable(torch.Tensor(self.out_channels).normal_()).cuda ()
        return F.conv2d(input,
                        self.weight_mu + self.weight_sigma.mul(weight_epsilon),
                        self.bias_mu + self.bias_sigma.mul(bias_epsilon),
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups=self.groups
                       )

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias_mu is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

class GCN(nn.Module):
    def __init__(self,in_ch, out_ch,k=7): #out_Channel=21 in paper
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(in_ch, out_ch, kernel_size=(k,1), padding =((k-1)//2,0))
        self.conv_l2 = nn.Conv2d(out_ch, out_ch, kernel_size=(1,k), padding =(0,(k-1)//2))
        self.conv_r1 = nn.Conv2d(in_ch, out_ch, kernel_size=(1,k), padding =((k-1)//2,0))
        self.conv_r2 = nn.Conv2d(out_ch, out_ch, kernel_size=(k,1), padding =(0,(k-1)//2))
        
    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)
        
        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)
        
        x = x_l + x_r
        
        return x

def test_models ():
    # features = [16, 32, 64, 128]
    # atrous_rates = [6, 12, 18]
    # aspp = ASPP (features[0], features[0], atrous_rates)

    with torch.cuda.device (0):
        m = NoisyConv2d(4, 2, (3,1), gpu_id=0).cuda ()
        input = torch.autograd.Variable(torch.randn(1, 4, 51, 3)).cuda ()
        output = m(input)
        print(output.size())