# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common
from e2cnn import nn as en
from e2cnn import gspaces
from e2cnn.nn.modules.equivariant_module import EquivariantModule
import torch
import torch.nn as nn


def make_model(args, parent=False):
    return RDN_e2cnn(args)

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, r2_act, kSize=3, tranNum=8):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        in_type = en.FieldType(r2_act, Cin*[r2_act.regular_repr])
        out_type = en.FieldType(r2_act, (G)*[r2_act.regular_repr])
        self.conv = nn.Sequential(*[
            en.R2Conv(in_type, out_type, kernel_size=kSize, padding=kSize//2, bias=True),
            en.ReLU(out_type, inplace=True)
        ])
        self.out_type = en.FieldType(r2_act, (Cin+G)*[r2_act.regular_repr])
    def forward(self, x):
        out = self.conv(x)
        out  = torch.cat((x.tensor, out.tensor), 1)
        return en.GeometricTensor(out, self.out_type)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, r2_act, kSize=3, tranNum=8):
        # RDB(growRate0 = G0, growRate = G, nConvLayers = C, kSize = kernel_size, r2_act=self.r2_act, tranNum=tranNum)
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G, r2_act, kSize=kSize, tranNum=tranNum))
        self.convs = nn.Sequential(*convs)
        # Local Feature Fusion
        
        in_type = en.FieldType(r2_act, (G0 + C*G)*[r2_act.regular_repr])
        out_type = en.FieldType(r2_act, (G0)*[r2_act.regular_repr])
        self.LFF = en.R2Conv(in_type, out_type, kernel_size=kSize, padding=kSize//2, bias=True)#nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x))*0.1 + x

class RDN_e2cnn(nn.Module):
    def __init__(self, args):
        super(RDN_e2cnn, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        kernel_size = int(args.kernel_size)       
        tranNum = args.tranNum        
        self.r2_act = gspaces.Rot2dOnR2(tranNum)
        self.sub_mean = common.MeanShift(args.rgb_range)
        
        begin_type = en.FieldType(self.r2_act, args.n_colors*[self.r2_act.trivial_repr])
        normal_type = en.FieldType(self.r2_act, G0*[self.r2_act.regular_repr]) 
        

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 4),
            'B': (16, 8, 8),
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = en.R2Conv(begin_type, normal_type, kernel_size=kernel_size, padding=kernel_size//2, bias=True)
        self.SFENet2 = en.R2Conv(normal_type, normal_type, kernel_size=kernel_size, padding=kernel_size//2, bias=True)

        # Redidual dense blocks and dense feature fusion
        self.RDBs = nn.ModuleList()
        for i in range(self.D):
            self.RDBs.append(
                RDB(growRate0 = G0, growRate = G, nConvLayers = C, r2_act=self.r2_act, kSize = kernel_size,  tranNum=tranNum)
            )
        G = G*tranNum
        G0 = G0 *tranNum
        # Global Feature Fusion
        
        self.GFF = nn.Sequential(*[
            nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])


        # Up-sampling net
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(r),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")
        self.begin_type = begin_type

    def forward(self, x):
        x = en.GeometricTensor(x, self.begin_type)
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        RDBs_out = []
        for i in range(self.D):
            x = self.RDBs[i](x)
            #print(x.tensor.size())
            RDBs_out.append(x.tensor)
        #print(torch.cat(RDBs_out,1).size())
        x = self.GFF(torch.cat(RDBs_out,1))
        x += f__1.tensor

        return self.UPNet(x)
