from model import common

import torch.nn as nn
from e2cnn import nn as en
from e2cnn import gspaces
# from e2cnn.nn.modules.equivariant_module import EquivariantModule

def make_model(args, parent=False):
    return EDSR_e2cnn(args)


class ResBlock(nn.Module):
    def __init__(self,normal_type, kernel_size, act, res_scale=0.1,bn = False):
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(en.R2Conv(normal_type, normal_type, kernel_size=kernel_size, padding=kernel_size//2, bias=True))
            if bn:
                m.append(en.InnerBatchNorm(normal_type))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)*(self.res_scale)
        res += x

        return res

class EDSR_e2cnn(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(EDSR_e2cnn, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = int(args.kernel_size)    
        scale = args.scale[0]

        # url_name = 'r{}f{}x{}'.format(n_resblocks, n_feats, scale)
        # if url_name in url:
        #     self.url = url[url_name]
        # else:
        self.url = None
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        tranNum = args.tranNum  
        self.r2_act = gspaces.Rot2dOnR2(tranNum)

        # define head module
        begin_type = en.FieldType(self.r2_act, args.n_colors*[self.r2_act.trivial_repr])
        normal_type = en.FieldType(self.r2_act, n_feats*[self.r2_act.regular_repr]) 
        act = en.ReLU(normal_type, inplace=True)
        
        m_head = [en.R2Conv(begin_type, normal_type, kernel_size=kernel_size, padding=kernel_size//2, bias=True)]

        # define body module
        m_body = [
            ResBlock(
                normal_type, kernel_size, act=act, res_scale=args.res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(en.R2Conv(normal_type, normal_type, kernel_size=kernel_size, padding=kernel_size//2, bias=True))

        # define tail module
        m_tail = [
            common.Upsampler(conv, scale, n_feats*tranNum, act=False),
            conv(n_feats*tranNum, args.n_colors, kernel_size)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)
        self.begin_type = begin_type

    def forward(self, x):
        x = self.sub_mean(x)
        x = en.GeometricTensor(x, self.begin_type)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res.tensor)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

