# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 18:36:08 2021

@author: XieQi
"""


# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

import torch.nn.functional as F
import torch.nn as nn
import torch


def make_model(args, parent=False):
    return Bicubic(args)


class Bicubic(nn.Module):
    def __init__(self, args):
        super(Bicubic, self).__init__()
        self.r = args.scale[0]
        self.weights = nn.Parameter(torch.randn(1), requires_grad=True)
      

    def forward(self, x):       
        w = self.weights
        w = w
        return  F.interpolate(x, scale_factor=self.r, mode='bicubic')
