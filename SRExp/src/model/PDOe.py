# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 15:58:33 2021

@author: XieQi
"""
import numpy as np
import torch
import torch.nn as nn
#import math
import torch.nn.functional as  F
#import MyLibForSteerCNN as ML
import scipy.io as sio    
import math
from PIL import Image
from math import *

class Fconv_PCA(nn.Module):
    
    def __init__(self,  sizeP, inNum, outNum, tranNum=8, inP = None, padding=None, ifIni=0, bias=True, Smooth = True):
       
        super(Fconv_PCA, self).__init__()
        if inP==None:
            inP = sizeP
        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum
        self.sizeP = sizeP
        Basis = GetBasis_PDOe(sizeP,tranNum,inP, Smooth = Smooth)        
        self.register_buffer("Basis", Basis)#.cuda())
                
        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
#        iniw = torch.randn(outNum, inNum, self.expand, self.Basis.size(3))*0.03
        iniw = Getini_reg(Basis.size(3), inNum, outNum, self.expand)
        self.weights = nn.Parameter(iniw, requires_grad=True)
        if padding == None:
            self.padding = 0
        else:
            self.padding = padding
        

        self.c = nn.Parameter(torch.zeros(1,outNum,1,1), requires_grad=bias)


        #self.register_buffer("filter", torch.zeros(outNum*tranNum, inNum*self.expand, sizeP, sizeP))

    def forward(self, input):
    
        if self.training:
          tranNum = self.tranNum
          outNum = self.outNum
          inNum = self.inNum
          expand = self.expand
          tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)
               
#          for i in range(expand):
#              ind = np.hstack((np.arange(expand-i,expand), np.arange(expand-i) ))
#              tempW[:,i,:,:,...] = tempW[:,i,:,ind,...]
              
          Num = tranNum//expand
          tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:]], dim = 3) for i in range(expand)]   
          tempW = torch.cat(tempWList, dim = 1)              
              
              
          _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, self.sizeP, self.sizeP ])
          _bias = self.c.repeat([1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1])
        else:
          _filter = self.filter
          _bias   = self.bias

  
        output = F.conv2d(input, _filter,
                        padding=self.padding,
                        dilation=1,
                        groups=1)
        return output + _bias
        
    def train(self, mode=True):
        if mode:
            # TODO thoroughly check this is not causing problems
            if hasattr(self, "filter"):
                del self.filter
        elif self.training:
            # avoid re-computation of the filter and the bias on multiple consecutive calls of `.eval()`
            #print('Using Fixed Filter')
            tranNum = self.tranNum
            outNum = self.outNum
            inNum = self.inNum
            expand = self.expand
            tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)
#            for i in range(expand):
#                ind = np.hstack((np.arange(expand-i,expand), np.arange(expand-i) ))
#                tempW[:,i,:,:,...] = tempW[:,i,:,ind,...]
            Num = tranNum//expand
            tempWList = [torch.cat([tempW[:,i*Num:(i+1)*Num,:,-i:,:,:],tempW[:,i*Num:(i+1)*Num,:,:-i,:,:]], dim = 3) for i in range(expand)]   
            tempW = torch.cat(tempWList, dim = 1)
           
            
            _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, self.sizeP, self.sizeP ])
            _bias = self.c.repeat([1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1])
            self.register_buffer("filter", _filter)
            self.register_buffer("bias", _bias)

        return super(Fconv_PCA, self).train(mode)  
    

class Fconv_1X1(nn.Module):
    
    def __init__(self, inNum, outNum, tranNum=8, ifIni=0, bias=True, Smooth = True):
       
        super(Fconv_1X1, self).__init__()

        self.tranNum = tranNum
        self.outNum = outNum
        self.inNum = inNum

                
        if ifIni:
            self.expand = 1
        else:
            self.expand = tranNum
        iniw = Getini_reg(1, inNum, outNum, self.expand)
        self.weights = nn.Parameter(iniw, requires_grad=True)

        self.padding = 0
        self.bias = bias

        if bias:
            self.c = nn.Parameter(torch.zeros(1,outNum,1,1), requires_grad=True)
        else:
            self.c = torch.zeros(1,outNum,1,1)

    def forward(self, input):
        tranNum = self.tranNum
        outNum = self.outNum
        inNum = self.inNum
        expand = self.expand
        # tempW = torch.einsum('ijok,mnak->monaij', self.Basis, self.weights)
        tempW = self.weights.unsqueeze(4).unsqueeze(1).repeat([1,tranNum,1,1,1,1])
               
        for i in range(expand):
            ind = np.hstack((np.arange(expand-i,expand), np.arange(expand-i) ))
            tempW[:,i,:,:,...] = tempW[:,i,:,ind,...]
        _filter = tempW.reshape([outNum*tranNum, inNum*self.expand, 1, 1 ])
                
#        sio.savemat('Filter2.mat', {'filter': _filter.cpu().detach().numpy()})  
        bias = self.c.repeat([1,1,tranNum,1]).reshape([1,outNum*tranNum,1,1])#.cuda()

        output = F.conv2d(input, _filter,
                        padding=self.padding,
                        dilation=1,
                        groups=1)
        return output+bias  
    
              
class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, tranNum=8, inP = None, 
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1,  Smooth = True):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(kernel_size, n_feats, n_feats, tranNum=tranNum, inP = inP, padding=(kernel_size-1)//2,  bias=bias, Smooth = Smooth))
            if bn:
                m.append(F_BN(n_feats, tranNum))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res
    

def Getini_reg(nNum, inNum, outNum,expand, weight = 1): 
    A = (np.random.rand(outNum,inNum,expand,nNum)-0.5)*2*2.4495/np.sqrt((inNum)*nNum)*np.expand_dims(np.expand_dims(np.expand_dims(weight, axis = 0),axis = 0),axis = 0)/10
    return torch.FloatTensor(A)

def GetBasis_PDOe(sizeP,tranNum=8,inP = None, Smooth = 1):
    p = tranNum
    partial_dict = [[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,0,0,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,0,1/2,0,0],[0,0,0,0,0],[0,0,-1/2,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,-1/4,0,1/4,0],[0,0,0,0,0],[0,1/4,0,-1/4,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,0,1,0,0],[0,0,-2,0,0],[0,0,1,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,0,0,0,0],[-1/2,1,0,-1,1/2],[0,0,0,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,1/2,-1,1/2,0],[0,0,0,0,0],[0,-1/2,1,-1/2,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,1,0,-1,0],[0,-1/2,0,1/2,0],[0,0,0,0,0]],
					[[0,0,1/2,0,0],[0,0,-1,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,-1/2,0,0]],
					[[0,0,0,0,0],[0,0,0,0,0],[1,-4,6,-4,1],[0,0,0,0,0],[0,0,0,0,0]],
					[[0,0,0,0,0],[-1/4,1/2,0,-1/2,1/4],[0,0,0,0,0],[1/4,-1/2,0,1/2,-1/4],[0,0,0,0,0]],
					[[0,0,0,0,0],[0,1,-2,1,0],[0,-2,4,-2,0],[0,1,-2,1,0],[0,0,0,0,0]],
					[[0,-1/4,0,1/4,0],[0,1/2,0,-1/2,0],[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,1/4,0,-1/4,0]],
					[[0,0,1,0,0],[0,0,-4,0,0],[0,0,6,0,0],[0,0,-4,0,0],[0,0,1,0,0]]]


    group_angle = [2*k*pi/p+pi/8 for k in range(p)]
    tran_to_partial_coef = [np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
									 [0,cos(x),sin(x),0,0,0,0,0,0,0,0,0,0,0,0],
									 [0,-sin(x),cos(x),0,0,0,0,0,0,0,0,0,0,0,0],
									 [0,0,0,pow(cos(x),2),2*cos(x)*sin(x),pow(sin(x),2),0,0,0,0,0,0,0,0,0],
									 [0,0,0,-cos(x)*sin(x),pow(cos(x),2)-pow(sin(x),2),sin(x)*cos(x),0,0,0,0,0,0,0,0,0],
									 [0,0,0,pow(sin(x),2),-2*cos(x)*sin(x),pow(cos(x),2),0,0,0,0,0,0,0,0,0],
									 [0,0,0,0,0,0,-pow(cos(x),2)*sin(x),pow(cos(x),3)-2*cos(x)*pow(sin(x),2),-pow(sin(x),3)+2*pow(cos(x),2)*sin(x), pow(sin(x),2)*cos(x),0,0,0,0,0],
									 [0,0,0,0,0,0,cos(x)*pow(sin(x),2),-2*pow(cos(x),2)*sin(x)+pow(sin(x),3),pow(cos(x),3)-2*cos(x)*pow(sin(x),2),sin(x)*pow(cos(x),2),0,0,0,0,0],
									 [0,0,0,0,0,0,0,0,0,0,pow(sin(x),2)*pow(cos(x),2),-2*pow(cos(x),3)*sin(x)+2*cos(x)*pow(sin(x),3),pow(cos(x),4)-4*pow(cos(x),2)*pow(sin(x),2)+pow(sin(x),4),-2*cos(x)*pow(sin(x),3)+2*pow(cos(x),3)*sin(x),pow(sin(x),2)*pow(cos(x),2)]]) for x in group_angle]

    partial_dict = torch.FloatTensor(np.array(partial_dict)) # (15,5,5)
    tran_to_partial_coef = torch.FloatTensor(np.array(tran_to_partial_coef)) #(8,9,15)    
    BasisR =  torch.einsum('ihw,tci->hwtc', partial_dict, tran_to_partial_coef)
    return BasisR/10   

def MaskC(SizeP):
        p = (SizeP-1)/2
        x = np.arange(-p,p+1)/p
        X,Y  = np.meshgrid(x,x)
        C    =X**2+Y**2
        
        Mask = np.ones([SizeP,SizeP])
#        Mask[C>(1+1/(4*p))**2]=0
        Mask = np.exp(-np.maximum(C-1,0)/0.2)
        
        return X, Y, Mask
    
    
class PointwiseAvgPoolAntialiased(nn.Module):
    
    def __init__(self, sizeF, stride, padding=None ):
        super(PointwiseAvgPoolAntialiased, self).__init__()
        sigma = (sizeF-1)/2/3
        self.kernel_size = (sizeF, sizeF)
        if isinstance(stride, int):
            self.stride = (stride, stride)
        elif stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride
        
        if padding is None:
            padding = int((sizeF-1)//2)
            
        if isinstance(padding, int):
            self.padding = (padding, padding)
        else:
            self.padding = padding

        # Build the Gaussian smoothing filter
        grid_x = torch.arange(sizeF).repeat(sizeF).view(sizeF, sizeF)
        grid_y = grid_x.t()
        grid = torch.stack([grid_x, grid_y], dim=-1)
        mean = (sizeF - 1) / 2.
        variance = sigma ** 2.
        r = -torch.sum((grid - mean) ** 2., dim=-1, dtype=torch.get_default_dtype())
        _filter = torch.exp(r / (2 * variance))
        _filter /= torch.sum(_filter)
        _filter = _filter.view(1, 1, sizeF, sizeF)
        self.filter = nn.Parameter(_filter, requires_grad=False)
        #self.register_buffer("filter", _filter)
    
    def forward(self, input):
        _filter = self.filter.repeat((input.shape[1], 1, 1, 1))
        output = F.conv2d(input, _filter, stride=self.stride, padding=self.padding, groups=input.shape[1])        
        return output
        
    
    
    
class F_BN(nn.Module):
    def __init__(self,channels, tranNum=8):
        super(F_BN, self).__init__()
        self.BN = nn.BatchNorm2d(channels)
        self.tranNum = tranNum
    def forward(self, X):
        X = self.BN(X.reshape([X.size(0), int(X.size(1)/self.tranNum), self.tranNum*X.size(2), X.size(3)]))
        return X.reshape([X.size(0), self.tranNum*X.size(1),int(X.size(2)/self.tranNum), X.size(3)])



class F_Dropout(nn.Module):
    def __init__(self,zero_prob = 0.5,  tranNum=8):
        # nn.Dropout2d
        self.tranNum = tranNum
        super(F_Dropout, self).__init__()
        self.Dropout = nn.Dropout2d(zero_prob)
    def forward(self, X):
        X = self.Dropout(X.reshape([X.size(0), int(X.size(1)/self.tranNum), self.tranNum*X.size(2), X.size(3)]))
        return X.reshape([X.size(0), self.tranNum*X.size(1),int(X.size(2)/self.tranNum), X.size(3)])


def build_mask(s, margin=2, dtype=torch.float32):
    mask = torch.zeros(1, 1, s, s, dtype=dtype)
    c = (s-1) / 2
    t = (c - margin/100.*c)**2
    sig = 2.
    for x in range(s):
        for y in range(s):
            r = (x - c) ** 2 + (y - c) ** 2
            if r > t:
                mask[..., x, y] = math.exp((t - r)/sig**2)
            else:
                mask[..., x, y] = 1.
    return mask


class MaskModule(nn.Module):

    def __init__(self, S: int, margin: float = 0.):

        super(MaskModule, self).__init__()

        self.margin = margin
        self.mask = torch.nn.Parameter(build_mask(S, margin=margin), requires_grad=False)


    def forward(self, input):

        assert input.shape[2:] == self.mask.shape[2:]

        out = input * self.mask
        return out

        


