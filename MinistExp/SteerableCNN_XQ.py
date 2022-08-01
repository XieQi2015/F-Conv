# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 17:33:06 2021

@author: XieQi
"""


import torch
import F_Conv as fn
import torch.nn as nn
import MyLib as ML
import numpy as np
         
class MinstSteerableCNN(torch.nn.Module):
    
    def __init__(self, n_classes=10, tranNum = 8):
        
        super(MinstSteerableCNN, self).__init__()

        self.tranNum = tranNum
        inP = 4
        zero_prob = 0.25
        
        self.block1 = nn.Sequential(
            fn.Fconv_PCA(7,1,16,tranNum,inP=5,padding=1, ifIni=1, bias=False),
            fn.F_BN(16,tranNum),
            nn.ReLU(inplace=True)
        )
        
        self.block2 = nn.Sequential(
            fn.Fconv_PCA(5,16,16,tranNum,inP,2,bias=False),
            fn.F_BN(16,tranNum),
            nn.ReLU(inplace=True),fn.F_Dropout(zero_prob,tranNum)
        )

        self.block3 = nn.Sequential(
            fn.Fconv_PCA(5,16,32,tranNum,inP,2,bias=False),
            fn.F_BN(32,tranNum),
            nn.ReLU(inplace=True),fn.F_Dropout(zero_prob,tranNum)
        )
        
        self.block4 = nn.Sequential(
            fn.Fconv_PCA(5,32,32,tranNum,inP,2,bias=False),
            fn.F_BN(32,tranNum),
            nn.ReLU(inplace=True),fn.F_Dropout(zero_prob,tranNum)
        )

        self.block5 = nn.Sequential(
            fn.Fconv_PCA(5,32,32,tranNum,inP,2,bias=False),
            fn.F_BN(32,tranNum),
            nn.ReLU(inplace=True),fn.F_Dropout(zero_prob,tranNum)
        ) 
        
        self.block6 = nn.Sequential(
            fn.Fconv_PCA(5,32,64,tranNum,inP,2,bias=False),
            fn.F_BN(64,tranNum),
            nn.ReLU(inplace=True),fn.F_Dropout(zero_prob,tranNum)
        )
        
        self.block7 = nn.Sequential(
            fn.Fconv_PCA(5,64,96,tranNum,inP,1,bias=False),
            fn.F_BN(96,tranNum),
            nn.ReLU(inplace=True),fn.F_Dropout(zero_prob,tranNum)
        )
        
        self.pool1 = nn.MaxPool2d(2,2,1)
        self.pool2 = nn.MaxPool2d(2,2,1)        
        self.pool3 = fn.PointwiseAvgPoolAntialiased(5, 1, padding=0)
        self.gpool = fn.GroupPooling(tranNum)
        
        self.fully_net = nn.Sequential(
            nn.Linear(96, 96),
            nn.BatchNorm1d(96),
            nn.ELU(inplace=True),nn.Dropout(0.7),
            nn.Linear(96, n_classes),
        )
    
    def forward(self, input: torch.Tensor, ifshow=0):
        # wrap the input tensor in a GeometricTensor

        
        x = self.block1(input)
        x1 = self.block2(x)
        x = self.pool1(x1)
#        print(x.shape)
        x = self.block3(x)
        x3 = self.block4(x)
        x = self.pool2(x3)
#        print(x.shape)
        x = self.block5(x)
        x2 = self.block6(x)
        x = self.block7(x2)
        
        x = self.pool3(x)
        x = self.gpool(x)
#        print(x.shape)


        
        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))
        
        if ifshow:
            
            I = x1[0,:,:,:].permute(1,2,0)
            sizeX = I.size(0)
            I1 = I.reshape(sizeX,sizeX, -1, self.tranNum)[:,:,0:3,:].permute(0,3,1,2).reshape(sizeX, sizeX*self.tranNum,3)
            I2 = I.reshape(sizeX,sizeX, self.tranNum, -1)[:,:,:,0:3].permute(0,2,1,3).reshape(sizeX, sizeX*self.tranNum,3)
            ML.imshow(np.vstack((I1.detach().cpu().numpy(),I2.detach().cpu().numpy())))
            
            
            I = x3[0,:,:,:].permute(1,2,0)
            sizeX = I.size(0)
            I1 = I.reshape(sizeX,sizeX, -1, self.tranNum)[:,:,0:3,:].permute(0,3,1,2).reshape(sizeX, sizeX*self.tranNum,3)
            I2 = I.reshape(sizeX,sizeX, self.tranNum, -1)[:,:,:,0:3].permute(0,2,1,3).reshape(sizeX, sizeX*self.tranNum,3)
            ML.imshow(np.vstack((I1.detach().cpu().numpy(),I2.detach().cpu().numpy())))
            
            
            I = x2[0,:,:,:].permute(1,2,0)
            sizeX = I.size(0)
            I1 = I.reshape(sizeX,sizeX, -1, self.tranNum)[:,:,0:3,:].permute(0,3,1,2).reshape(sizeX, sizeX*self.tranNum,3)
            I2 = I.reshape(sizeX,sizeX, self.tranNum, -1)[:,:,:,0:3].permute(0,2,1,3).reshape(sizeX, sizeX*self.tranNum,3)
            ML.imshow(np.vstack((I1.detach().cpu().numpy(),I2.detach().cpu().numpy())))
            
            
            pass
        return x
    
#    def load_state_dict(self, state_dict, strict=False):
#        own_state = self.state_dict()
#        for name, param in state_dict.items():
#            if name in own_state:
#                if isinstance(param, nn.Parameter):
#                    param = param.data
#                try:
#                    own_state[name].copy_(param)
#                except Exception:
#                    raise RuntimeError('While copying the parameter named {}, '
#                                           'whose dimensions in the model are {} and '
#                                           'whose dimensions in the checkpoint are {}.'
#                                           .format(name, own_state[name].size(), param.size()))
#            elif strict:
#                if name.find('tail') == -1:
#                    raise KeyError('unexpected key "{}" in state_dict'
#                                   .format(name))
#
#        if strict:
#            missing = set(own_state.keys()) - set(state_dict.keys())
#            if len(missing) > 0:
#                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
    
    
    
class MinstSteerableCNN_simple(torch.nn.Module):
    
    def __init__(self, n_classes=10, tranNum = 8):
        
        super(MinstSteerableCNN_simple, self).__init__()

        self.tranNum = tranNum
        inP = 4
        
        self.block1 = nn.Sequential(
            fn.Fconv_PCA(7,1,10,tranNum,inP=5,padding=1, ifIni=1, bias=False),
            fn.F_BN(10,tranNum),
            nn.ReLU(inplace=True)
        )
        
        self.block2 = nn.Sequential(
            fn.Fconv_PCA(5,10,10,tranNum,inP,2,bias=False),
            fn.F_BN(10,tranNum),
            nn.ReLU(inplace=True),#fn.F_Dropout(zero_prob,tranNum)
        )

        self.block3 = nn.Sequential(
            fn.Fconv_PCA(5,10,10,tranNum,inP,2,bias=False),
            fn.F_BN(10,tranNum),
            nn.ReLU(inplace=True),#fn.F_Dropout(zero_prob,tranNum)
        )
        
        self.block4 = nn.Sequential(
            fn.Fconv_PCA(5,10,10,tranNum,inP,2,bias=False),
            fn.F_BN(10,tranNum),
            nn.ReLU(inplace=True),#fn.F_Dropout(zero_prob,tranNum)
        )

        self.block5 = nn.Sequential(
            fn.Fconv_PCA(5,10,10,tranNum,inP,2,bias=False),
            fn.F_BN(10,tranNum),
            nn.ReLU(inplace=True),#fn.F_Dropout(zero_prob,tranNum)
        ) 
        
        
        self.block6 = nn.Sequential(
            fn.Fconv_PCA(5,10,10,tranNum,inP,1,bias=False),
            fn.F_BN(10,tranNum),
            nn.ReLU(inplace=True),#fn.F_Dropout(zero_prob,tranNum)
        )
        
        self.pool1 = nn.MaxPool2d(2,2,1)
        self.pool2 = nn.MaxPool2d(2,2,1)        
        self.pool3 = fn.PointwiseAvgPoolAntialiased(5, 1, padding=0)
        self.gpool = fn.GroupPooling(tranNum)
        
        self.fully_net = nn.Sequential(
            nn.Linear(10, 10),
            nn.BatchNorm1d(10),
            nn.ELU(inplace=True),nn.Dropout(0.2),
            nn.Linear(10, n_classes),
        )
    
    def forward(self, input: torch.Tensor, ifshow=0):
        # wrap the input tensor in a GeometricTensor

    
        x = self.block1(input)
        x1 = self.block2(x)
        x = self.pool1(x1)
        x = self.block3(x)
        x3 = self.block4(x)
        x = self.pool2(x3)
        x = self.block5(x)
        x2 = self.block6(x)      
        x = self.pool3(x2)
        x = self.gpool(x)

        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))
        
        if ifshow:
            
            I = x1[0,:,:,:].permute(1,2,0)
            sizeX = I.size(0)
            I1 = I.reshape(sizeX,sizeX, -1, self.tranNum)[:,:,0:3,:].permute(0,3,1,2).reshape(sizeX, sizeX*self.tranNum,3)
            I2 = I.reshape(sizeX,sizeX, self.tranNum, -1)[:,:,:,0:3].permute(0,2,1,3).reshape(sizeX, sizeX*self.tranNum,3)
            ML.imshow(np.vstack((I1.detach().cpu().numpy(),I2.detach().cpu().numpy())))
            
            
            I = x3[0,:,:,:].permute(1,2,0)
            sizeX = I.size(0)
            I1 = I.reshape(sizeX,sizeX, -1, self.tranNum)[:,:,0:3,:].permute(0,3,1,2).reshape(sizeX, sizeX*self.tranNum,3)
            I2 = I.reshape(sizeX,sizeX, self.tranNum, -1)[:,:,:,0:3].permute(0,2,1,3).reshape(sizeX, sizeX*self.tranNum,3)
            ML.imshow(np.vstack((I1.detach().cpu().numpy(),I2.detach().cpu().numpy())))
            
            
            I = x2[0,:,:,:].permute(1,2,0)
            sizeX = I.size(0)
            I1 = I.reshape(sizeX,sizeX, -1, self.tranNum)[:,:,0:3,:].permute(0,3,1,2).reshape(sizeX, sizeX*self.tranNum,3)
            I2 = I.reshape(sizeX,sizeX, self.tranNum, -1)[:,:,:,0:3].permute(0,2,1,3).reshape(sizeX, sizeX*self.tranNum,3)
            ML.imshow(np.vstack((I1.detach().cpu().numpy(),I2.detach().cpu().numpy())))

        return x
#    def load_state_dict(self, state_dict, strict=False):
#        own_state = self.state_dict()
#        for name, param in state_dict.items():
#            if name in own_state:
#                if isinstance(param, nn.Parameter):
#                    param = param.data
#                try:
#                    own_state[name].copy_(param)
#                except Exception:
#                    raise RuntimeError('While copying the parameter named {}, '
#                                           'whose dimensions in the model are {} and '
#                                           'whose dimensions in the checkpoint are {}.'
#                                           .format(name, own_state[name].size(), param.size()))
#        if strict:
#            missing = set(own_state.keys()) - set(state_dict.keys())
#            if len(missing) > 0:
#                raise KeyError('missing keys in state_dict: "{}"'.format(missing))