# -*- coding: utf-8 -*-
"""
Created on Tue May  1 12:50:06 2018

@author: XieQi
"""

import numpy as np
import matplotlib.pyplot as plt
import MyLib as ML
import os
#import torch

def linear_interpolation_2D(input_array, indices, outside_val=0, boundary_correction=True):
    # http://stackoverflow.com/questions/6427276/3d-interpolation-of-numpy-arrays-without-scipy
    output = np.empty(indices[0].shape)
    ind_0 = indices[0,:]
    ind_1 = indices[1,:]

    N0, N1 = input_array.shape

    x0_0 = ind_0.astype(np.integer)
    x1_0 = ind_1.astype(np.integer)
    x0_1 = x0_0 + 1
    x1_1 = x1_0 + 1

    # Check if inds are beyond array boundary:
    if boundary_correction:
        # put all samples outside datacube to 0
        inds_out_of_range = (x0_0 < 0) | (x0_1 < 0) | (x1_0 < 0) | (x1_1 < 0) |  \
                            (x0_0 >= N0) | (x0_1 >= N0) | (x1_0 >= N1) | (x1_1 >= N1)

        x0_0[inds_out_of_range] = 0
        x1_0[inds_out_of_range] = 0
        x0_1[inds_out_of_range] = 0
        x1_1[inds_out_of_range] = 0

    w0 = ind_0 - x0_0
    w1 = ind_1 - x1_0
    # Replace by this...
    # input_array.take(np.array([x0_0, x1_0, x2_0]))
    output = (input_array[x0_0, x1_0] * (1 - w0) * (1 - w1)  +
              input_array[x0_1, x1_0] * w0 * (1 - w1)  +
              input_array[x0_0, x1_1] * (1 - w0) * w1  +
              input_array[x0_1, x1_1] * w0 * w1 )


    if boundary_correction:
        output[inds_out_of_range] = 0

    return output

def rotate_grid_2D(grid, theta):
    """ Rotate grid """
    theta = np.deg2rad(theta)

    x0 = grid[0, :] * np.cos(theta) - grid[1, :] * np.sin(theta)
    x1 = grid[0, :] * np.sin(theta) + grid[1, :] * np.cos(theta)

    grid[0, :] = x0
    grid[1, :] = x1
    return grid

def rotate_im(im, theta):
    grid = getGrid([28, 28])
    grid = rotate_grid_2D(grid, theta)
    grid += 13.5
    data = linear_interpolation_2D(im, grid)
    data = np.reshape(data, [28, 28])
    return data.astype('float32')

def getGrid(siz):
    """ Returns grid with coordinates from -siz[0]/2 : siz[0]/2, -siz[1]/2 : siz[1]/2, ...."""
    space = [np.linspace( -(N/2), (N/2), N ) for N in siz]
    mesh = np.meshgrid( *space, indexing='ij' )
    mesh = [np.expand_dims( ax.ravel(), 0) for ax in mesh]

    return np.concatenate(mesh)


def MaskC(SizeP):
        p = (SizeP-1)/2
        x = np.arange(-p,p+1)/p
        X,Y  = np.meshgrid(x,x)
        C    =X**2+Y**2
        
        Mask = np.ones([SizeP,SizeP])
        Mask[C>(1+1/(4*p))**2]=0
#        for i in range(SizeP):
#            for j in range(SizeP):
#                if C[i,j]>(1+1/(4*p))**2:
#                    Mask[i,j]= 0
        
#        Mask = (C<=(1+1/(4*p))**2)
#        Mask.astype(int)
        X = np.expand_dims(np.expand_dims(X,axis=0),axis=0)
        Y = np.expand_dims(np.expand_dims(Y,axis=0),axis=0)
        Mask = np.expand_dims(np.expand_dims(Mask,axis=0),axis=0)
#        X = np.tile(X, (sizeB,1,1,1))
#        Y = np.tile(Y, (sizeB,1,1,1))
#        Mask = np.tile(Mask, (sizeB,1,1,1))
        
        return X, Y, Mask

def normalized(X):
    maxX = np.max(X)
    minX = np.min(X)
    X = (X-minX)/(maxX - minX)
    return X

def setRange(X, maxX = 1, minX = 0):
    X = (X-minX)/(maxX - minX+0.00001)
    return X


def get3band_of_tensor(outX,nbanch=0,nframe=[0,1,2]):
    outX = np.transpose(outX.cpu().detach().numpy() ,[0,2,3,1])
    X = outX[:,:,:,nframe]
    X = X[nbanch,:,:,:]
    return X

def get3band_of_tensorCPU(outX,nbanch=0,nframe=[0,1,2]):
    outX = np.transpose(outX ,[0,2,3,1])
    X = outX[:,:,:,nframe]
    X = X[nbanch,:,:,:]
    return X


def get3band_of_tensor2(outX,nbanch=0,nframe=[0,1,2]):
    outX = np.transpose(outX.cpu().detach().numpy() ,[0,2,3,1])
    X = outX[:,:,:,nframe]
    X = X[nbanch,:,:,:]
    for i in range(len(nframe)):
        X[:,:,i] = ML.normalized(X[:,:,i])
    return X


#def imshow2(X):
##    X = ML.normalized(X)
#    X = np.maximum(X,0)
#    X = np.minimum(X,1)
#    plt.imshow(X[:,:,::-1]) 
#    plt.axis('off') 
#    plt.show()  
    
def imwrite2(X,saveload='tempIm'):
    plt.imsave(saveload, ML.normalized(X[:,:,::-1]))    
    
def plot(X):
#    X = ML.normalized(X)
    X = np.maximum(X,0)
    X = np.minimum(X,1)
    plt.plot(X) 
#    plt.axis('off') 
    plt.show() 


def imshow(X,ifnormalize = 0):
#    X = ML.normalized(X)
    if ifnormalize!=0:
        X = ML.normalized(X)
    X = np.maximum(X,0)
    X = np.minimum(X,1)
    plt.imshow(X) 
    plt.axis('off') 
    plt.show() 
    
def imshow2(X, ifnormalize = 0):
    X = X.detach().numpy()
    if ifnormalize!=0:
        X = ML.normalized(X)
    X = np.maximum(X,0)
    X = np.minimum(X,1)
    plt.imshow(X) 
    plt.axis('off') 
    plt.show()  
    
def imwrite(X,saveload='tempIm'):
    plt.imsave(saveload, ML.normalized(X))
    


def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   #判断是否存在文件夹如果不存在则创建为文件夹
		os.makedirs(path)            #makedirs 创建文件时如果路径不存在会创建这个路径
		print("---  new folder...  ---")
		print("---  "+path+"  ---")
	else:
		print("---  There is "+ path + " !  ---")
        
        
def mypadding(X, paddnum = 8):
    
    H = X.shape[1]
    W = X.shape[2]
    
    tempX = np.zeros(np.array(X.shape)+[0,paddnum*2,paddnum*2,0])
    
    tempX[:, paddnum:H+paddnum, paddnum:W+paddnum, :] = X
    # 四个角的padding
    temptemp           = X[:,0:paddnum,0:paddnum,:]
    tempX[:,0:paddnum,0:paddnum,:] = temptemp[:,::-1,::-1,:]
    
    temptemp           = X[:,H-paddnum:H,W-paddnum:W,:]
    tempX[:,paddnum+H:H+paddnum*2,paddnum+W:W+paddnum*2,:] = temptemp[:,::-1,::-1,:]
    
    temptemp           = X[:,H-paddnum:H,0:paddnum,:]
    tempX[:,paddnum+H:H+paddnum*2,0:paddnum,:] = temptemp[:,::-1,::-1,:]
    
    temptemp           = X[:,0:paddnum,W-paddnum:W,:]
    tempX[:,0:paddnum,paddnum+W:W+paddnum*2,:] = temptemp[:,::-1,::-1,:]
    
    # 四个边的padding
    temptemp           = X[:,0:paddnum,:,:]
    tempX[:,0:paddnum,paddnum:W+paddnum,:] = temptemp[:,::-1,:,:]
    
    temptemp           = X[:,H-paddnum:H,:,:]
    tempX[:,paddnum+H:H+paddnum*2,paddnum:W+paddnum,:] = temptemp[:,::-1,:,:]
    
    temptemp           = X[:,:,0:paddnum,:]
    tempX[:,paddnum:H+paddnum,0:paddnum,:] = temptemp[:,:,::-1,:]
    
    temptemp           = X[:,:,W-paddnum:W,:]
    tempX[:,paddnum:H+paddnum,paddnum+W:W+paddnum*2,:] = temptemp[:,:,::-1,:]
    
    return tempX

def getC(Y, Z, inC, sizeC=32, ratio= 32):
    #用来估计RC的代码
    inC = np.reshape(inC,  [sizeC*sizeC,1])
    v1  = np.ones([sizeC*sizeC,1])
    h,w,outDim = Z.shape
#    print(outDim)
    Z = np.reshape(Z, [h*w,outDim])
    Ypatch = ML.im2Patch(Y, sizeC, ratio)
    YY = np.linalg.inv(np.matmul(Ypatch.T,Ypatch)+np.eye(sizeC*sizeC)*0.001)
    vYYv = np.matmul(np.matmul(v1.T,YY),v1)
#    inC = np.ones([sizeC,sizeC])/sizeC/sizeC
     
    numMS = Y.shape[2]
    
    for i in range(30):
        CY = np.matmul(Ypatch, inC)
        R  = np.matmul(np.matmul(np.linalg.inv(np.matmul(Z.T,Z)), Z.T),np.reshape(CY,[h*w,numMS]))
        ZR = np.matmul(Z,R)
        uZR = np.reshape(ZR,[1,h*w*numMS])

        inC = np.matmul((np.matmul(uZR,Ypatch)- (np.matmul(np.matmul(np.matmul(uZR,Ypatch),YY),v1) -1 )/vYYv*v1.T),YY)
        inC = inC.T
#    print(np.sum(inC))
    inC = np.reshape(inC,[sizeC,sizeC])
    inC = np.expand_dims(inC[::-1,::-1], axis = 2)
    inC = np.expand_dims(inC, axis = 3)
    return inC

def getC2(Y, Z, allC, sizeC=32, ratio= 32):
    #用来估计RC的代码
    
    numA = allC.shape[2]
    allC = np.reshape(allC, [sizeC*sizeC, numA])
    
    alpha = np.ones([numA,1])
    
#    inC = np.reshape(inC,  [sizeC*sizeC,1])
    v1  = np.ones([numA,1])
    h,w,outDim = Z.shape
#    print(outDim)
    Z = np.reshape(Z, [h*w,outDim])
    Ypatch = ML.im2Patch(Y, sizeC, ratio)
    YCall  = np.matmul(Ypatch, allC)
    
    YY = np.linalg.inv(np.matmul(YCall.T,YCall)+np.eye(numA)*0.0000001)
    vYYv = np.matmul(np.matmul(v1.T,YY),v1)
#    inC = np.ones([sizeC,sizeC])/sizeC/sizeC
     
    for i in range(30):
        CY = np.matmul(YCall, alpha)
        R  = np.matmul(np.matmul(np.linalg.inv(np.matmul(Z.T,Z)), Z.T),np.reshape(CY,[h*w,3]))
        ZR = np.matmul(Z,R)
        uZR = np.reshape(ZR,[1,h*w*3])

        alpha = np.matmul((np.matmul(uZR,YCall)- (np.matmul(np.matmul(np.matmul(uZR,YCall),YY),v1) -1 )/vYYv*v1.T),YY)
        alpha = alpha.T
#    print(np.sum(inC))
    inC = np.matmul(allC, alpha)
    inC = np.reshape(inC,[sizeC,sizeC])
    inC = np.expand_dims(inC[::-1,::-1], axis = 2)
    inC = np.expand_dims(inC, axis = 3)
    return inC, R


def im2Patch(Y, sizeC, ratio):
    k = 0
    H,W,C = Y.shape
    h = int(H/ratio)
    w = int(W/ratio)
    Ypatch = np.zeros([h*w*C, sizeC*sizeC],'f')
#    print(Ypatch.shape)
#    padY = mypadding(Y, (sizeC-ratio)/2)
    padY = np.squeeze(ML.mypadding(np.expand_dims(Y, axis = 0), int((sizeC-ratio)/2)),axis = 0)
    
    for i in range(sizeC):
        for j in range(sizeC):
            temp = padY[i:h*ratio+i:ratio,j:w*ratio+j:ratio,:]
#            print(temp.shape)
            Ypatch[:,k] = np.reshape(temp,[h*w*C])
            k=k+1
    
    return Ypatch

def gauss(kernel_size, sigma):
    
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size//2 - 0.5
    if sigma<=0:
        sigma = ((kernel_size-1)*0.5-1)*0.3+0.8
    
    s = sigma**2
    sum_val =  0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i-center, j-center
            
            kernel[i, j] = np.exp(-(x**2+y**2)/2/s)
            sum_val += kernel[i, j]


    kernel = kernel/sum_val
    return kernel
		
#file = "test/"
#mkdir(file)  