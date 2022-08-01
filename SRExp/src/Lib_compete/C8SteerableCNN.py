import torch
import FCNN as fn
from e2cnn import gspaces
from e2cnn import nn
import MyLibForSteerCNN as ML
import numpy as np


class MinstSteerableCNN_simple(torch.nn.Module):
    
    def __init__(self, n_classes=10, N = 8):
        
        super(MinstSteerableCNN_simple, self).__init__()
        
        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = gspaces.Rot2dOnR2(N)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # convolution 1
        out_type = nn.FieldType(self.r2_act, 7*[self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
#            nn.MaskModule(in_type, 29, margin=1),
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 10*[self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
#        self.pool1 = nn.SequentialModule(
#            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
#        )
        
        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 10*[self.r2_act.regular_repr])
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 10*[self.r2_act.regular_repr])
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
#        self.pool2 = nn.SequentialModule(
#            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
#        )
        
        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 10*[self.r2_act.regular_repr])
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        
        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields of C8
#        out_type = nn.FieldType(self.r2_act, 10*[self.r2_act.regular_repr])
#        self.block6 = nn.SequentialModule(
#            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
#            nn.InnerBatchNorm(out_type),
#            nn.ReLU(out_type, inplace=True)
#        )
#        
#        # convolution 7
#        # the old output type is the input type to the next layer
#        in_type = self.block6.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 10*[self.r2_act.regular_repr])
        self.block7 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
                
        self.pool1 = torch.nn.MaxPool2d(2,2,1)
        self.pool2 = torch.nn.MaxPool2d(2,2,1)  
        self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)
        self.gpool = nn.GroupPooling(out_type) 

        
        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(10, 10),
            torch.nn.BatchNorm1d(10),
            torch.nn.ELU(inplace=True),torch.nn.Dropout(0.2),
            torch.nn.Linear(10, n_classes),
        )
    
    def forward(self, input: torch.Tensor, ifshow=0):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, self.input_type)
        
        x = self.block1(x)
        x1 = self.block2(x)
        x = self.pool1(x1.tensor)
        x = nn.GeometricTensor(x, self.block2.out_type)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x.tensor)
        x = nn.GeometricTensor(x, self.block4.out_type)
        x = self.block5(x)
#        x = self.block6(x)
        x = self.block7(x)
        x = self.pool3(x)
        x = self.gpool(x)
        x = x.tensor

        
        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))
        
        if ifshow:
            
            I = x1.tensor[0,:,:,:].permute(1,2,0)
#
#            I1 = I.contiguous().view(25,25, 48, 8)[:,:,0:3,:].permute(0,1,3,2).contiguous().view(25, 25*8,3)
            print('上行：沿角度方向； 下行：沿通道方向')
            #所以在 384 个通道里，角度方向是连续放在一起的
            I1 = I.reshape(25,25, 48, 8)[:,:,0:3,:].permute(0,3,1,2).reshape(25, 25*8,3)
            I2 = I.reshape(25,25, 8, 48)[:,:,:,0:3].permute(0,2,1,3).reshape(25, 25*8,3)
            #可以看出
#            I1 = I.reshape(25,25, 48, 8)[:,:,0:3,0]
#            I2 = I.reshape(25,25, 8, 48)[:,:,0,0:3]

            ML.imshow(np.vstack((I1.detach().cpu().numpy(),I2.detach().cpu().numpy())))
            
            
            pass
        return x


class MinstSteerableCNN_2(torch.nn.Module):
    
    def __init__(self, n_classes=10, N = 8):
        
        super(MinstSteerableCNN_2, self).__init__()
        
        zero_prob = 0.25
        
        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = gspaces.Rot2dOnR2(N)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # convolution 1
        out_type = nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
#            nn.MaskModule(in_type, 29, margin=1),
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 16*[self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True),nn.FieldDropout(out_type,zero_prob)
        )
#        self.pool1 = nn.SequentialModule(
#            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
#        )
        
        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True),nn.FieldDropout(out_type,zero_prob)
        )
        
        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True),nn.FieldDropout(out_type,zero_prob)
        )
#        self.pool2 = nn.SequentialModule(
#            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
#        )
        
        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True),nn.FieldDropout(out_type,zero_prob)
        )

        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True),nn.FieldDropout(out_type,zero_prob)
        )
        
        # convolution 7
        # the old output type is the input type to the next layer
        in_type = self.block6.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block7 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True),nn.FieldDropout(out_type,zero_prob)
        )
                
        self.pool1 = torch.nn.MaxPool2d(2,2,1)
        self.pool2 = torch.nn.MaxPool2d(2,2,1)  
        self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)
        self.gpool = nn.GroupPooling(out_type) 

        
        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(96, 96),
            torch.nn.BatchNorm1d(96),
            torch.nn.ELU(inplace=True),torch.nn.Dropout(0.7),
            torch.nn.Linear(96, n_classes),
        )
    
    def forward(self, input: torch.Tensor, ifshow=0):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, self.input_type)
        
        x = self.block1(x)
        x1 = self.block2(x)
        x = self.pool1(x1.tensor)
        x = nn.GeometricTensor(x, self.block2.out_type)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x.tensor)
        x = nn.GeometricTensor(x, self.block4.out_type)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.pool3(x)
        x = self.gpool(x)
        x = x.tensor

        
        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))
        
        if ifshow:
            
            I = x1.tensor[0,:,:,:].permute(1,2,0)
#
#            I1 = I.contiguous().view(25,25, 48, 8)[:,:,0:3,:].permute(0,1,3,2).contiguous().view(25, 25*8,3)
            print('上行：沿角度方向； 下行：沿通道方向')
            #所以在 384 个通道里，角度方向是连续放在一起的
            I1 = I.reshape(25,25, 48, 8)[:,:,0:3,:].permute(0,3,1,2).reshape(25, 25*8,3)
            I2 = I.reshape(25,25, 8, 48)[:,:,:,0:3].permute(0,2,1,3).reshape(25, 25*8,3)
            #可以看出
#            I1 = I.reshape(25,25, 48, 8)[:,:,0:3,0]
#            I2 = I.reshape(25,25, 8, 48)[:,:,0,0:3]

            ML.imshow(np.vstack((I1.detach().cpu().numpy(),I2.detach().cpu().numpy())))
            
            
            pass
        return x



class C8SteerableCNN(torch.nn.Module):
    
    def __init__(self, n_classes=10, N = 8):
        
        super(C8SteerableCNN, self).__init__()
        
        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = gspaces.Rot2dOnR2(N)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # convolution 1
        out_type = nn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.MaskModule(in_type, 29, margin=1),
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)
        
        self.gpool = nn.GroupPooling(out_type)
        
        # number of output channels
        c = self.gpool.out_type.size
        
        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(c, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64, n_classes),
        )
    
    def forward(self, input: torch.Tensor, ifshow=0):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, self.input_type)
        
        # apply each equivariant block
        
        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
#        print(x.shape)
        x = self.block1(x)
        x1 = self.block2(x)
        x = self.pool1(x1)
#        print(x.shape)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
#        print(x.shape)
        x = self.block5(x)
        x = self.block6(x)
        
        # pool over the spatial dimensions
        x = self.pool3(x)
#        print(x.shape)
        # pool over the group
        x = self.gpool(x)
#        print(x.shape)
        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor

        
        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))
        
        if ifshow:
            
            I = x1.tensor[0,:,:,:].permute(1,2,0)
#
#            I1 = I.contiguous().view(25,25, 48, 8)[:,:,0:3,:].permute(0,1,3,2).contiguous().view(25, 25*8,3)
            print('上行：沿角度方向； 下行：沿通道方向')
            #所以在 384 个通道里，角度方向是连续放在一起的
            I1 = I.reshape(25,25, 48, 8)[:,:,0:3,:].permute(0,3,1,2).reshape(25, 25*8,3)
            I2 = I.reshape(25,25, 8, 48)[:,:,:,0:3].permute(0,2,1,3).reshape(25, 25*8,3)
            #可以看出
#            I1 = I.reshape(25,25, 48, 8)[:,:,0:3,0]
#            I2 = I.reshape(25,25, 8, 48)[:,:,0,0:3]

            ML.imshow(np.vstack((I1.detach().cpu().numpy(),I2.detach().cpu().numpy())))
            
            
            pass
        return x



class C8SteerableCNN1(torch.nn.Module):
    
    def __init__(self, n_classes=10):
        
        super(C8SteerableCNN1, self).__init__()
        
        # the model is equivariant under rotations by 45 degrees, modelled by C8
        self.r2_act = gspaces.Rot2dOnR2(N=8)
        
        # the input image is a scalar field, corresponding to the trivial representation
        in_type = nn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        
        # we store the input type for wrapping the images into a geometric tensor during the forward pass
        self.input_type = in_type
        
        # convolution 1
        # first specify the output type of the convolutional layer
        # we choose 24 feature fields, each transforming under the regular representation of C8
        out_type = nn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        self.block1 = nn.SequentialModule(
            nn.MaskModule(in_type, 29, margin=1),
            nn.R2Conv(in_type, out_type, kernel_size=7, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 2
        # the old output type is the input type to the next layer
        in_type = self.block1.out_type
        # the output type of the second convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block2 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool1 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        # convolution 3
        # the old output type is the input type to the next layer
        in_type = self.block2.out_type
        # the output type of the third convolution layer are 48 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 48*[self.r2_act.regular_repr])
        self.block3 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 4
        # the old output type is the input type to the next layer
        in_type = self.block3.out_type
        # the output type of the fourth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block4 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool2 = nn.SequentialModule(
            nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        # convolution 5
        # the old output type is the input type to the next layer
        in_type = self.block4.out_type
        # the output type of the fifth convolution layer are 96 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block5 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=2, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        
        # convolution 6
        # the old output type is the input type to the next layer
        in_type = self.block5.out_type
        # the output type of the sixth convolution layer are 64 regular feature fields of C8
        out_type = nn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block6 = nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size=5, padding=1, bias=False),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace=True)
        )
        self.pool3 = nn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=1, padding=0)
        
        self.gpool = nn.GroupPooling(out_type)
        
        # number of output channels
        c = self.gpool.out_type.size
        
        # Fully Connected
        self.fully_net = torch.nn.Sequential(
            torch.nn.Linear(c, 64),
            torch.nn.BatchNorm1d(64),
            torch.nn.ELU(inplace=True),
            torch.nn.Linear(64, n_classes),
        )
    
    def forward(self, input: torch.Tensor, ifshow=0):
        # wrap the input tensor in a GeometricTensor
        # (associate it with the input type)
        x = nn.GeometricTensor(input, self.input_type)
        
        # apply each equivariant block
        
        # Each layer has an input and an output type
        # A layer takes a GeometricTensor in input.
        # This tensor needs to be associated with the same representation of the layer's input type
        #
        # The Layer outputs a new GeometricTensor, associated with the layer's output type.
        # As a result, consecutive layers need to have matching input/output types
#        print(x.shape)
        x = self.block1(x)
        x1 = self.block2(x)
        x = self.pool1(x1)
#        print(x.shape)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool2(x)
#        print(x.shape)
        x = self.block5(x)
        x = self.block6(x)
        
        # pool over the spatial dimensions
        x = self.pool3(x)
#        print(x.shape)
        # pool over the group
        x = self.gpool(x)
#        print(x.shape)
        # unwrap the output GeometricTensor
        # (take the Pytorch tensor and discard the associated representation)
        x = x.tensor

        
        # classify with the final fully connected layers)
        x = self.fully_net(x.reshape(x.shape[0], -1))
        
        if ifshow:
            
            I = x1.tensor[0,:,:,:].permute(1,2,0)
#
#            I1 = I.contiguous().view(25,25, 48, 8)[:,:,0:3,:].permute(0,1,3,2).contiguous().view(25, 25*8,3)
            print('上行：沿角度方向； 下行：沿通道方向')
            #所以在 384 个通道里，角度方向是连续放在一起的
            I1 = I.reshape(25,25, 48, 8)[:,:,0:3,:].permute(0,3,1,2).reshape(25, 25*8,3)
            I2 = I.reshape(25,25, 8, 48)[:,:,:,0:3].permute(0,2,1,3).reshape(25, 25*8,3)
            #可以看出
#            I1 = I.reshape(25,25, 48, 8)[:,:,0:3,0]
#            I2 = I.reshape(25,25, 8, 48)[:,:,0,0:3]

            ML.imshow(np.vstack((I1.detach().cpu().numpy(),I2.detach().cpu().numpy())))
            
            
            pass
        return x