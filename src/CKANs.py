from .kan_convolutional.KANConv import KAN_Convolutional_Layer  # use kan replace KANconv if you want to symbolic
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from kan import *
from .kan_convolutional.KANLinear import KANLinear


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")





class InceptionCKANs(nn.Module):
    def __init__(self, device):
        super(InceptionCKANs, self).__init__()

        self.conv1 = KAN_Convolutional_Layer(
            n_convs=1,
            kernel_size=(10, 10),
            stride=(1, 1),
            padding=(0, 0),
            device=device
        )

        self.conv3 = nn.Sequential(
            KAN_Convolutional_Layer(
                n_convs=1,
                kernel_size=(8, 8),
                stride=(1, 1),
                padding=(0, 0),
                device=device
            ),
            KAN_Convolutional_Layer(
                n_convs=1,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(0, 0),  
                device=device
            )
        )


    def forward(self, x):
        conv1_out = self.conv1(x)
        conv3_out = self.conv3(x)
        output = torch.cat([conv1_out, conv3_out], dim=1)
        return output



class CKANs(nn.Module):
    def __init__(self, input_channels=1, num_classes=3, device=device):
        super(CKANs, self).__init__()
        self.device = device


        # Rebin layer Rebin to 20x20
        self.Rebinlayers_1 = nn.Sequential(
            KAN_Convolutional_Layer(n_convs=1, kernel_size=(5, 5), stride=(5, 5), device=device),
        )

        self.Rebinlayers_2 = nn.Sequential(
            KAN_Convolutional_Layer(n_convs=1, kernel_size=(5, 5), stride=(5, 5), device=device),
        )
        
        self.Rebinlayers_3 = nn.Sequential(
            KAN_Convolutional_Layer(n_convs=1, kernel_size=(5, 5), stride=(5, 5), device=device),
        )



        self.conv1 = nn.Sequential(
            KAN_Convolutional_Layer(
            n_convs=1,
            kernel_size=(8, 8),
            device=device
        ),      
            InceptionCKANs(device=device)     
        )

        self.conv2 = nn.Sequential(
            KAN_Convolutional_Layer(
            n_convs=1,
            kernel_size=(8, 8),
            device=device
        ),      
            InceptionCKANs(device=device)     
        )

        self.conv3 = nn.Sequential(
            KAN_Convolutional_Layer(
            n_convs=1,
            kernel_size=(8, 8),
            device=device
        ),      

            InceptionCKANs(device=device)     
        )

        self.flat = nn.Flatten()


        # self.kan1 = KAN(width=[32, 3], grid=5, k=3, device=device,seed=0,base_fun=torch.nn.Tanh()).to(self.device)
        # self.kan2 = KAN(width=[32, 3], grid=5, k=3, device=device,seed=0,base_fun=torch.nn.Tanh()).to(self.device)
        # self.kan3 = KAN(width=[32, 3], grid=5, k=3, device=device,seed=0,base_fun=torch.nn.Tanh()).to(self.device)

        self.kan1 = KANLinear(in_features=32,out_features=3,base_activation=torch.nn.Tanh).to(self.device)
        self.kan2 = KANLinear(in_features=32,out_features=3,base_activation=torch.nn.Tanh).to(self.device)
        self.kan3 = KANLinear(in_features=32,out_features=3,base_activation=torch.nn.Tanh).to(self.device)



        self.weight1 = nn.Parameter(torch.tensor(1.0))
        self.weight2 = nn.Parameter(torch.tensor(1.0))
        self.weight3 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        x1 = x[:, 0:1, :, :].to(self.device)
        x2 = x[:, 1:2, :, :].to(self.device)
        x3 = x[:, 2:3, :, :].to(self.device)
        x1 = self.Rebinlayers_1(x1)
        x2 = self.Rebinlayers_2(x2)
        x3 = self.Rebinlayers_3(x3)


        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        
        
    
        x1 = self.kan1(self.flat(x1))
        x2 = self.kan2(self.flat(x2))
        x3 = self.kan3(self.flat(x3))

        x = self.weight1 *  x1 + self.weight2 * x2 + self.weight3 * x3

        return x
    





class InceptionBig(nn.Module):
    def __init__(self, num_classes, device):
        super(InceptionBig, self).__init__()
        self.device = device
        
        # 初始卷积层（单通道输入）
        self.conv = KAN_Convolutional_Layer(
            n_convs=1,
            kernel_size=(12, 12),
            device=device
        )
        
        # Inception 模块
        self.inception = InceptionModule(device=device)
        
        # 全连接层
        self.fc = nn.Linear(128, num_classes)  # 设定Inception模块的输出通道数为32
        
    def forward(self, x):
        x = self.conv(x)
        print(x.size())
        x = self.inception(x)

        x = torch.flatten(x, 1)
        # print(x.size())
        x = self.fc(x)
        return x



class CKANs_InceptionBig(nn.Module):
    def __init__(self, input_channels=1, num_classes=3, device=device):
        super(CKAN_Model, self).__init__()
        self.device = device
        self.Rebinlayers_1 = nn.Sequential(
            KAN_Convolutional_Layer(n_convs=1, kernel_size=(5, 5), stride=(5, 5), device=device),
        )

        self.Rebinlayers_2 = nn.Sequential(
            KAN_Convolutional_Layer(n_convs=1, kernel_size=(5, 5), stride=(5, 5), device=device),
        )
        
        self.Rebinlayers_3 = nn.Sequential(
            KAN_Convolutional_Layer(n_convs=1, kernel_size=(5, 5), stride=(5, 5), device=device),
        )



        self.conv1 = nn.Sequential(
            KAN_Convolutional_Layer(
            n_convs=1,
            kernel_size=(12, 12),
            device=device
        ),      
        # Inception 模块
            InceptionBig(device=device)     
        )


        self.conv2 = nn.Sequential(
            KAN_Convolutional_Layer(
            n_convs=1,
            kernel_size=(12, 12),
            device=device
        ),      
        # Inception 模块
            InceptionBig(device=device)     
        )

        self.conv3 = nn.Sequential(
            KAN_Convolutional_Layer(
            n_convs=1,
            kernel_size=(12, 12),
            device=device
        ),      
        # Inception 模块
            InceptionBig(device=device)     
        )
        self.flat = nn.Flatten()

        # 定义 KAN
        self.kan1 = KAN(width=[128, 3], grid=5, k=3, device=device,seed=0,base_fun=torch.nn.Tanh()).to(self.device)
        self.kan2 = KAN(width=[128, 3], grid=5, k=3, device=device,seed=0,base_fun=torch.nn.Tanh()).to(self.device)
        self.kan3 = KAN(width=[128, 3], grid=5, k=3, device=device,seed=0,base_fun=torch.nn.Tanh()).to(self.device)

        self.weight1 = nn.Parameter(torch.tensor(1.0))
        self.weight2 = nn.Parameter(torch.tensor(1.0))
        self.weight3 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # 将图像的三个通道分别输入同一个 CNN
        x1 = x[:, 0:1, :, :].to(self.device)

        x2 = x[:, 1:2, :, :].to(self.device)
        x3 = x[:, 2:3, :, :].to(self.device)

        x1 = self.Rebinlayers_1(x1)
        x2 = self.Rebinlayers_2(x2)
        x3 = self.Rebinlayers_3(x3)


        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        
        
    
        x1 = self.kan1(self.flat(x1))
        x2 = self.kan2(self.flat(x2))
        x3 = self.kan3(self.flat(x3))

        x = self.weight1 *  x1 + self.weight2 * x2 + self.weight3 * x3

        return x




class CKANs_BigConvs(nn.Module):
    def __init__(self, input_channels=1, num_classes=3, device=device):
        super(CKANs_BigConvs, self).__init__()
        self.device = device
        self.Rebinlayers_1 = nn.Sequential(
            KAN_Convolutional_Layer(n_convs=1, kernel_size=(5, 5), stride=(5, 5), device=device),
        )

        self.Rebinlayers_2 = nn.Sequential(
            KAN_Convolutional_Layer(n_convs=1, kernel_size=(5, 5), stride=(5, 5), device=device),
        )
        
        self.Rebinlayers_3 = nn.Sequential(
            KAN_Convolutional_Layer(n_convs=1, kernel_size=(5, 5), stride=(5, 5), device=device),
        )



        self.conv1 = nn.Sequential(
            KAN_Convolutional_Layer(n_convs=1, kernel_size=(16, 16), device=device)
        )

        self.conv2 = nn.Sequential(
            KAN_Convolutional_Layer(n_convs=1, kernel_size=(16, 16), device=device)

        )

        self.conv3 = nn.Sequential(
            KAN_Convolutional_Layer(n_convs=1, kernel_size=(16, 16), device=device)
        )

        self.flat = nn.Flatten()

        # 定义 KAN
        self.kan1 = KAN(width=[25, 8, 3], grid=5, k=3, device=device,seed=0).to(self.device)
        self.kan2 = KAN(width=[25, 8, 3], grid=5, k=3, device=device,seed=0).to(self.device)
        self.kan3 = KAN(width=[25, 8, 3], grid=5, k=3, device=device,seed=0).to(self.device)

        self.weight1 = nn.Parameter(torch.tensor(1.0))
        self.weight2 = nn.Parameter(torch.tensor(1.0))
        self.weight3 = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # 将图像的三个通道分别输入同一个 CNN
        x1 = x[:, 0:1, :, :].to(self.device)

        x2 = x[:, 1:2, :, :].to(self.device)
        x3 = x[:, 2:3, :, :].to(self.device)

        x1 = self.Rebinlayers_1(x1)
        x2 = self.Rebinlayers_2(x2)
        x3 = self.Rebinlayers_3(x3)


        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        
        
    
        x1 = self.kan1(self.flat(x1))
        x2 = self.kan2(self.flat(x2))
        x3 = self.kan3(self.flat(x3))

        x = self.weight1 * x1 + self.weight2 * x2 + self.weight3 * x3

        return x