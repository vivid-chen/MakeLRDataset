import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as d_sets
from torch.utils.data import DataLoader as d_loader
import matplotlib.pyplot as plt
from PIL import Image

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.blocks = nn.Identity()
        self.activate = activation_func(activation)
        self.shortcut = nn.Identity()   
    
    def forward(self, x):
        residual = x
        if self.should_apply_shortcut: residual = self.shortcut(x)
        x = self.blocks(x)
        x += residual
        x = self.activate(x)
        return x
    
    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN,self).__init__()
        self.conv0 = nn.Conv2d(3,64,kernel_size=15,padding=15//2,padding_mode='replicate')
        self.conv0 = nn.DataParallel(self.conv0) # , device_ids=[0,1]

        self.relu0 = nn.ReLU()
        self.relu0 = nn.DataParallel(self.relu0) # , device_ids=[0,1]



        self.conv1 = nn.Conv2d(3,32,kernel_size=15,padding=15//2,padding_mode='replicate')
        self.conv1 = nn.DataParallel(self.conv1) # , device_ids=[0,1]

        # self.BN1 = nn.BatchNorm2d(64)
        # self.BN1 = nn.DataParallel(self.BN1)
        self.relu1 = nn.ReLU()
        self.relu1 = nn.DataParallel(self.relu1) # , device_ids=[0,1]

        self.pooling1 = nn.AvgPool2d(2,2) # AvgPool2d
        self.pooling1 = nn.DataParallel(self.pooling1) # , device_ids=[0,1]

        self.conv2 = nn.Conv2d(32,16,kernel_size=1,padding=1//2,padding_mode='replicate')
        self.conv2 = nn.DataParallel(self.conv2) # , device_ids=[0,1]

        # self.BN2 = nn.BatchNorm2d(32)
        # self.BN2 = nn.DataParallel(self.BN2)
        self.relu2 = nn.ReLU()
        self.relu2 = nn.DataParallel(self.relu2) # , device_ids=[0,1]

        self.pooling2 = nn.AvgPool2d(2,2) # AvgPool2d
        self.pooling2 = nn.DataParallel(self.pooling2) # , device_ids=[0,1]

        self.conv3 = nn.Conv2d(16,3,kernel_size=3,padding=3//2,padding_mode='replicate')
        self.conv3 = nn.DataParallel(self.conv3) # , device_ids=[0,1]
        
    def forward(self,x):
        # out = self.conv0(x)
        # out = self.relu0(out)

        out = self.conv1(x)
        out = self.relu1(out)

        out = self.pooling1(out)

        out = self.conv2(out)
        out = self.relu2(out)

        out = self.pooling2(out)

        out = self.conv3(out)

        return out

    @property
    def should_apply_shortcut(self):
        return self.in_channels != self.out_channels

    