import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as d_sets
from torch.utils.data import DataLoader as d_loader
from torch.utils.checkpoint import checkpoint
import matplotlib.pyplot as plt
from PIL import Image

# device_0 = torch.device("cuda:0")
# device_1 = torch.device("cuda:1")

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN,self).__init__()
        self.conv0 = nn.Conv2d(3,64,kernel_size=5,padding=5//2,padding_mode='replicate')

        self.relu0 = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(3,64,kernel_size=13,padding=13//2,padding_mode='replicate')
        self.relu1 = nn.ReLU(inplace=True)

        self.pooling1 = nn.AvgPool2d(2,2) # AvgPool2d

        self.conv2 = nn.Conv2d(64,32,kernel_size=1,padding=1//2,padding_mode='replicate')
        self.relu2 = nn.ReLU(inplace=True)

        self.pooling2 = nn.AvgPool2d(2,2) # AvgPool2d

        self.conv3 = nn.Conv2d(32,3,kernel_size=1,padding=1//2,padding_mode='replicate')

    def run_first_half(self, *args):
        x = args[0]

        x = self.relu0(self.conv0(x))
        x = self.relu1(self.conv1(x))
        x = self.pooling1(x)

        return x
    
    def run_second_half(self, *args):
        x = args[0]

        x = self.relu2(self.conv2(x))
        x = self.pooling2(x)
        
        return x



    def forward(self,x):
        
        # x = self.relu0(self.conv0(x))

        x = self.relu1(self.conv1(x))

        x = self.pooling1(x)

        x = self.relu2(self.conv2(x))

        x = self.pooling2(x)

        x = self.conv3(x)

        return x


        # x = checkpoint(self.run_first_half, x)
        # x = checkpoint(self.run_second_half, x)
        # x = self.conv3(x)
        # # x.sum.backward()
        # return x



    