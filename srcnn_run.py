from __future__ import print_function
import argparse
import torch
import math
import os
from torch.autograd import Variable
from PIL import Image

from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--input_image', type=str, default = '/media/lab1008/A2FE99FDBDECF1EA/CZY/MakeLRDataset/dataset/RealSR/images/HR/003', help='input image to use')
parser.add_argument('--model', type=str, default = '/media/lab1008/A2FE99FDBDECF1EA/CZY/MakeLRDataset/model/model_epoch_300.pth', help='model file to use')
parser.add_argument('--output_filename', type=str, default = '/media/lab1008/A2FE99FDBDECF1EA/CZY/MakeLRDataset/dataset/RealSR/images/LR/003', help='where to save the output image')
parser.add_argument('--scale_factor', type=float, default = 4, help='factor by which super resolution needed')

parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()

print(opt)
model = torch.load(opt.model)

data_dir = opt.input_image
files = os.listdir(data_dir)
for f in files:
    img = Image.open(data_dir+"/"+f)

    input = Variable(ToTensor()(img)).view(1, -1, img.size[1], img.size[0])
    opt.cuda = True
    if opt.cuda:
        model = model.cuda()
        input = input.cuda()

    out = model(input)
    out = out.cpu()

    print ("type = ",type(out))
    tt = transforms.ToPILImage()

    img_out = tt(out.data[0])

    # img_out = img_out.convert('RGB')

    img_out.save(opt.output_filename+"/"+f)

# exit()

