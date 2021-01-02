from __future__ import print_function
import argparse
from math import log10

from tensorboardX import SummaryWriter 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from srcnn_data import get_training_set, get_test_set
from srcnn_model import SRCNN

writer = SummaryWriter()

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batch_size', type=int, default=15, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=10, help='testing batch size')
parser.add_argument('--epochs', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.01')
# parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--cuda', type=bool, default=True, help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=1, help='random seed to use. Default=123')
opt = parser.parse_args()
writer.add_text('batch_size', str(opt.batch_size))
writer.add_text('test_batch_size', str(opt.test_batch_size))
writer.add_text('epochs', str(opt.epochs))
writer.add_text('lr_init', str(opt.lr))

print(opt)

# 是否用GPU
opt.cuda = True # CZY add
use_cuda = opt.cuda
if use_cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

# GPU随机种子
torch.manual_seed(opt.seed)
if use_cuda:
    torch.cuda.manual_seed(opt.seed)


train_set = get_training_set(opt.upscale_factor)
test_set = get_test_set(opt.upscale_factor)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batch_size, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.test_batch_size, shuffle=False)


srcnn = SRCNN()
criterion = nn.MSELoss()


if(use_cuda):
	srcnn.cuda()
	criterion = criterion.cuda()

optimizer = optim.Adam(srcnn.parameters(),lr=opt.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1) # 优化器

def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        optimizer.zero_grad()
        #print ("input shape = " , input.shape)
        #print ("target shape = ", target.shape)
        model_out = srcnn(input)
        #print ("model_out shape =" , model_out.shape)
        loss = criterion(model_out, target)
        # epoch_loss += loss.data[0] # CZY
        epoch_loss += loss.data

        # optimizer.zero_grad() # 反向传播前手动将梯度置零
        loss.backward()
        optimizer.step()

        # print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data[0])) # CZY
        # print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.data))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    writer.add_scalar("epoch_loss", epoch_loss / len(training_data_loader), global_step = epoch) # 记log




def test(epoch):
    avg_psnr = 0
    for batch in testing_data_loader:
        input, target = Variable(batch[0]), Variable(batch[1])
        if use_cuda:
            input = input.cuda()
            target = target.cuda()

        prediction = srcnn(input)
        mse = criterion(prediction, target)
        # psnr = 10 * log10(1 / mse.data[0]) # CZY
        psnr = 10 * log10(1 / mse.data)
        avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    writer.add_scalar("test_avg_PSNR", avg_psnr / len(testing_data_loader), global_step=epoch)


def checkpoint(epoch):
    model_out_path = "./model/model_epoch_{}.pth".format(epoch)
    torch.save(srcnn, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(1, opt.epochs + 1):
    scheduler.step()
    train(epoch)
    test(epoch)
    if(epoch%10==0):
        # checkpoint(epoch) # CZY 暂时不保存模型，看一下loss
    
    # print(optimizer.param_groups[0]['lr'])


writer.close()

