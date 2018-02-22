# Reference
# https://github.com/yunjey/pytorch-tutorial
# http://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math

class LenNet(nn.Module):
    '''Restaurant at the end of the universe.'''
    def __init__(self, weight_init=None):
        super(LenNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # init
        if weight_init=='xavier':
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                    module.weight.data.normal_(0, math.sqrt(2. / n))
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ResBlock(nn.Module):
    '''Block for ResNet.
    
    As many layers for ResNet share the same kernel size and number of channels, we design this module.
    Typically, there are 2 conv layers for each block.
    '''
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual = nn.Sequential()
        if stride!=1 or in_channels!=out_channels:
            self.residual = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(out_channels),
                    )

    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output += self.residual(x)
        output = F.relu(output)
        return output

    """
    def __init__(self, in_channels, out_channels, is_top_block=False):
        '''If it's the first block of layer:
            - we use a stride of 2 for the first conv layer of this block
            - double in_channels
            Otherwise:
            - stride = 1
            - in_channels = out_channels
        '''
        super(ResBlock, self).__init__()
        self.is_top_block = is_top_block

        if is_top_block:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
            # double number of channels
            self.downsample = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, stride=2, padding=0, bias=False),
                        #nn.BatchNorm2d(out_channels),
                    ) 
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.downsample = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.conv2(y)
        #print('Y shape: ')
        #print(y.data.numpy().shape)
        if self.downsample:
            x = self.downsample(x)
            #print('X shape: ')
            #print(x.data.numpy().shape)
        y += x
        y = self.bn2(y)
        return F.relu(y)
    """

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride,] + [1,] * (num_blocks-1)
        res_layer = []
        for stride in strides:
            res_layer.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*res_layer)

    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.layer1(output)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = F.avg_pool2d(output, 4)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        return output

def ResNet18():
    return ResNet(ResBlock, [2,2,2,2])
        

class ResNet_34(nn.Module):
    '''ResNet-34.'''
    def __init__(self, block=ResBlock):
        super(ResNet_34, self).__init__()
        # input sz: (224, 224)
        self.conv1 = nn.Conv2d(3, 64, 7, padding=3, stride=2)
        self.bn1 = nn.BatchNorm2d(64)

        # input sz: (112, 112)
        self.pool1 = nn.MaxPool2d(2, 2)

        # input sz: (56, 56)
        # 1st layer of blocks, 3 blocks, NO dimension change
        self.block_layer1 = nn.Sequential(
                    block(64, 64),
                    block(64, 64),
                    block(64, 64),
                )

        # input sz: (56, 56)
        # 2nd layer of blockes, 4 blocks, double output channels 
        self.block_layer2 = nn.Sequential(
                    block(64, 128, True),
                    block(128, 128),
                    block(128, 128),
                    block(128, 128),
                )

        # input sz: (28, 28)
        # 3rd layer of blockes, 6 blockes, double output channels
        self.block_layer3 = nn.Sequential(
                    block(128, 256, True),
                    block(256, 256),
                    block(256, 256),
                    block(256, 256),
                    block(256, 256),
                    block(256, 256),
                )

        # input sz: (14, 14)
        # 4th layer of blockes, 3 blocks, double output channels 
        self.block_layer4 = nn.Sequential(
                    block(256, 512, True),
                    block(512, 512),
                    block(512, 512),
                )

        # input sz: (7, 7)
        self.pool2 = nn.AvgPool2d(7)    

        # input sz: (1, 1)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.block_layer1(x)
        x = self.block_layer2(x)
        x = self.block_layer3(x)
        x = self.block_layer4(x)
        x = self.pool2(x)
        
        #x = x.view(-1, 64)
        #print('after pooling: ')
        #print(x.size())
        x = x.view(x.size(0), -1)
        #print('before fc: ')
        #print(x.size())
        x = self.fc(x)
        return x

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 96, 11, padding = 2, stride = 4)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(96)

        self.conv2 = nn.Conv2d(96, 256, 5, padding = 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(256)

        self.conv3 = nn.Conv2d(256, 384, 3, padding = 1)
        self.bn3 = nn.BatchNorm2d(384)

        self.conv4 = nn.Conv2d(384, 384, 3, padding = 1)
        self.bn4 = nn.BatchNorm2d(384)

        self.conv5 = nn.Conv2d(384, 256, 3, padding = 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.bn5 = nn.BatchNorm2d(256)

        self.fc1 = nn.Linear(6*6*256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 1000)

        self.drop = nn.Dropout()

    def forward(self, x):
        x = self.bn1(self.pool1(F.relu(self.conv1(x))))
        x = self.bn2(self.pool2(F.relu(self.conv2(x))))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.bn5(self.pool3(F.relu(self.conv5(x))))
        x = x.view(-1, 6*6*256)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x
