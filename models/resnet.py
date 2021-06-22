# -*- coding: utf-8 -*-

'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()

        self.in_planes = in_planes
        self.stride = stride
        self.planes = planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

        self.shortcut_conv = nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False)
        self.shortcut_bn = nn.BatchNorm2d(self.expansion*planes)
                

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.stride != 1 or self.in_planes != self.expansion*self.planes:
            x_tmp = self.shortcut_conv(x)
            x_tmp = self.shortcut_bn(x_tmp)
            out += x_tmp
        out = F.relu(out)
        return out

class SameParams(nn.Module):
    '''
    This model has approximately same amount of parameters with TransAndConv model.
    '''
    def __init__(self, block, num_classes=10):
        super(SameParams, self).__init__()
        
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, 4, stride=1) 
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 7, stride=2)
        self.linear = nn.Linear(256*4, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)  
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion # in_plain increased
        return nn.Sequential(*layers)

    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out) # removed last layer of resnet
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out) # classification
        return out

class SameFLOPs(nn.Module):  
    '''
    This model has approximately same FLOPs with TransAndConv model.
    '''
    def __init__(self, block, num_blocks, num_classes=10):
        super(SameFLOPs, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) 
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.linear = nn.Linear(256*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)  
        layers = []
        for stride in strides:
            print('stride, self.in_planes, planes', stride, self.in_planes, planes)
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion # increased in_plaine as network goes deeper
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def Sameparam():
    return SameParams(Bottleneck)

def Sameflops():
    return SameFLOPs(Bottleneck, [3, 4, 1,3])

# class ResNet(nn.Module):  # 在这份code里，不同resnet结构只是每个layer的block数量不一样
#     def __init__(self, block, num_blocks, num_classes=10):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1) # block = 3
#         # self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         # self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
#         self.linear = nn.Linear(64*block.expansion, num_classes)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1]*(num_blocks-1)  # stride总是1
#         layers = []
#         for stride in strides:
#             print('stride, self.in_planes, planes', stride, self.in_planes, planes)
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion # 经过一个block后，in_planes就增加为上一层的输出
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         # print('x', x.size())  # torch.Size([16, 3, 32, 32])  所以batch size应该是16吧
#         out = F.relu(self.bn1(self.conv1(x)))
#         # print('out', out.size()) # torch.Size([16, 64, 32, 32])
#         out = self.layer1(out)
#         # print('out_layer1', out.size()) # torch.Size([16, 256, 32, 32])
#         # out = self.layer2(out)
#         # print('out_layer2', out.size()) # torch.Size([16, 512, 16, 16])
#         # out = self.layer3(out)
#         # print('out_layer3', out.size()) # torch.Size([16, 1024, 8, 8])
#         # out = self.layer4(out)
#         # print('out_layer4', out.size()) # torch.Size([16, 2048, 4, 4])
#         out = F.avg_pool2d(out, 32)
#         # print('F.avg_pool2d', out.size()) # torch.Size([16, 2048, 1, 1])
#         out = out.view(out.size(0), -1)
#         # print('out.view', out.size()) # torch.Size([16, 2048])
#         out = self.linear(out)
#         # print('final out', out.size()) # torch.Size([16, 10])
#         return out

# def ResNet18():
#     return ResNet(BasicBlock, [1,1,1,1])

# def ResNet34():
#     return ResNet(BasicBlock, [3,4,6,3])

# def ResNet50():
#     return ResNet(Bottleneck, [3,4,6,3])

# def ResNet101():
#     return ResNet(Bottleneck, [3,4,23,3])

# def ResNet152():
#     return ResNet(Bottleneck, [3,8,36,3])


# def test():
#     net = ResNet18()
#     y = net(torch.randn(1,3,32,32))
#     print(y.size())

# test(),


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion*planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(self.expansion*planes)
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out