from thop import profile
from thop import clever_format
from models.vit import TransAndConv
from models import *
import torch

from torchstat import stat

from torchsummary import summary

'''
本代码可被用于验证所计算的模型参数量与计算量
'''


net = TransAndConv()
# net = Sameflops()
# net = Sameparam()


# 导入模型，输入一张输入图片的尺寸
stat(net, (3, 32, 32))
