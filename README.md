# Requirements

安装mmsegmentation。官方教程：https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md#installation


其中mmsegmentation应以dev模式安装，从而可以直接替换修改mmsegmentation文件夹下的文件。


# Files

本repository存储了实验时用到的修改后的mmsegmentation文件夹。

其中出现变动的文件主要包括：

1. ./configs文件夹下：修改了DeepLabv3的配置文件，将SyncBN修改为BN，因为实验环境中只有一块GPU。同时修改了data对应的配置文件，将每块GPU上的图片数量设置为3。

2. 创建了包括train.sh在内的一系列脚本文件，存放在当前目录，用于训练模型，绘制损失函数，评价指标，获取FLOPs等。

# Running 

训练：

预测：

