# Requirements

环境：

PyTorch 1.3.1

cuda 10.0


安装mmsegmentation。官方教程：https://github.com/open-mmlab/mmsegmentation/blob/master/docs/get_started.md#installation


其中mmsegmentation应以dev模式安装，从而可以直接替换修改mmsegmentation文件夹下的文件。

另需创建data文件夹，存放在mmsegmentation下。向其中放入cityscapes数据集，包含gtFine与leftImg8bit两个文件夹，分别存放图像与标注信息。

创建work_dirs文件夹，存放在mmsegmentation下。在其中创建名为deeplabv3plus_r101-d16-mg124_512x1024_40k_cityscapes的文件夹，在其中存放训练完毕的模型与训练日志。


# Files

本repository存储了实验时用到的修改后的mmsegmentation文件夹内的内容。

其中出现变动的文件主要包括：

1. ./configs文件夹下：修改了DeepLabv3的配置文件，将SyncBN修改为BN，因为实验环境中只有一块GPU。同时修改了data对应的配置文件，将每块GPU上的图片数量设置为3。

2. 创建了包括train.sh在内的一系列脚本文件，存放在当前目录，用于训练模型，绘制损失函数，评价指标，获取FLOPs等。

# Running 

训练：运行train.sh脚本。

预测：运行test.sh脚本。

