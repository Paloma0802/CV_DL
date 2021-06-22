# https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/vit_pytorch.py

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

MIN_NUM_PATCHES = 16

class Residual(nn.Module): # 涉及计算，没有参数
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        # x torch.Size([2, 65, 512])
        # self.fn(x, **kwargs) torch.Size([2, 65, 512])
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim) #  dim * 2  在特征维度归一化（计算前几个维度乘积个方差和均值）和缩放
    def forward(self, x, **kwargs):
        # x torch.Size([b, 65, 512])  # 所以它的输出是？
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # 【512， 512】
            nn.GELU(), # activate function
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), # 【512， 512】
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim=512, heads = 8, dropout = 0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias = False) # 【512，1536】
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim), # 【512，512】
            nn.Dropout(dropout)
        )

    def forward(self, x, mask = None):
        
        # x [b, 65, 512]
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)  # linear
        # qkv tuple, len 3, qkv[0], tensor, [b, 65, 512]
        # 512 -> 8*64, 8 heads

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv) # 调整size，获得qkv

        # q torch.Size([b, 8, 65, 64])
        # k torch.Size([b, 8, 65, 64])

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale # 计算点乘，得到相似性，然后乘以缩放因子
        # dots torch.Size([b, 8, 65, 65])

        if mask is not None:
            print('mask not none???')
            mask = F.pad(mask.flatten(1), (1, 0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        # attention score
        attn = dots.softmax(dim=-1)        
        # attn torch.Size([b, 8, 65, 65])
        
        # v torch.Size([b, 8, 65, 64])

        out = torch.einsum('bhij,bhjd->bhid', attn, v) # torch.Size([b, 65, 512])
        out = rearrange(out, 'b h n d -> b n (h d)') # torch.Size([b, 65, 512])
        out =  self.to_out(out) # linear + dropout # torch.Size([b, 65, 512])
        return out

class Transformer(nn.Module): #【b， 65， 512】
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads = heads, dropout = dropout))), # layernorm是在attention和feedforward之后的
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))) # linear， linear
            ]))
    def forward(self, x, mask = None):
        for attn, ff in self.layers: # attention and feedforward
            x = attn(x, mask = mask)
            x = ff(x)
        return x

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
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

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim=512, depth=6, heads, mlp_dim=512, channels = 3, dropout = 0., emb_dropout = 0., block = Bottleneck):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2 # 64
        # patch_dim = channels * patch_size ** 2
        patch_dim = 256 * patch_size ** 2 # 4069
        assert num_patches > MIN_NUM_PATCHES, f'your number of patches ({num_patches}) is way too small for attention to be effective. try decreasing your patch size'

        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False) # 做一个不会让feature map变小的卷积层
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks = 3, stride=1)

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim)) # [1, 4070, 512]
        # print('patch_dim, dim', patch_dim, dim)
        self.patch_to_embedding = nn.Linear(patch_dim, dim) # 4069,512
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim)) # [1, 1, 512]
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity() # 保存输入，不做处理

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim), # 512, 512
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, num_classes) # 512, 10
        )

    def _make_layer(self, block, planes, num_blocks, stride):

        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion # 每经过一个block，in_planes就增加为上一层的输出
        return nn.Sequential(*layers)

    def forward(self, img, mask = None):
        p = self.patch_size

        # print('original img', img.size())
        img = F.relu(self.bn1(self.conv1(img)))
        img = self.layer1(img) # [b, 256, 32, 32]

        # 重组张量的形状
        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p) # torch.Size([b, 64, 4096])
        x = self.patch_to_embedding(x) # linear层 # torch.Size([b, 64, 512])
        b, n, _ = x.shape        

        cls_tokens = self.cls_token.expand(b, -1, -1) # parameter
        x = torch.cat((cls_tokens, x), dim=1) # torch.Size([b, 65, 512])

        # self.pos_embedding[:, :(n + 1)] torch.Size([1, 65, 512])
        x += self.pos_embedding[:, :(n + 1)] # torch.Size([b, 65, 512])
        x = self.dropout(x) # torch.Size([b, 65, 512])

        x = self.transformer(x, mask)  # torch.Size([b, 65, 512]) 大头来了hhh

        # nn.idendity
        x = self.to_cls_token(x[:, 0]) # torch.Size([b, 512])

        x = self.mlp_head(x) # 有两个linear torch.Size([b, 10])

        return x



def TransAndConv():
    return ViT(
    image_size = 32,
    patch_size = 4, # 每个patch的大小
    num_classes = 10,
    dim = 512,
    depth = 3, 
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1, block = Bottleneck)