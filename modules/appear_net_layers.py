import torch
import torch.nn as nn
import torchvision

from .dropblock import DropBlock2D
from .vgg import vgg16_bn_128, vgg16_bn_256, vgg16_bn_512  # noqa


class SkipPool(nn.Module):

    def __init__(self, channels, reduction, out_channels, dropblock_size=5):
        super(SkipPool, self).__init__()
        self.channels = channels
        self.reduction = reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropblock = None
        if dropblock_size:
            self.dropblock = DropBlock2D(block_size=dropblock_size)
        self.fc = nn.Sequential(
            nn.GroupNorm(1, channels),
            nn.Conv2d(channels, max(channels // reduction, 64), 1, 1),
            nn.GroupNorm(1, max(channels // reduction, 64)),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 64), out_channels, 1, 1),
            nn.GroupNorm(1, out_channels), nn.ReLU(inplace=True))

    def forward(self, x):
        if self.dropblock is not None:
            x = self.dropblock(x)
        out = self.avg_pool(x)
        out = self.fc(out).view((x.size(0), -1))
        return out

def parse_vgg_layers(loaded_model,dropblock,out_channels):
    layers = []
    blocks = []
    pool_layers = []
    channels = 0
    first = 0
    block_size = 0
    for m in loaded_model.features.children():
        blocks.append(m)
        if isinstance(m, nn.MaxPool2d):
            first += 1
            if first == 1:
                continue
            if first > 3 and dropblock:
                block_size = dropblock
            layers.append(nn.Sequential(*blocks))
            blocks = []
            pool_layers.append(
                make_scalar_layer(
                    channels,
                    4,
                    out_channels // 4,
                    dropblock_size=block_size))

        elif isinstance(m, nn.Conv2d):
            channels = m.out_channels

    return nn.ModuleList(layers), nn.ModuleList(pool_layers)

def make_scalar_layer( channels,
                        reduction,
                        out_channels,
                        dropblock_size=5):
    return SkipPool(channels, reduction, out_channels, dropblock_size)

def parse_appearance_layers(
                arch='vgg',
                out_channels=512,
                skippool=True,
                fpn = False,
                dropblock=5):
    reduction = 512 // out_channels
    if arch == 'vgg':
        base_channel = 64 // reduction
        vgg_net = eval("vgg16_bn_%s" % str(out_channels))
        loaded_model = vgg_net()
        if skippool:
            print("use Skip Pooling in appearance model")
            layers, global_pool = parse_vgg_layers(
                loaded_model,dropblock,out_channels)
            
    return layers, global_pool 
