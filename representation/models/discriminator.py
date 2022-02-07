import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def conv3x3(in_channels, out_channels, stride=1,
            padding=1, bias=True, groups=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


class Block(nn.Module):
    """
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, group_size=4):
        super(Block, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ngroups = int(np.maximum(1, out_channels // group_size))

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn = nn.GroupNorm(num_groups=self.ngroups,
                               num_channels=self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.conv2(x))
        return x


class DownBlock(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, group_size=4, pooling_mode='max'):
        super(DownBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ngroups = int(np.maximum(1, out_channels // group_size))

        self.conv1 = conv3x3(self.in_channels, self.out_channels)
        self.bn = nn.GroupNorm(num_groups=self.ngroups,
                               num_channels=self.out_channels)
        self.conv2 = conv3x3(self.out_channels, self.out_channels)

        if pooling_mode == "max":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pooling_mode == "avg":
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            raise ValueError(f"pooling mode {pooling_mode} is not supported")

    def forward(self, x):
        x = F.relu(self.bn(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x


class DiscNet(nn.Module):
    def __init__(self, img_height, img_width, input_nc, filter_base, num_block, group_size):
        super(DiscNet, self).__init__()
        self.blocks = torch.nn.ModuleList()
        self.blocks.append(DownBlock(in_channels=input_nc, out_channels=filter_base, group_size=group_size,
                                     pooling_mode='max'))
        self.blocks.append(Block(in_channels=filter_base,
                           out_channels=filter_base, group_size=group_size))

        for i in range(num_block - 1):
            self.blocks.append(DownBlock(in_channels=filter_base * 2 ** i, out_channels=filter_base * 2 ** (i + 1),
                                         group_size=group_size,
                                         pooling_mode='max'))
            self.blocks.append(Block(in_channels=filter_base * 2 ** (i + 1), out_channels=filter_base * 2 ** (i + 1),
                                     group_size=group_size))

        # The height and width of downsampled image
        height = img_height // 2 ** num_block
        width = img_width // 2 ** num_block
        self.final_conv = nn.Conv2d(in_channels=filter_base * 2 ** (num_block - 1), out_channels=1,
                                    kernel_size=1, stride=1, padding=0, bias=True)
        self.adv_layer = nn.Linear(height * width, 1)

    def forward(self, x):
        for module in self.blocks:
            x = module(x)
        x = self.final_conv(x)
        validity = self.adv_layer(x.reshape(x.shape[0], -1))
        return validity
