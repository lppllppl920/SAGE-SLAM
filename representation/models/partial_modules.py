import torch
import torch.nn.functional
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(PartialConv2d, self).__init__(*args, **kwargs)

        self.weight_maskUpdater = torch.ones(
            1, 1, self.kernel_size[0], self.kernel_size[1]).cuda()

        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * \
            self.weight_maskUpdater.shape[3]

    def forward(self, input, mask):
        with torch.no_grad():
            assert (self.stride == (1, 1) and self.dilation == (1, 1))
            update_mask = F.conv2d(mask, weight=self.weight_maskUpdater, bias=None, stride=self.stride, 
                                        padding=self.padding, dilation=self.dilation, groups=1) / \
                self.slide_winsize
            binary_mask = (update_mask >= 0.01).float()

        raw_out = self._conv_forward(input * mask, self.weight, self.bias)

        # NOTE: optional argument needs to be first assigned to a local variable and then use if-else statment to inform the compiler 
        # https://pytorch.org/docs/stable/jit_language_reference.html
        bias = self.bias
        if bias is not None:
            bias = bias.reshape(1, self.out_channels, 1, 1)
            output = torch.div(raw_out - bias, update_mask + 1.0e-8) + bias
            output = output * binary_mask
        else:
            output = torch.div(raw_out, update_mask + 1.0e-8)
            output = output * binary_mask

        return output, binary_mask



def partial_conv3x3(in_channels, out_channels, stride=1,
                    padding=1, bias=True, groups=1):
    return PartialConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        bias=bias,
        groups=groups)


class PartialDownConvNoPre(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True, group_size=4, pool_factor=2):
        super(PartialDownConvNoPre, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.ngroups = int(np.maximum(1, out_channels // group_size))

        self.conv1 = partial_conv3x3(
            self.in_channels, self.out_channels)
        self.bn = nn.GroupNorm(num_groups=self.ngroups,
                               num_channels=self.out_channels)
        self.conv2 = partial_conv3x3(
            self.out_channels, self.out_channels)
        self.pool_factor = pool_factor

        self.mask_pool = nn.MaxPool2d(
            kernel_size=pool_factor, stride=pool_factor)
        self.pool = nn.MaxPool2d(
            kernel_size=pool_factor, stride=pool_factor)

    def forward(self, x, mask):
        x, mask = self.conv1(x, mask)
        x = F.relu(self.bn(x))
        x, mask = self.conv2(x, mask)
        x = F.relu(x)

        if self.pooling:
            mask = self.mask_pool(mask)
            x = self.pool(x)

        return x, mask


class PartialDownConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, pooling=True, group_size=4, pool_factor=2):
        super(PartialDownConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.ngroups = int(np.maximum(1, out_channels // group_size))

        self.conv1 = partial_conv3x3(
            self.in_channels, self.out_channels)
        self.bn = nn.GroupNorm(num_groups=self.ngroups,
                               num_channels=self.out_channels)
        self.conv2 = partial_conv3x3(
            self.out_channels, self.out_channels)
        self.pool_factor = pool_factor

        if self.pooling:
            self.mask_pool = nn.MaxPool2d(
                kernel_size=pool_factor, stride=pool_factor)
            self.pool = nn.MaxPool2d(
                kernel_size=pool_factor, stride=pool_factor)

    def forward(self, x, mask):
        x, mask = self.conv1(x, mask)
        x = F.relu(self.bn(x))
        x, mask = self.conv2(x, mask)
        x = F.relu(x)

        pre_pool = x
        if self.pooling:
            mask = self.mask_pool(mask)
            x = self.pool(x)

        return x, pre_pool, mask


class PartialBlock(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 MaxPool.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels, out_activation, group_size=4, eps=1.0e-8):
        super(PartialBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ngroups = int(np.maximum(1, out_channels // group_size))

        self.conv1 = partial_conv3x3(
            self.in_channels, self.out_channels)
        self.bn = nn.GroupNorm(num_groups=self.ngroups,
                               num_channels=self.out_channels)
        self.conv2 = partial_conv3x3(
            self.out_channels, self.out_channels)
        self.out_activation = out_activation
        self.eps = eps

    def forward(self, x, mask):
        x, mask = self.conv1(x, mask)
        x = torch.relu(self.bn(x))
        x, mask = self.conv2(x, mask)

        if self.out_activation.lower() == "relu":
            x = torch.relu(x)
        elif self.out_activation.lower() == "normalize":
            x = (x + self.eps) / (torch.linalg.norm(x +
                                                    self.eps, ord=None, dim=1, keepdim=True))
        elif self.out_activation.lower() == "tanh":
            x = torch.tanh(x)
        elif self.out_activation.lower() == "linear":
            pass
        elif self.out_activation.lower() == "abs":
            x = torch.abs(x)
        elif self.out_activation.lower() == "sigmoid":
            x = torch.sigmoid(x)

        return x, mask


class PartialUpConv(nn.Module):
    """
    A helper Module that performs 2 convolutions and 1 UpConvolution.
    A ReLU activation follows each convolution.
    """

    def __init__(self, in_channels, out_channels,
                 merge_mode='concat', group_size=4):
        super(PartialUpConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.merge_mode = merge_mode
        self.ngroups = int(np.maximum(1, out_channels // group_size))

        self.upsample = nn.Upsample(mode='nearest', scale_factor=2)

        self.conv1 = partial_conv3x3(self.in_channels, self.out_channels)

        self.bn = nn.GroupNorm(num_groups=self.ngroups,
                               num_channels=self.out_channels)
                               
        self.conv2 = partial_conv3x3(self.out_channels, self.out_channels)

    def forward(self, enc_output, dec_output, mask):
        dec_output = self.upsample(dec_output)

        if self.merge_mode == 'concat':
            x = torch.cat([dec_output, enc_output], dim=1)
        else:
            x = dec_output + enc_output

        x, mask = self.conv1(x, mask)
        x = F.relu(self.bn(x))
        x, mask = self.conv2(x, mask)
        x = F.relu(x)
        return x, mask
