import torch
import torch.nn.functional
import torch.nn as nn

from .partial_modules import PartialDownConv, PartialUpConv, PartialBlock


class FeatureNet(nn.Module):
    def __init__(self, in_channel, num_pre_steps, filter_list,
                 bottle_neck_filter, desc_inner_filter_list,
                 map_inner_filter_list, desc_out_activation, map_out_activation,
                 merge_mode='concat', gn_group_size=4):
        super(FeatureNet, self).__init__()

        if merge_mode in ('concat', 'add'):
            self.merge_mode = merge_mode
        else:
            raise ValueError("\"{}\" is not a valid mode for"
                             "merging up and down paths. "
                             "Only \"concat\" and "
                             "\"add\" are allowed.".format(merge_mode))

        self.down_convs = torch.nn.ModuleList()
        self.pre_down_convs = torch.nn.ModuleList()
        self.up_convs = torch.nn.ModuleList()
        self.feat_map_convs = torch.nn.ModuleList()
        self.feat_desc_convs = torch.nn.ModuleList()
        self.desc_mid_convs = torch.nn.ModuleList()

        self.total_feat_channel = 0

        pre_filter_list = filter_list[:num_pre_steps]
        inner_filter_list = filter_list[num_pre_steps:]
        enc_pre_filter_list = [in_channel] + pre_filter_list

        for i in range(len(enc_pre_filter_list) - 1):
            pre_down_conv = PartialDownConv(enc_pre_filter_list[i], enc_pre_filter_list[i + 1],
                                            pooling=True)
            self.pre_down_convs.append(pre_down_conv)

        enc_filter_list = [enc_pre_filter_list[-1]] + inner_filter_list
        # create the encoder pathway
        for i in range(len(enc_filter_list) - 1):
            down_conv = PartialDownConv(enc_filter_list[i], enc_filter_list[i + 1],
                                        pooling=True)
            self.down_convs.append(down_conv)

        # create bottleneck layer
        self.bottle_neck = PartialBlock(enc_filter_list[-1], bottle_neck_filter, group_size=gn_group_size,
                                        out_activation="relu")

        # create the decoder pathway
        dec_filter_list = [bottle_neck_filter] + \
            list(reversed(inner_filter_list))
        for i in range(len(dec_filter_list) - 1):
            up_conv = PartialUpConv(dec_filter_list[i] + enc_filter_list[-i - 1], dec_filter_list[i + 1],
                                    merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        desc_inner_filter_list = [dec_filter_list[-1]] + desc_inner_filter_list
        for i in range(len(desc_inner_filter_list) - 1):
            if i == len(desc_inner_filter_list) - 2:
                self.feat_desc_convs.append(
                    PartialBlock(desc_inner_filter_list[i], desc_inner_filter_list[i + 1],
                                 group_size=gn_group_size, out_activation=desc_out_activation))
            else:
                self.feat_desc_convs.append(
                    PartialBlock(desc_inner_filter_list[i], desc_inner_filter_list[i + 1], group_size=gn_group_size,
                                 out_activation="relu"))

        map_inner_filter_list = [dec_filter_list[-1]] + map_inner_filter_list
        for i in range(len(map_inner_filter_list) - 1):
            if i == len(map_inner_filter_list) - 2:
                self.feat_map_convs.append(
                    PartialBlock(map_inner_filter_list[i], map_inner_filter_list[i + 1],
                                 group_size=gn_group_size, out_activation=map_out_activation))
            else:
                self.feat_map_convs.append(
                    PartialBlock(map_inner_filter_list[i], map_inner_filter_list[i + 1], group_size=gn_group_size,
                                 out_activation="relu"))

    @torch.jit.export
    def forward(self, x, mask):
        encoder_outs = []
        encoder_masks = []

        for module in self.pre_down_convs:
            x, _, mask = module(x, mask)

        for i, module in enumerate(self.down_convs):
            encoder_masks.append(mask)
            x, pre_pool, mask = module(x, mask)
            encoder_outs.append(pre_pool)

        x, mask = self.bottle_neck(x, mask)

        for i, module in enumerate(self.up_convs):
            enc_out = encoder_outs[-(i + 1)]
            mask = encoder_masks[-(i + 1)]
            x, _ = module(enc_out, x, mask)

        mask = encoder_masks[0]

        feature_desc = x
        feature_map = x
        for module in self.feat_desc_convs:
            feature_desc, mask = module(feature_desc, mask)

        for module in self.feat_map_convs:
            feature_map, mask = module(feature_map, mask)

        return feature_map, feature_desc
