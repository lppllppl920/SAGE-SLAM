import torch
import torch.nn as nn

from .partial_modules import PartialDownConv, PartialUpConv, PartialBlock, PartialDownConvNoPre


class DepthNet(nn.Module):
    def __init__(self, in_channel, num_pre_steps, filter_list,
                 bottle_neck_filter, bias_inner_filter_list,
                 basis_inner_filter_list,
                 bias_out_activation, basis_out_activation,
                 merge_mode='concat', gn_group_size=4):
        super(DepthNet, self).__init__()

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
        self.dpt_bias_convs = torch.nn.ModuleList()

        self.gn_group_size = gn_group_size

        self.dpt_basis_convs_hierarchy = torch.nn.ModuleDict()
        self.basis_inner_filter_list_hierarchy = basis_inner_filter_list
        self.num_basis_levels = len(self.basis_inner_filter_list_hierarchy)

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
        assert (self.num_basis_levels <= len(dec_filter_list))

        for i in range(len(dec_filter_list) - 1):
            up_conv = PartialUpConv(dec_filter_list[i] + enc_filter_list[-i - 1], dec_filter_list[i + 1],
                                    merge_mode=merge_mode)
            self.up_convs.append(up_conv)

        for id, basis_filter_list in enumerate(self.basis_inner_filter_list_hierarchy):
            basis_filter_list = [dec_filter_list[-1]] + basis_filter_list
            self.dpt_basis_convs_hierarchy[f"basis_{id:d}"] = \
                self.generate_convs_block(inner_filter_list=basis_filter_list,
                                          out_activation=basis_out_activation, pool_factor=int(2 ** id))

        bias_inner_filter_list = [dec_filter_list[-1]] + bias_inner_filter_list
        for i in range(len(bias_inner_filter_list) - 1):
            if i == len(bias_inner_filter_list) - 2:
                self.dpt_bias_convs.append(
                    PartialBlock(bias_inner_filter_list[i], bias_inner_filter_list[i + 1],
                                 group_size=gn_group_size, out_activation=bias_out_activation))
            else:
                self.dpt_bias_convs.append(
                    PartialBlock(bias_inner_filter_list[i], bias_inner_filter_list[i + 1], group_size=gn_group_size,
                                 out_activation="relu"))

        self.dec_filter_list = dec_filter_list

    def generate_convs_block(self, inner_filter_list, out_activation, pool_factor):
        convs_list = torch.nn.ModuleList()
        for i in range(len(inner_filter_list) - 1):
            if i == 0:
                convs_list.append(
                    PartialDownConvNoPre(inner_filter_list[i], inner_filter_list[i + 1], group_size=self.gn_group_size,
                                         pool_factor=pool_factor, pooling=True if (pool_factor > 1) else False))
            elif i == len(inner_filter_list) - 2:
                convs_list.append(
                    PartialBlock(inner_filter_list[i], inner_filter_list[i + 1],
                                 group_size=self.gn_group_size, out_activation=out_activation))
            else:
                convs_list.append(
                    PartialBlock(inner_filter_list[i], inner_filter_list[i + 1], group_size=self.gn_group_size,
                                 out_activation="relu"))

        return convs_list

    def forward_train(self, x, mask, return_basis):
        out_dpt_basis_hierarchy = list()
        encoder_outs = list()
        encoder_masks = list()

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
            x, mask = module([enc_out, x, mask])

        if return_basis:
            # basis branch
            for id in range(self.num_basis_levels):
                mask = encoder_masks[0]
                dpt_basis = x
                for module in self.dpt_basis_convs_hierarchy[
                        f"basis_{id:d}"]:
                    dpt_basis, mask = module(dpt_basis, mask)

                out_dpt_basis_hierarchy.append(dpt_basis)

        # bias branch
        mask = encoder_masks[0]
        out_dpt_bias = x
        for module in self.dpt_bias_convs:
            out_dpt_bias, mask = module(out_dpt_bias, mask)

        if return_basis:
            return out_dpt_bias, out_dpt_basis_hierarchy
        else:
            return out_dpt_bias

    def forward(self, x, mask):
        out_dpt_basis_hierarchy = list()
        encoder_outs = list()
        encoder_masks = list()

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
            x, mask = module([enc_out, x, mask])

        # basis branch
        for id in range(self.num_basis_levels):
            mask = encoder_masks[0]
            dpt_basis = x
            for module in self.dpt_basis_convs_hierarchy[
                    f"basis_{id:d}"]:
                dpt_basis, mask = module(dpt_basis, mask)

            out_dpt_basis_hierarchy.append(dpt_basis)

        # bias branch
        mask = encoder_masks[0]
        out_dpt_bias = x
        for module in self.dpt_bias_convs:
            out_dpt_bias, mask = module(out_dpt_bias, mask)

        return out_dpt_bias, torch.cat(out_dpt_basis_hierarchy, dim=1)
