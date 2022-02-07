import torch


class ScaleInvariantLoss(torch.nn.Module):
    def __init__(self, epsilon=1.0e-3):
        super(ScaleInvariantLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        # gt_depths: 1 x 1 x H x W
        gt_depths, pred_depths, masks = x

        # N x 1 x H x W
        depth_ratio_map = torch.log(torch.clamp_min(masks * pred_depths, min=self.epsilon)) - \
                          torch.log(torch.clamp_min(masks * gt_depths, min=self.epsilon))

        weighted_sum = torch.sum(masks, dim=(1, 2, 3))
        loss_1 = torch.sum(depth_ratio_map * depth_ratio_map,
                           dim=(1, 2, 3)) / weighted_sum
        sum_2 = torch.sum(depth_ratio_map, dim=(1, 2, 3))
        loss_2 = (sum_2 * sum_2) / (weighted_sum * weighted_sum)
        return torch.mean(loss_1 + loss_2)


class BasisDecorrelationLoss(torch.nn.Module):
    def __init__(self):
        super(BasisDecorrelationLoss, self).__init__()

    def forward(self, depth_basis, mask):
        # depth_basis B x C x H x W
        batch, channel, height, width = depth_basis.shape
        # B x C x 1 x 1
        mean_per_channel = torch.mean(depth_basis, dim=(2, 3), keepdim=True)

        centered_depth_basis = depth_basis - mean_per_channel

        flatten_mask = mask.reshape(batch, 1, 1, height * width)
        # B x C x 1 x H*W - B x 1 x C x H*W -> B x C x C x H*W -> B x C x C
        basis_covariance = torch.sum((centered_depth_basis.reshape(batch, channel, 1, height * width) *
                                      centered_depth_basis.reshape(batch, 1, channel, height * width)) * flatten_mask,
                                     dim=3,
                                     keepdim=False) / torch.sum(flatten_mask, dim=3, keepdim=False)

        # B x C
        basis_covariance = torch.clamp_min(basis_covariance, min=1.0e-10)
        basis_sigma = torch.sqrt(torch.diagonal(basis_covariance, dim1=-2, dim2=-1))
        # B x C x C
        basis_zncc = basis_covariance / (basis_sigma.reshape(batch, channel, 1) *
                                         basis_sigma.reshape(batch, 1, channel))

        loss = torch.mean(torch.square(basis_zncc))

        return loss