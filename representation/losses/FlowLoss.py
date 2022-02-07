import torch
from typing import List


class NormalizedMaskedL2Loss(torch.nn.Module):
    def __init__(self, eps=1.0e-2):
        super(NormalizedMaskedL2Loss, self).__init__()
        self.eps = eps

    def forward(self, x):
        gt_flow, pred_flow, mask = x

        _, _, height, width = gt_flow.shape

        gt_flow = torch.cat([gt_flow[:, 0:1, :, :] / width, gt_flow[:, 1:2, :, :] / height], dim=1)

        # N x 2 x H x W
        if isinstance(pred_flow, List):
            pred_flow = torch.cat(pred_flow, dim=0)

        pred_flow = torch.cat([pred_flow[:, 0:1, :, :] / width,
                               pred_flow[:, 1:2, :, :] / height], dim=1)

        with torch.no_grad():
            # B
            mean_flow_magnitude = 0.5 * (torch.sum(mask * torch.square(gt_flow), dim=(1, 2, 3)) / (
                    1.0 + torch.sum(mask, dim=(1, 2, 3))) + torch.sum(mask * torch.square(pred_flow),
                                                                      dim=(1, 2, 3)) / (
                                                 1.0 + torch.sum(mask, dim=(1, 2, 3)))) + self.eps

        loss = torch.sum(mask * torch.square(gt_flow - pred_flow), dim=(1, 2, 3)) \
               / (mean_flow_magnitude * (torch.sum(mask, dim=(1, 2, 3)) + 1.0))

        # mean_flow_magnitude *
        return torch.mean(loss)
