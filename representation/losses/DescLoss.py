import torch
import torch.nn.functional


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, x):
        # K x C x 1
        src_cdf, tgt_cdf, far_cdf = x
        # C x 1
        pos_dist = torch.mean(torch.square(src_cdf - tgt_cdf), dim=0, keepdim=False)
        neg_dist = torch.mean(torch.square(src_cdf - far_cdf), dim=0, keepdim=False)

        return torch.mean(torch.relu(pos_dist - neg_dist + self.margin))