import torch
import torch.nn as nn

from .vn_occupancy import VNOccupancyNet
from util.registration import batch_procrustes
from util.metric import compute_registration_loss, compute_occupancy_loss

class CoarseRegister(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.decode = args.decode

        if self.decode:
            self.decoder = VNOccupancyNet(c_dim=128*3)


    def forward(self, src_feat, tgt_feat, src_p=None, tgt_p=None, decode=False):
        self.r_pred, self.t_pred = batch_procrustes(src_feat, tgt_feat)

        if decode:
            assert src_p is not None and tgt_p is not None, 'Please pass sampled points'
            self.src_occ_pred = self.decoder(src_p, src_feat)
            self.tgt_occ_pred = self.decoder(tgt_p, tgt_feat)

        return self.r_pred, self.t_pred


    def compute_loss(self, r_gt, t_gt, src_occ_gt, tgt_occ_gt):
        registration_loss = compute_registration_loss(self.r_pred, self.t_pred, r_gt, t_gt)

        if self.decode:
            occupancy_loss = compute_occupancy_loss(self.src_occ_pred, src_occ_gt) \
                           + compute_occupancy_loss(self.tgt_occ_pred, tgt_occ_gt)
        else:
            occupancy_loss = 0.0

        return 10 * registration_loss + 0.1 * occupancy_loss
