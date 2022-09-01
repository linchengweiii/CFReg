import torch
import torch.nn as nn

from model.feature_extractor import SE3EquivariantNetwork
from model.coarse_register import CoarseRegister
from model.fine_register import FineRegister
from util.transformation import se3_transform

class CoarseToFineNetwork(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.device = None
        self.decode = args.decode
        self.coarse = 'coarse' in args.modules
        self.fine = 'fine' in args.modules

        self.feature_extractor = SE3EquivariantNetwork(args)
        if self.coarse:
            self.coarse_register = CoarseRegister(args)
        if self.fine:
            self.fine_register = FineRegister(args)


    def register(self, src_pcd, tgt_pcd, src_p=None, tgt_p=None, r_gt=None, t_gt=None, decode=False):
        if self.training:
            src_gt = se3_transform(src_pcd, r_gt, t_gt)
        else:
            src_gt = None

        src_global_feat, src_local_feat = self.feature_extractor(src_pcd)
        tgt_global_feat, tgt_local_feat = self.feature_extractor(tgt_pcd)

        if self.coarse:
            self.r_coarse, self.t_coarse = self.coarse_register(
                    src_global_feat, tgt_global_feat, src_p, tgt_p, decode=decode)
        else:
            batch_size = src_pcd.size(0)
            self.r_coarse = torch.eye(3, device=self.device).expand(batch_size, -1, -1)
            self.t_coarse = torch.zeros(3, device=self.device).expand(batch_size, -1)

        src_pcd = se3_transform(src_pcd, self.r_coarse, self.t_coarse)

        if self.fine:
            self.r_fine, self.t_fine = self.fine_register(
                src_pcd, tgt_pcd, src_local_feat, tgt_local_feat, src_gt)
        else:
            batch_size = src_pcd.size(0)
            self.r_fine = torch.eye(3, device=self.device).expand(batch_size, -1, -1)
            self.t_fine = torch.zeros(3, device=self.device).expand(batch_size, -1)

        self.r_pred = self.r_fine @ self.r_coarse
        self.t_pred = (self.r_fine @ self.t_coarse.unsqueeze(-1)).squeeze() + self.t_fine

        return self.r_pred, self.t_pred


    def forward(self, src_pcd, tgt_pcd, src_p, tgt_p, r_gt, t_gt):
        return self.register(src_pcd, tgt_pcd, src_p, tgt_p, r_gt, t_gt, decode=self.decode)


    def compute_loss(self, src, tgt, r_gt, t_gt, src_occ_gt, tgt_occ_gt):
        if self.coarse:
            coarse_loss = self.coarse_register.compute_loss(
                    r_gt, t_gt, src_occ_gt, tgt_occ_gt)
        else:
            coarse_loss = 0.0

        src_gt = se3_transform(src, r_gt, t_gt)
        src = se3_transform(src, self.r_coarse, self.t_coarse)

        if self.fine:
            fine_loss = self.fine_register.compute_loss()
        else:
            fine_loss = 0.0

        return coarse_loss + fine_loss


    def to(self, device):
        self.device = device
        return super().to(device)
