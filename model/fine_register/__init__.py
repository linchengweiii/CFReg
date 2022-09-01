import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import svd

from .util import batch_choice, square_distance


class Conv1DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv1DBNReLU, self).__init__()
        self.conv = nn.Conv1d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv1DBlock(nn.Module):
    def __init__(self, channels, ksize=1):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels)-2):
            self.conv.append(Conv1DBNReLU(channels[i], channels[i+1], ksize))
        self.conv.append(nn.Conv1d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


class Conv2DBNReLU(nn.Module):
    def __init__(self, in_channel, out_channel, ksize):
        super(Conv2DBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, ksize, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Conv2DBlock(nn.Module):
    def __init__(self, channels, ksize=1):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.ModuleList()
        for i in range(len(channels)-2):
            self.conv.append(Conv2DBNReLU(channels[i], channels[i+1], ksize))
        self.conv.append(nn.Conv2d(channels[-2], channels[-1], ksize))

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        return x


def weighted_svd(src, src_corr, weights):
    ### hybrid point elimination
    weights = torch.sigmoid(weights)
    weights = weights * (weights >= weights.median(-1, keepdim=True)[0]).float()
    weights = weights / (weights.sum(-1, keepdim=True) + 1e-8)

    ### weighted svd
    src_centered = src - src.mean(dim=2, keepdim=True)
    src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

    H = torch.matmul(src_centered * weights.unsqueeze(1), src_corr_centered.transpose(2, 1).contiguous())
    U, D, V = svd(H)
    S = torch.eye(3, device=H.device).unsqueeze(0).repeat(src.size(0), 1, 1)
    S[U.det() * V.det() < 0, -1, -1] = -1

    R = V @ (S @ U.transpose(1, 2))
    t = torch.matmul(-R, (weights.unsqueeze(1) * src).sum(dim=2, keepdim=True)) + \
        (weights.unsqueeze(1) * src_corr).sum(dim=2, keepdim=True)

    return R, t.view(src.size(0), 3)


class FineRegister(nn.Module):
    eps = 1e-8
    def __init__(self, args, feat_dim=64):
        super().__init__()
        self.feat_dim = feat_dim
        self.sig_fc = Conv1DBlock([feat_dim, 64, 32, 1])
        self.sim_feat_conv = Conv2DBlock([feat_dim*2+4, 32, 32])
        self.sim_mat_conv = Conv2DBlock([32, 32, 1])
        self.weight_fc = Conv1DBlock([32, 32, 1])

    def register_iter(self, it, src, tgt, stack_feat, match_labels=None,
                      src_sig_score=None, tgt_sig_score=None, src_gt=None):
        ### compute distances
        diff = src.unsqueeze(-1) - tgt.unsqueeze(-2)
        dist = torch.sqrt((diff ** 2).sum(1, keepdim=True))
        diff = diff / (dist + self.eps)

        ### similarity matrix convolution to get features
        sim_feat = torch.cat([stack_feat, dist, diff], 1)
        sim_feat = self.sim_feat_conv(sim_feat)
        
        ### soft point elimination
        weights = sim_feat.max(-1)[0]
        weights = self.weight_fc(weights).squeeze(1)
        
        ### similarity matrix convolution to get similarities
        sim_mat = self.sim_mat_conv(sim_feat)
        sim_mat = sim_mat.squeeze(1)
        sim_mat = sim_mat.clamp(min=-20, max=20)

        ### finding correspondences
        corr_idx = sim_mat.max(-1)[1]
        src_corr = tgt[np.arange(tgt.size(0))[:, np.newaxis], :, corr_idx].transpose(1, 2)

        ### get rotation and translation
        rotation, translation = weighted_svd(src, src_corr, weights)
        rotation = rotation.detach()
        translation = translation.detach()

        ### compute loss
        if self.training:
            ### negative entropy loss
            if it == 0:
                src_neg_ent = torch.softmax(sim_mat, dim=-1)
                src_neg_ent = (src_neg_ent * torch.log(src_neg_ent)).sum(-1)
                tgt_neg_ent = torch.softmax(sim_mat, dim=-2)
                tgt_neg_ent = (tgt_neg_ent * torch.log(tgt_neg_ent)).sum(-2)
                neg_ent_loss = F.mse_loss(src_sig_score, src_neg_ent.detach()) \
                             + F.mse_loss(tgt_sig_score, tgt_neg_ent.detach())
            else:
                neg_ent_loss = 0.0

            ### match loss
            temp = torch.softmax(sim_mat, dim=-1)
            temp = temp[:, np.arange(temp.size(-2)), np.arange(temp.size(-1))]
            temp = -torch.log(temp)
            match_loss = (temp * match_labels).sum() / match_labels.sum()
            
            ### weight label
            weight_labels = (corr_idx == torch.arange(corr_idx.size(1)).to(src.device).unsqueeze(0)).float()

            ### weight loss
            weight_loss = F.binary_cross_entropy_with_logits(weights, weight_labels)

            ### update loss
            self.loss += neg_ent_loss + match_loss + weight_loss

        return rotation, translation


    def forward(self, src, tgt, src_feat, tgt_feat, src_gt=None):
        B, D, N = src_feat.size()

        ### get correspondence labels
        if self.training:
            dist = src_gt.unsqueeze(-1) - tgt.unsqueeze(-2)
            min_dist, min_idx = (dist ** 2).sum(1).min(-1)
            min_dist = torch.sqrt(min_dist)
            min_idx = min_idx.cpu().numpy() # drop to cpu for numpy
            match_labels = (min_dist < 0.05).float()
            indicator = match_labels.cpu().numpy()
            indicator += 1e-5
            pos_probs = indicator / indicator.sum(-1, keepdims=True)
            indicator = 1 + 1e-5 * 2 - indicator
            neg_probs = indicator / indicator.sum(-1, keepdims=True)
        else:
            match_labels = None

        ### get significance score
        src_sig_score = self.sig_fc(src_feat).squeeze(1)
        tgt_sig_score = self.sig_fc(tgt_feat).squeeze(1)

        ### hard point elimination
        num_preserved = N // 6
        if self.training:
            candidates = np.tile(np.arange(src.size(-1)), (src.size(0), 1))
            pos_idx = batch_choice(candidates, num_preserved // 2,  p=pos_probs)
            neg_idx = batch_choice(candidates, num_preserved - num_preserved // 2, p=neg_probs)
            src_idx = np.concatenate([pos_idx, neg_idx], 1)
            tgt_idx = min_idx[np.arange(len(src))[:, np.newaxis], src_idx]
        else:
            src_idx = src_sig_score.topk(k=num_preserved, dim=-1)[1]
            src_idx = src_idx.cpu().numpy()
            tgt_idx = tgt_sig_score.topk(k=num_preserved, dim=-1)[1]
            tgt_idx = tgt_idx.cpu().numpy()
        batch_idx = np.arange(B)[:, np.newaxis]
        if self.training:
            match_labels = match_labels[batch_idx, src_idx]
        else:
            match_labels = None
        src_select = src[batch_idx, :, src_idx].transpose(1, 2)
        src_feat = src_feat[batch_idx, :, src_idx].transpose(1, 2)
        src_sig_score = src_sig_score[batch_idx, src_idx]
        tgt_select = tgt[batch_idx, :, tgt_idx].transpose(1, 2)
        tgt_feat = tgt_feat[batch_idx, :, tgt_idx].transpose(1, 2)
        tgt_sig_score = tgt_sig_score[batch_idx, tgt_idx]

        ### stack features
        _src_feat = src_feat.unsqueeze(-1).repeat(1, 1, 1, num_preserved)
        _tgt_feat = tgt_feat.unsqueeze(-2).repeat(1, 1, num_preserved, 1)
        stack_feat = torch.cat([_src_feat, _tgt_feat], 1)

        ### initialize
        rotation = torch.eye(3, device=src.device).unsqueeze(0).expand(B, -1, -1)
        translation = torch.zeros(B, 3, device=src.device)
        self.loss = 0.0

        ### iterative registration
        if self.training:
            self.num_iter = 3
        else:
            self.num_iter = 3

        for it in range(self.num_iter):
            rotation_iter, translation_iter = self.register_iter(it, src_select, tgt_select, stack_feat, match_labels, src_sig_score, tgt_sig_score, src_gt)
            src_select = rotation_iter @ src_select + translation_iter.unsqueeze(-1)
            rotation = rotation_iter @ rotation
            translation = (rotation_iter @ translation.unsqueeze(-1)).squeeze() + translation_iter

        return rotation, translation
        

    def compute_loss(self):
        return self.loss
