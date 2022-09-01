import torch
import torch.nn as nn
import torch.nn.functional as F

import model.vnn as vnn
from .layers import CResnetBlockConv1d, CBatchNorm1d

class VNOccupancyNet(nn.Module):
    ''' Decoder with conditional batch normalization (CBN) class.
    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        leaky (bool): whether to use leaky ReLUs
        legacy (bool): whether to use the legacy structure
    '''

    def __init__(self, dim=3, c_dim=128, hidden_size=256, leaky=False):
        super().__init__()

        # self.fc_p = nn.Conv1d(dim, hidden_size, 1)
        self.fc_p = nn.Conv1d(c_dim//3+1, hidden_size, 1)
        self.block0 = CResnetBlockConv1d(c_dim, hidden_size)
        self.block1 = CResnetBlockConv1d(c_dim, hidden_size)
        self.block2 = CResnetBlockConv1d(c_dim, hidden_size)
        self.block3 = CResnetBlockConv1d(c_dim, hidden_size)
        self.block4 = CResnetBlockConv1d(c_dim, hidden_size)

        self.bn = CBatchNorm1d(c_dim, hidden_size)

        self.fc_out = nn.Conv1d(hidden_size, 1, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.std_feature = vnn.StdFeature(c_dim//3, dim=4, normalize_frame=False, negative_slope=0.0)

    def forward(self, p, c, **kwargs):
        p = p.transpose(1, 2)
        batch_size, D, T = p.size()  # B*3*F

        ### preprocess p to be translation invariant
        # p = p - p.mean(2).unsqueeze(2)

        ### preprocess p to be rotation invariant
        p_norm = torch.linalg.norm(p, dim=1, keepdim=True)  # B*1*T
        ### c: B*F*3
        c = c.transpose(1, 2)  # B*3*F

        ### preprocess c to be translation invarient
        # c = c - c.mean(2).unsqueeze(2)

        c = c.unsqueeze(2)  # B*3*1*F
        p = p.unsqueeze(3)  # B*3*T*1

        inner_p = (c*p).sum(1)  # B*T*F
        inner_p = inner_p.transpose(1, 2)  # B*F*T
        p = torch.cat([inner_p, p_norm], dim=1)  # B*(F+1)*T

        ### preprocess c to be rotation invariant
        c = c.permute(0, 3, 1, 2)  # B*F*3*1
        c, c_transmat = self.std_feature(c)  # B*F*3*1
        c = c.flatten(1)  # B*F

        ### start the network
        net = self.fc_p(p)

        net = self.block0(net, c)
        net = self.block1(net, c)
        net = self.block2(net, c)
        net = self.block3(net, c)
        net = self.block4(net, c)

        out = self.fc_out(self.actvn(self.bn(net, c)))
        out = out.squeeze(1)

        return out
