import torch
import torch.nn as nn

import model.vnn as vnn
from .vn_resnet_pointnet import VNResnetPointnet


class SE3EquivariantNetwork(nn.Module):
    GLOBAL_CHANNELS = [128, 128, 128, 128, 128, 128, 128]
    LOCAL_CHANNELS = [128, 64, 64, 64]
    def __init__(self, args):
        super().__init__()
        self.local = 'fine' in args.modules

        self.global_module = GlobalModule(self.GLOBAL_CHANNELS)

        if self.local:
            self.local_module = LocalModule(self.LOCAL_CHANNELS)


    def forward(self, pcd, k=20):
        latent_feat, global_feat = self.global_module(pcd, k)

        if self.local:
            local_feat = self.local_module(latent_feat)
        else:
            local_feat = None

        return global_feat, local_feat


class GlobalModule(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.fc_pre = vnn.Linear(3, 2 * channels[0])
        self.pool_pre = vnn.MeanPool()

        self.vn_network_0 = VNResnetPointnet(channels[0: 4])
        self.pool_latent = vnn.MeanPool()
        self.vn_network_1 = VNResnetPointnet(channels[3: -1])

        self.pool_global = vnn.MeanPool()
        self.actvn_global = vnn.LeakyReLU(channels[-2])
        self.fc_global = vnn.Linear(channels[-2], channels[-1])


    def preprocess(self, pcd, k):
        pcd = pcd.unsqueeze(1)
        net = vnn.util.get_graph_feature_cross(pcd, k=k)
        net = self.fc_pre(net)
        net = self.pool_pre(net)

        return net


    def forward(self, pcd, k):
        # Mean Subtraction
        center = pcd.mean(-1, keepdim=True)
        pcd = pcd - center

        # VN-based Network
        feat = self.preprocess(pcd, k)
        latent_feat = self.vn_network_0(feat)
        pooled = self.pool_latent(latent_feat, dim=3, keepdim=True).expand(latent_feat.size())
        feat = torch.cat([latent_feat, pooled], dim=1)
        global_feat = self.vn_network_1(feat)

        # Average Pooling
        global_feat = self.pool_global(global_feat)
        global_feat = self.fc_global(self.actvn_global(global_feat))

        # Latent Feature
        num_points = pcd.size(-1)
        expanded_feat = global_feat.unsqueeze(-1).expand(-1, -1, -1, num_points)
        latent_feat = torch.cat([latent_feat, expanded_feat], dim=1)

        # Mean Addition
        global_feat = global_feat + center.transpose(1, -1)

        return latent_feat, global_feat


class LocalModule(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.vn_network_2 = VNResnetPointnet(channels[0: -1])
        self.std_feature = vnn.StdFeature(channels[-2])
        self.conv_local = nn.Conv1d(3 * channels[-2], channels[-1], kernel_size=1, bias=False)
        self.bn_local = nn.BatchNorm1d(channels[-1])
        self.relu_local = nn.ReLU()
        

    def forward(self, latent_feat):
        # VN-based Network
        local_feat = self.vn_network_2(latent_feat)

        # Rotation Offset
        local_feat = self.std_feature(local_feat)[0]
        local_feat = torch.flatten(local_feat, start_dim=1, end_dim=2)
        local_feat = self.relu_local(self.bn_local(self.conv_local(local_feat)))
        
        return local_feat
