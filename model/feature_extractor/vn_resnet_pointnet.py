import torch
import torch.nn as nn

import model.vnn as vnn


class VNResnetPointnet(nn.Module):
    ''' PointNet-based network with ResNet blocks.
    Args:
        channels (int): channels of each layer
    '''

    def __init__(self, channels):
        super().__init__()

        self.blocks = nn.ModuleList()
        for in_dim, out_dim in zip(channels[:-1], channels[1:]):
            self.blocks.append(VNResnetBlock(2*in_dim, out_dim))

        self.pool = vnn.MeanPool()


    def forward(self, net):
        for block in self.blocks[:-1]:
            net = block(net)
            pooled = self.pool(net, dim=3, keepdim=True).expand(net.size())
            net = torch.cat([net, pooled], dim=1)

        net = self.blocks[-1](net)

        return net


class VNResnetBlock(nn.Module):
    ''' Fully connected Vector Neuron ResNet block.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None, share_nonlinearity=False, negative_slope=0.0):
        super().__init__()
        # Attributes
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        # Submodules
        self.actvn0 = vnn.LeakyReLU(self.size_in, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        self.actvn1 = vnn.LeakyReLU(self.size_h, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        self.fc0 = vnn.Linear(self.size_in, self.size_h)
        self.fc1 = vnn.Linear(self.size_h, self.size_out)

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = vnn.Linear(self.size_in, self.size_out)

        # Initialization
        nn.init.zeros_(self.fc1.map_to_feat.weight)


    def forward(self, x):
        net = self.actvn0(x)
        net = self.fc0(net)
        net = self.actvn1(net)
        dx = self.fc1(net)

        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x

        return x_s + dx
