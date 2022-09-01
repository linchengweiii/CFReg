import os
import numpy as np
import torch.utils.data

from util.transformation import random_transformation
from util.pointcloud import jitter, farthest_points_sampling

class Dataset(torch.utils.data.Dataset):
    def __init__(self, phase, num_points=1024, gaussian_noise=False, 
                 independently_sample=False, subsample_ratio=1.0,
                 rotation_range=np.pi, translation_range=0.5):
        self.phase = phase
        self.num_points = num_points
        self.gaussian_noise = gaussian_noise
        self.independently_sample = independently_sample
        self.num_subsample_points = int(subsample_ratio * num_points)
        self.rotation_range = rotation_range / 180 * np.pi
        self.translation_range = translation_range


    def __len__(self):
        assert self.src.shape[0] == self.tgt.shape[0], \
               f'source length {self.src.shape[0]} is different from target length {self.tgt.shape[0]}'
        return self.src.shape[0]
