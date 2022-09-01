import os
import glob
import h5py
import numpy as np

from data.base_dataset import Dataset
from util.transformation import random_transformation
from util.pointcloud import jitter, farthest_points_sampling, get_distances_to_nearest_neighbor

class ModelNet40(Dataset):
    def __init__(self, phase, num_points=1024, gaussian_noise=False,
                 independently_sample=False, subsample_ratio=1.0,
                 rotation_range=np.pi, translation_range=0.5,
                 data_folder='.'):
        super().__init__(phase, num_points, gaussian_noise,
                         independently_sample, subsample_ratio,
                         rotation_range, translation_range)
        print(f'Loading {phase} dataset...')

        data, label, points, occupancies = [], [], [], []
        pointcloud_select = np.random.randint(100000, size=2048)
        points_select = np.random.randint(100000, size=2048)
        for filename in glob.glob(os.path.join(data_folder, f'{phase}*.h5')):
            with h5py.File(filename) as f:
                data.append(f['data'][:][:, pointcloud_select].astype('float32'))
                label.append(f['label'][:].astype('int64'))
                points.append(f['points'][:][:, points_select].astype('float32'))
                occupancies.append(f['occupancies'][:][:, points_select].astype('float32'))

        self.src = np.concatenate(data, axis=0)
        self.tgt = np.concatenate(data, axis=0)

        self.label = np.concatenate(label, axis=0)
        self.points = np.concatenate(points, axis=0)
        self.occupancies = np.concatenate(occupancies, axis=0)


    def __getitem__(self, index):
        pointcloud1 = self.src[index]
        pointcloud2 = self.tgt[index]

        if self.independently_sample:
            np.random.shuffle(pointcloud2)

        pointcloud1 = np.random.permutation(pointcloud1[:self.num_points])
        pointcloud2 = np.random.permutation(pointcloud2[:self.num_points])
        
        if self.phase == 'test' and self.num_subsample_points < self.num_points:
            pointcloud1 = farthest_points_sampling(pointcloud1, self.num_subsample_points)
            pointcloud2 = farthest_points_sampling(pointcloud2, self.num_subsample_points)

        rotation, translation = random_transformation(self.rotation_range, self.translation_range)
        pointcloud2 = pointcloud2 @ rotation.T + translation

        if self.phase == 'train' and self.num_subsample_points < self.num_points:
            pointcloud1 = farthest_points_sampling(pointcloud1, self.num_subsample_points)
            pointcloud2 = farthest_points_sampling(pointcloud2, self.num_subsample_points)
        
        if self.gaussian_noise:
            pointcloud1 = jitter(pointcloud1)
            pointcloud2 = jitter(pointcloud2)

        ### Sample points and occupancies
        points1 = self.points[index]
        points2 = self.points[index] @ rotation.T

        occupancy = self.occupancies[index]

        return {
            'src_sample_points': points1.astype('float32'),
            'tgt_sample_points': points2.astype('float32'),
            'src_occupancy': occupancy.astype('float32'),
            'tgt_occupancy': occupancy.astype('float32'),
            'src': pointcloud1.T.astype('float32'),
            'tgt': pointcloud2.T.astype('float32'),
            'rotation': rotation.astype('float32'),
            'translation': translation.astype('float32')
        }
