import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski
from scipy.spatial import cKDTree

def jitter(pointcloud, sigma=0.01, clip=0.05):
    gaussian_noise = np.clip(sigma * np.random.randn(*pointcloud.shape), -1 * clip, clip)
    return pointcloud + gaussian_noise


def farthest_points_sampling(pointcloud, num_subsample_points):
    nbrs = NearestNeighbors(n_neighbors=num_subsample_points, algorithm='auto',
                            metric=lambda x, y: minkowski(x, y)).fit(pointcloud)
    random_p = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    indices = nbrs.kneighbors(random_p, return_distance=False).reshape((num_subsample_points,))
    return pointcloud[indices, :]


def get_distances_to_nearest_neighbor(pointcloud1, pointcloud2):
    tree = cKDTree(pointcloud2)
    distances, _ = tree.query(pointcloud1, k=1)
    return distances
