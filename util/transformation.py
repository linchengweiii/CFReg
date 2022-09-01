import torch
import numpy as np

def se3_transform(target, rotation, translation=None):
    '''
    When translation is None, the second parameter
    should be transformation instead of rotation.

    Parameter:
        target (tensor): (3, N), (B, 3, N) or (N, B, 3, C)
        rotation (tensor): (3, 3) or (B, 3, 3)
        transformation (tensor): (3, 4) or (B, 3, 4)
        translation (tensor): (3, ) or (B, 3)
    '''
    if translation is None:
        transformation = rotation
        rotation = transformation[..., :3, :3]
        translation = transformation[..., :3, 3]

    if len(target.shape) == len(translation.shape) + 1:
        translation = translation[..., None]
    elif len(target.shape) == len(translation.shape) + 2:
        translation = translation[None, ..., None]
    elif len(target.shape) == len(translation.shape) + 3:
        translation = translation[None, None, ..., None]
    else:
        raise NotImplementedError
    
    return rotation @ target + translation

def random_transformation(angle=np.pi, max_translation=0.5):
    return random_rotation(angle), random_translation(max_translation)

def random_translation(max_translation=0.5):
    return np.random.uniform(-1 * max_translation, max_translation, size=3)

def rotation_by_angle(anglex=np.pi, angley=np.pi, anglez=np.pi):
    cosx = np.cos(anglex)
    cosy = np.cos(angley)
    cosz = np.cos(anglez)
    sinx = np.sin(anglex)
    siny = np.sin(angley)
    sinz = np.sin(anglez)
    Rx = np.array([[1, 0, 0],
                    [0, cosx, -sinx],
                    [0, sinx, cosx]])
    Ry = np.array([[cosy, 0, siny],
                    [0, 1, 0],
                    [-siny, 0, cosy]])
    Rz = np.array([[cosz, -sinz, 0],
                    [sinz, cosz, 0],
                    [0, 0, 1]])
    R = Rx.dot(Ry).dot(Rz)

    return R
    

def random_rotation(angle=np.pi):
    anglex = np.random.uniform() * angle
    angley = np.random.uniform() * angle
    anglez = np.random.uniform() * angle

    return rotation_by_angle(anglex, angley, anglez)
