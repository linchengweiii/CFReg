import torch
import torch.nn.functional as F
import numpy as np

def assert_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    return tensor

def registration_error(r_pred, t_pred, r_gt, t_gt):
    r_pred = assert_numpy(r_pred)
    t_pred = assert_numpy(t_pred)
    r_gt = assert_numpy(r_gt)
    t_gt = assert_numpy(t_gt)

    eps = 1e-16
    rotation_error = (np.trace(np.einsum('ijk, ijl->ikl', r_pred, r_gt), axis1=1, axis2=2) - 1) / 2
    rotation_error = np.arccos(np.clip(rotation_error, -1 + eps, 1 - eps)) * 180 / np.pi

    translation_error = np.linalg.norm(t_pred - t_gt, axis=1)

    return rotation_error, translation_error

### Losses
def compute_registration_loss(r_pred, t_pred, r_gt, t_gt, device=None):
    '''
        rotation: (B, 3, 3)
        translation: (B, 3)
    '''
    if device is None:
        device = r_gt.device

    identity = torch.eye(3, device=device).expand_as(r_gt)
    registration_loss = F.mse_loss(r_pred.transpose(1, 2) @ r_gt, identity) * 9 \
                      + F.mse_loss(t_pred, t_gt) * 3
    return registration_loss

def compute_occupancy_loss(occ_pred, occ_gt):
    '''
        occ: (B, N)
    '''
    bce_loss = F.binary_cross_entropy_with_logits(occ_pred, occ_gt, reduction='none')
    occupancy_loss = bce_loss.sum(-1).mean()
    return occupancy_loss
