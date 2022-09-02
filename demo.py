import numpy as np
import open3d as o3d
import time
import torch

from config import parser
from trainer import Trainer
from model import CoarseToFineNetwork
from util.transformation import se3_transform, random_transformation


def main():
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Point Cloud Data

    pcd = o3d.io.read_point_cloud(args.pcd)
    src_points = np.asarray(pcd.points)
    tgt_points = np.random.permutation(src_points)

    src = np.random.permutation(src_points[:args.num_points])
    tgt = np.random.permutation(tgt_points[:args.num_points])

    rotation, translation = random_transformation()
    src = (src - translation) @ rotation

    src_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(src))
    tgt_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(tgt))

    src_pcd.paint_uniform_color([1, 0, 0])
    tgt_pcd.paint_uniform_color([0, 0, 1])

    src = torch.from_numpy(src.T).unsqueeze(0).float().to(device)
    tgt = torch.from_numpy(tgt.T).unsqueeze(0).float().to(device)

    # Load Model and Inference
    model = CoarseToFineNetwork(args).to(device)
    model.load_state_dict(torch.load(args.weights)['model'])
    model.eval()

    r_pred, t_pred = model.register(src, tgt)

    # Visualization
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(src_pcd)
    vis.add_geometry(tgt_pcd)
    ctrl = vis.get_view_control()
    ctrl.rotate(0, -500)
    vis.poll_events()
    vis.update_renderer()
    time.sleep(2)

    vis.capture_screen_image('assets/input.png')
    
    src_pcd.rotate(model.r_coarse.squeeze(0).detach().cpu().numpy(), center=(0, 0, 0))
    src_pcd.translate(model.t_coarse.squeeze(0).detach().cpu().numpy())

    vis.update_geometry(src_pcd)

    vis.poll_events()
    vis.update_renderer()
    time.sleep(2)

    vis.capture_screen_image('assets/coarse_result.png')

    src_pcd.rotate(model.r_fine.squeeze(0).detach().cpu().numpy(), center=(0, 0, 0))
    src_pcd.translate(model.t_fine.squeeze(0).detach().cpu().numpy())

    vis.update_geometry(src_pcd)

    vis.poll_events()
    vis.update_renderer()
    time.sleep(2)

    vis.capture_screen_image('assets/final_result.png')

    vis.destroy_window()


if __name__ == '__main__':
    main()
