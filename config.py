import argparse

arg_bool = lambda x: x.lower() in ['true', 't', '1']

parser = argparse.ArgumentParser(description='Point Cloud Registration')

misc = parser.add_argument_group('Misc')
misc.add_argument('--epochs', type=int, default=1000)
misc.add_argument('--val_freq', type=int, default=5)
misc.add_argument('--batch_size', type=int, default=32)
misc.add_argument('--weights', type=str, default='.')
misc.add_argument('--coarse_weights', type=str, default='.')
misc.add_argument('--output', type=str, default='checkpoint')

network = parser.add_argument_group('Network')
network.add_argument('--decode', type=arg_bool, default='True')
network.add_argument('--modules', type=str, nargs='+', default=['coarse', 'fine'])

data = parser.add_argument_group('Data')
data.add_argument('--dataset', type=str, default='ModelNet40')
data.add_argument('--num_points', type=int, default=1024)
data.add_argument('--gaussian_noise', type=arg_bool, default='False')
data.add_argument('--independently_sample', type=arg_bool, default='False')
data.add_argument('--subsample_ratio', type=float, default=1.0)
data.add_argument('--rotation_range', type=float, default=180.0)
data.add_argument('--translation_range', type=float, default=0.5)
data.add_argument('--data_folder', type=str, default='datasets/modelnet40_with_occupancy_label')

evaluate = parser.add_argument_group('Evaluate')
evaluate.add_argument('--rotation_error_thresh', type=float, default=5)
evaluate.add_argument('--translation_error_thresh', type=float, default=0.2)

demo = parser.add_argument_group('Demo')
demo.add_argument('--pcd', type=str, default='assets/lamp.ply')
