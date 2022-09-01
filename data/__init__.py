from torch.utils.data import DataLoader

from data.modelnet40 import ModelNet40

ALL_DATASETS = [ModelNet40]
DATASET_MAPPING = {d.__name__: d for d in ALL_DATASETS}

def make_loader(args, phase, rotation_range=None):
    shuffle = drop_last = phase == 'train'

    Dataset = DATASET_MAPPING[args.dataset]

    if rotation_range is None:
        rotation_range = args.rotation_range

    dataset = Dataset(
            phase=phase,
            num_points=args.num_points,
            subsample_ratio=args.subsample_ratio,
            gaussian_noise=args.gaussian_noise,
            independently_sample=args.independently_sample,
            rotation_range=rotation_range,
            translation_range=args.translation_range,
            data_folder=args.data_folder)

    loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=8)

    return loader
