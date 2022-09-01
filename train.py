import torch

from config import parser
from data import make_loader
from trainer import Trainer
from model import CoarseToFineNetwork


def check_and_resume(args, model):
    assert args.weights == '.' or args.coarse_weights == '.', 'Dual pretrained weights are restricted.'
    if args.weights == '.' and args.coarse_weights == '.':
        return model

    if args.coarse_weights != '.':
        checkpoint = torch.load(args.coarse_weights)
        model.feature_extractor.global_module.load_state_dict(checkpoint['global_feature_module'])
        model.coarse_register.load_state_dict(checkpoint['coarse_register'])

    elif args.weights != '.':
        checkpoint = torch.load(args.weights)
        model.load_state_dict(checkpoint['model'])

    return model


def main():
    args = parser.parse_args()

    train_loader = make_loader(args, 'train')
    test_loaders = list(map(lambda r: make_loader(args, 'test', 45*r), (1, 2, 3, 4)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CoarseToFineNetwork(args).to(device)
    model = check_and_resume(args, model)

    trainer = Trainer(args, model)

    trainer.train(args.epochs, args.val_freq, train_loader, test_loaders)


if __name__ == '__main__':
    main()
