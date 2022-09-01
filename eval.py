import torch

from config import parser
from data import make_loader
from trainer import Trainer
from model import CoarseToFineNetwork


def main():
    args = parser.parse_args()

    test_loaders = list(map(lambda r: make_loader(args, 'test', 45*r), (1, 2, 3, 4)))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CoarseToFineNetwork(args).to(device)
    model.load_state_dict(torch.load(args.weights)['model'])

    trainer = Trainer(args, model)

    trainer.eval(test_loaders)


if __name__ == '__main__':
    main()
