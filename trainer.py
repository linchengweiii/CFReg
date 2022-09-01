import os
import numpy as np
import time
import torch
from tqdm import tqdm

from util.metric import registration_error

class Trainer:
    def __init__(self, args, model):
        self.model = model
        self.batch_size = args.batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.coarse = 'coarse' in args.modules
        self.fine = 'fine' in args.modules

        if self.fine:
            params = list(self.model.feature_extractor.local_module.parameters()) + \
                     list(self.model.fine_register.parameters())
        elif self.coarse:
            params = list(self.model.feature_extractor.global_module.parameters()) + \
                     list(self.model.coarse_register.parameters())

        self.optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-3)

        self.rotation_error_thresh = args.rotation_error_thresh
        self.translation_error_thresh = args.translation_error_thresh

        self.output = args.output


    def train(self, epochs, val_freq, train_loader, test_loaders):
        for epoch in range(0, epochs):
            print(f'==== Epoch {epoch + 1} =====')

            train_error = self.train_one_epoch(train_loader)
            self.print_metric('TRAIN', *train_error)

            if (epoch + 1) % val_freq == 0:
                self.eval(test_loaders)

            self.save_checkpoint(epoch)


    def eval(self, test_loaders):
        for i, test_loader in enumerate(test_loaders):
            test_error = self.test_one_epoch(test_loader)
            self.print_metric(f'TEST-{str(45*(i+1))}', *test_error)
        

    def train_one_epoch(self, loader):
        self.model.train()
        return self.run_one_epoch('train', loader)


    @torch.no_grad()
    def test_one_epoch(self, loader):
        self.model.eval()
        return self.run_one_epoch('test', loader)


    def run_one_epoch(self, phase, loader):
        r_errors, t_errors = [], []
        reg_times = []
        for batch_dict in tqdm(loader, ncols=150):
            src_p = batch_dict['src_sample_points'].to(self.device)
            tgt_p = batch_dict['tgt_sample_points'].to(self.device)
            src_occ_gt = batch_dict['src_occupancy'].to(self.device)
            tgt_occ_gt = batch_dict['tgt_occupancy'].to(self.device)
            src = batch_dict['src'].to(self.device)
            tgt = batch_dict['tgt'].to(self.device)
            r_gt = batch_dict['rotation'].to(self.device)
            t_gt = batch_dict['translation'].to(self.device)

            start_time = time.time()

            if phase == 'train':
                self.optimizer.zero_grad()

            if phase == 'train':
                r_pred, t_pred = self.model(src, tgt, src_p, tgt_p, r_gt, t_gt)
            else:
                r_pred, t_pred = self.model.register(src, tgt)

            if phase == 'train':
                self.backpropagate(src, tgt, r_gt, t_gt, src_occ_gt, tgt_occ_gt)

            end_time = time.time()

            r_error, t_error = registration_error(r_pred, t_pred, r_gt, t_gt)
            r_errors.append(r_error)
            t_errors.append(t_error)
            reg_times.append((end_time - start_time) / self.batch_size)

        r_errors = np.concatenate(r_errors, axis=0)
        t_errors = np.concatenate(t_errors, axis=0)
        reg_time = np.mean(reg_times)

        return r_errors, t_errors, reg_time


    def backpropagate(self, src, tgt, r_gt, t_gt, src_occ_gt, tgt_occ_gt):
        loss = self.model.compute_loss(src, tgt, r_gt, t_gt,
                                       src_occ_gt, tgt_occ_gt)
        loss.backward()
        self.optimizer.step()

    def print_metric(self, phase, r_errors, t_errors, reg_time):
        r_recall = r_errors < self.rotation_error_thresh
        t_recall = t_errors < self.translation_error_thresh
        recall_rate = np.mean(r_recall & t_recall)
        print(f'{phase:<15}',
              f'Recall Rate: {recall_rate:<15.3f}',
              f'Rotation Error: {np.mean(r_errors):<15.4f}',
              f'Translation Error: {np.mean(t_errors):<15.4f}',
              f'Time: {reg_time * 1000: .2f} ms',
              sep='')

    def save_checkpoint(self, epoch):
        checkpoint = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'global_feature_module': self.model.feature_extractor.global_module.state_dict(),
            'local_feature_module': self.model.feature_extractor.local_module.state_dict() if self.fine else None,
            'coarse_register': self.model.coarse_register.state_dict() if self.coarse else None,
            'fine_register': self.model.fine_register.state_dict() if self.fine else None,
        }

        torch.save(checkpoint, os.path.join('checkpoints', f'{self.output}.pth'))
