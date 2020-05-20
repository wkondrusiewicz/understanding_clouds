import os
import argparse
import json
import time

from subprocess import Popen, PIPE
from typing import Mapping

import numpy as np
import torch
import torchvision

from torch.utils.data import DataLoader

from loggify import Loggify

from understanding_clouds.datasets.unet_dataset import UnetDataset
from understanding_clouds.models.unet.dice_loss import DiceLoss, BCEDiceLoss


class CloudsUnet:
    def __init__(self, experiment_dirpath: str, init_lr: float = 0.001, weight_decay: float = 0.005, gamma: float = 0.9):
        self.experiment_dirpath = experiment_dirpath
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.net = get_mask_unet_net(4, self.device)
        params = [p for p in self.net.parameters()
                  if p.requires_grad == True]
        # self.loss_fn = torch.nn.MSELoss() # wrong, but left for now
        self.loss_fn = BCEDiceLoss(eps=1e-7, activation=None)
        self.optimizer = torch.optim.Adam(
            params, lr=self.init_lr, weight_decay=self.weight_decay)

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=self.gamma)

        os.makedirs(self.experiment_dirpath, exist_ok=True)

    def train(self, train_dataloader: torch.utils.data.DataLoader,
              valid_dataloader: torch.utils.data.DataLoader,
              epochs: int, snapshot_frequency: int = 10, print_freq=10):

        optimizer = self.optimizer
        loss_fn = self.loss_fn
        model = self.net
        losses = {'TRAIN': {}, 'VALID': {}}

        print('Beginning training...')
        print('Using cuda...' if torch.cuda.is_available()
              else 'Using cpu...')

        for i, epoch in enumerate(range(1, epochs + 1)):
            print(f'\nEpoch {epoch}')
            model.train()
            t1 = time.time()
            loss_tmp = []

            for j, (image, masks) in enumerate(train_dataloader):
                t11 = time.time()
                optimizer.zero_grad()
                output = model(image)
                loss = loss_fn(output, masks)
                loss_tmp.append(loss.item())

                loss.backward()
                optimizer.step()
                t12 = time.time()
                if (j +1 ) % print_freq == 0:
                    print(
                    f'\tTRAIN: [{j+1}/{len(train_dataloader)}], mean time per print freq {print_freq*np.round(t12-t11, 2)} seconds, loss = {loss.item()}\n')

            mean = np.mean(loss_tmp)
            print(f'\tTRAIN ended with total losss {mean}\n\n\n')
            losses['TRAIN'][epoch] = {'batch_losses': dict(zip(range(len(loss_tmp)), loss_tmp)), 'total_loss': mean}
            if i % snapshot_frequency == 0:
                self.save_model(epoch)

            #validation phase
            model.eval()
            del image, masks
            loss_tmp.clear()

            with torch.no_grad():
                for j, (image, masks) in enumerate(valid_dataloader):
                    t11 = time.time()
                    output = model(image)
                    loss = loss_fn(output, masks)
                    loss_tmp.append(loss.item())
                    t12 = time.time()
                    if (j +1 ) % print_freq == 0:
                        print(
                        f'\tVALID: [{j+1}/{len(valid_dataloader)}], mean time per print freq {print_freq*np.round(t12-t11, 2)} seconds, loss = {loss.item()}\n')

            mean = np.mean(loss_tmp)
            print(f'\tVALID ended with total losss {mean}\n\n\n')
            losses['VALID'][epoch] = {'batch_losses': dict(zip(range(len(loss_tmp)), loss_tmp)), 'total_loss': mean}

            p = Popen(['nvidia-smi'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
            output, _ = p.communicate(b"input data that is passed to subprocess' stdin")

            gpu_usage = [x for x in str(output).split('|') if 'MiB' in x][0]

            t2 = time.time()
            print(f'\tEpoch {epoch} took {np.round(t2-t1, 2)} seconds\n\tGPU utilization during this epoch was {gpu_usage}\n')

        print('Done!')
        print('Saving model...')
        self.save_model(epoch)
        with open(os.path.join(self.experiment_dirpath, 'losses.json'), 'w') as f:
            json.dump(losses, f)
        print('Done!')

    def save_model(self, epoch):
        os.makedirs(self.experiment_dirpath, exist_ok=True)
        checkpoint = {'epoch': epoch,
                      'state_dict': self.net.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'lr_scheduler': self.lr_scheduler.state_dict()}
        torch.save(checkpoint, os.path.join(
            self.experiment_dirpath, 'model.pth'))

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print('Model loaded successfully!')


def get_mask_unet_net(num_classes, device):
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=1, init_features=32, pretrained=True).to(device)
    model.conv = torch.nn.Conv2d(
        32, num_classes, kernel_size=(1, 1), stride=(1, 1)).to(device)
    return model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epochs', help="Number of epochs",
                        type=int, default=10)
    parser.add_argument(
        '--init_lr', help='Initial learning_rate', type=float, default=0.001)
    parser.add_argument('-trb', '--train_batch_size',
                        help="Train batch size", type=int, default=1)
    parser.add_argument('-vab', '--valid_batch_size',
                        help="Valid batch size", type=int, default=1)
    parser.add_argument('--experiment_dirpath',
                        help='Where to save the model', required=True, type=str)
    parser.add_argument('--pretrained_model_path', help='Path to pretrained model',
                        type=str, default=None)
    parser.add_argument(
        '--data_path', help='Path to data', required=True, type=str)
    parser.add_argument('--gamma',
                        default=0.9, type=float)
    parser.add_argument('--weight_decay',
                        default=0.005, type=float)
    parser.add_argument('--subsample', default=100, type=int)
    parser.add_argument('-tts', '--train_test_split',
                        default=0.05, type=float)
    parser.add_argument('--print_freq', default=10, type=int)
    args = parser.parse_args()
    return args


def main_without_args(args):
    if args.train_test_split:
        from glob import glob
        from sklearn.model_selection import train_test_split
        df_len = len(
            glob(os.path.join(args.data_path, 'train_images/*')))
        train_ids, valid_ids = train_test_split(
            range(df_len), test_size=args.train_test_split, random_state=42)
    print('Loading datasets...')
    train_dataset = UnetDataset(
        images_dirpath=args.data_path, subsample=args.subsample, split_ids=train_ids)
    valid_dataset = UnetDataset(
        images_dirpath=args.data_path, subsample=args.subsample, split_ids=valid_ids)
    print('Done!')
    print('Preparing dataloaders...')
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.valid_batch_size, shuffle=False)
    print('Done!')
    print('Declaring model...')
    clouds_model = CloudsUnet(experiment_dirpath=args.experiment_dirpath,
                              init_lr=args.init_lr, weight_decay=args.weight_decay, gamma=args.gamma)
    print('Done!')
    if args.pretrained_model_path is not None:
        args.pretrained_model_path = os.path.abspath(
            args.pretrained_model_path)
        clouds_model.load_model(args.pretrained_model_path)

    print('Initiating training...')
    clouds_model.train(train_dataloader, valid_dataloader, args.epochs, print_freq=args.print_freq)
    training_params = {'epochs': args.epochs,
                       'init_lr': args.init_lr,
                       'train_batch_size': args.train_batch_size,
                       'data_path': os.path.abspath(args.data_path),
                       'weight_decay': args.weight_decay,
                       'gamma': args.gamma,
                       'pretrained_model_path': args.pretrained_model_path or None}

    with open(os.path.join(args.experiment_dirpath, 'training_params.json'), 'w') as f:
        json.dump(training_params, f)


def main():
    args = parse_args()
    os.makedirs(args.experiment_dirpath, exist_ok=True)
    with Loggify(os.path.join(args.experiment_dirpath, 'log.txt')):
        main_without_args(args)


if __name__ == '__main__':
    main()
