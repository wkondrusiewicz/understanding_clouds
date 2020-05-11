import os
import argparse
import json
import time
from subprocess import Popen, PIPE

from typing import Mapping
from copy import deepcopy

import numpy as np
import torch
import torchvision

from torch.utils.data import DataLoader

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from loggify import Loggify

from understanding_clouds.datasets.mask_rcnn_dataset import MaskRCNNDataset
from understanding_clouds.models.mask_rcnn.prediction import MaskRCNNPrediction
from understanding_clouds.utils import collate_fn


class CloudsMaskRCNN:
    def __init__(self, experiment_dirpath: str, init_lr: float = 0.001, weight_decay: float = 0.005, gamma: float = 0.96):
        self.experiment_dirpath = experiment_dirpath
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.net = get_mask_rcnn_net(5).to(self.device)
        params = [p for p in self.net.parameters()
                  if p.requires_grad == True]
        self.optimizer = torch.optim.Adam(
            params, lr=self.init_lr, weight_decay=self.weight_decay)


        os.makedirs(self.experiment_dirpath, exist_ok=True)

    def _single_forward_pass(self, images, targets, phase):
        images = [img.cuda() for img in images]
        targets = [{k: v.cuda() for k, v in target.items()}
                   for target in targets]
        # it has a structure of {'loss_type': torch.tensor corresponding to the loss}
        loss_dict = self.net(images, targets)
        return loss_dict

    def train(self, dataloaders: Mapping[str, torch.utils.data.DataLoader],
              epochs: int, snapshot_frequency: int = 10, print_freq=None):
        losses_to_save = {}
        self.net.train()

        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer=self.optimizer, gamma=self.gamma)


        # self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer=self.optimizer, milestones=[int(0.6 * epochs), int(0.9 * epochs)])

        for i, epoch in enumerate(range(1, epochs + 1)):
            t1 = time.time()
            epoch_losses = {}
            print(f'Epoch {epoch}')

            for phase, dataloader in dataloaders.items():
                phase_loss = 0

                if print_freq is None:
                    print_freq = len(dataloader) // 3

                per_batch_losses = {}

                for j, (images, targets) in enumerate(dataloader):
                    t11 = time.time()
                    self.optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'TRAIN'):
                        loss_dict = self._single_forward_pass(
                            images, targets, phase)


                        # loss_dict['loss_mask']=2 * loss_dict['loss_mask']


                        loss = sum([l for l in loss_dict.values()])
                        phase_loss += loss.item()

                        if phase == 'TRAIN':
                            loss.backward()
                            self.optimizer.step()

                    loss_dict_printable = {''.join(k.split('loss_')): np.round(
                        v.item(), 5) for k, v in loss_dict.items()}
                    loss_dict_printable['total_loss']=np.round(loss.item(),5)
                    per_batch_losses[str(j)]=loss_dict_printable
                    t12 = time.time()
                    if (j +1 ) % print_freq == 0:
                        print(
                            f'\t{phase}, [{j+1}/{len(dataloader)}], mean time per print freq {print_freq*np.round(t12-t11, 2)} seconds, total loss = {loss.item()}, particular losses:\n\t{loss_dict_printable}\n')

                phase_loss /= len(dataloader)
                epoch_losses[phase] = {'total_phase_loss': phase_loss, 'per_batch_losses': per_batch_losses}
                log_string = f'\t{phase} ended with total losss {phase_loss}\n\n\n '
                print(log_string)

            losses_to_save[str(epoch)] = epoch_losses

            # self.lr_scheduler.step()

            if i % snapshot_frequency == 0:
                self.save_model(epoch)

            t2 = time.time()

            p = Popen(['nvidia-smi'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
            output, _ = p.communicate(b"input data that is passed to subprocess' stdin")

            gpu_usage = [x for x in str(output).split('|') if 'MiB' in x][0]

            print(
            f'\tEpoch {epoch} took {np.round(t2-t1, 2)} seconds\n\tGPU utilization during this epoch was {gpu_usage}\n')

        with open(os.path.join(self.experiment_dirpath, 'losses.json'), 'w') as f:
            json.dump(losses_to_save, f)
        self.save_model(epoch)

    def predict(self, dataloader: DataLoader, on_test: bool = False):
        self.net.cpu()
        self.net.eval()
        predictions = []
        for data in dataloader:
            images, targets = (data, None) if on_test else data
            predictions.append(self._single_prediction(images, targets))
        return predictions

    def _single_prediction(self, images, targets=None):
        images = [img.cpu() for img in images]
        if targets is not None:
            targets = [{k: v.cpu() for k, v in target.items()}
                       for target in targets]
        outputs = self.net(deepcopy(images))
        return MaskRCNNPrediction(raw_images=images, raw_outputs=outputs, raw_targets=targets)

    def save_model(self, epoch):
        os.makedirs(self.experiment_dirpath, exist_ok=True)
        checkpoint = {'epoch': epoch,
                      'state_dict': self.net.state_dict(),
                      # 'lr_scheduler': self.lr_scheduler.state_dict(),
                      'optimizer': self.optimizer.state_dict()}
        torch.save(checkpoint, os.path.join(
            self.experiment_dirpath, 'model.pth'))

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.net.load_state_dict(checkpoint['state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print('Model loaded successfully!')


def get_mask_rcnn_net(num_classes):
    kw = {'min_size': 350, 'max_size': 525}
    # kw = {'min_size': 200, 'max_size': 300}
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True, **kw)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Use regression for OMTF')

    parser.add_argument('-e', '--epochs', help="Number of epochs",
                        type=int, default=10)
    parser.add_argument(
        '--init_lr', help='Initial learning_rate', type=float, default=0.001)
    parser.add_argument('-trb', '--train_batch_size',
                        help="Trian batch size", type=int, default=1)
    parser.add_argument('-vab', '--valid_batch_size',
                        help="Trian batch size", type=int, default=1)
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
    parser.add_argument('-tts', '--train_test_split', default=0.05, type=float)
    parser.add_argument('--print_freq', default=None, type=int)
    parser.add_argument('--snap_freq', default=5, type=int)
    args = parser.parse_args()
    return args


def main_without_args(args):
    if args.train_test_split:
        from glob import glob
        from sklearn.model_selection import train_test_split
        df_len = len(glob(os.path.join(args.data_path,'train_images/*')))
        train_ids, valid_ids = train_test_split(range(df_len), test_size=args.train_test_split, random_state=42)
    ds_train = MaskRCNNDataset(
        images_dirpath=args.data_path, subsample=args.subsample, split_ids=train_ids)
    ds_valid = MaskRCNNDataset(
        images_dirpath=args.data_path, subsample=args.subsample, split_ids=valid_ids)
    dataloaders = {'TRAIN': DataLoader(ds_train, batch_size=args.train_batch_size,
                                       shuffle=True, collate_fn=collate_fn),
                   'VALID': DataLoader(ds_valid, batch_size=args.valid_batch_size, shuffle=False,
                                       collate_fn=collate_fn)}
    clouds_model = CloudsMaskRCNN(experiment_dirpath=args.experiment_dirpath,
                                  init_lr=args.init_lr, weight_decay=args.weight_decay, gamma=args.gamma)
    if args.pretrained_model_path is not None:
        args.pretrained_model_path = os.path.abspath(
            args.pretrained_model_path)
        clouds_model.load_model(args.pretrained_model_path)

    clouds_model.train(dataloaders=dataloaders, epochs=args.epochs, print_freq=args.print_freq, snapshot_frequency=args.snap_freq)
    training_params = {'epochs': args.epochs,
                       'init_lr': args.init_lr,
                       'train_batch_size': args.train_batch_size,
                       'valid_batch_size': args.valid_batch_size,
                       'data_path': os.path.abspath(args.data_path),
                       'weight_decay': args.weight_decay,
                       'gamma': args.gamma,
                       'pretrained_model_path': args.pretrained_model_path or None,
                       'subsample': args.subsample or None,
                       'train_test_split': args.train_test_split or None}

    with open(os.path.join(args.experiment_dirpath, 'training_params.json'), 'w') as f:
        json.dump(training_params, f)


def main():
    args = parse_args()
    os.makedirs(args.experiment_dirpath, exist_ok=True)
    with Loggify(os.path.join(args.experiment_dirpath, 'log.txt')):
        main_without_args(args)


if __name__ == '__main__':
    main()
