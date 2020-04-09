import os
import argparse
import json

from typing import Mapping

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from loggify import Loggify

from understanding_clouds.datasets.mask_rcnn_dataset import MaskRCNNDataset
from understanding_clouds.torchvision_references.engine import train_one_epoch, evaluate
from understanding_clouds.torchvision_references.utils import collate_fn


class CloudsMaskRCNN:
    def __init__(self, experiment_dirpath: str, init_lr: float = 0.001, weight_decay: float = 0.005, gamma: float = 0.9):
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

        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=self.gamma)

        os.makedirs(self.experiment_dirpath, exist_ok=True)

    def train(self, dataloaders: Mapping[str, torch.utils.data.DataLoader], epochs: int, snapshot_frequency: int = 10):
        for i, epoch in enumerate(range(1, epochs + 1)):
            for phase, dataloader in dataloaders.items():
                if phase == 'TRAIN':
                    train_one_epoch(self.net, self.optimizer, dataloader,
                                    self.device, epoch, print_freq=len(dataloader) // 5)
                    self.lr_scheduler.step()
                else:
                    evaluate(self.net, dataloader, device=self.device)
            if i % snapshot_frequency == 0:
                self.save_model(epoch)
        self.save_model(epoch)

    def predict(self, dataloader):
        pass

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


def get_mask_rcnn_net(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(
        pretrained=True)

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
    args = parser.parse_args()
    return args



def main_without_args(args):
    ds_train = MaskRCNNDataset(images_dirpath=args.data_path)
    dataloaders = {'TRAIN': torch.utils.data.DataLoader(ds_train, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn)}
    clouds_model = CloudsMaskRCNN(experiment_dirpath=args.experiment_dirpath, init_lr=args.init_lr, weight_decay=args.weight_decay, gamma=args.gamma)
    if args.pretrained_model_path is not None:
        args.pretrained_model_path = os.path.abspath(args.pretrained_model_path)
        clouds_model.load_model(args.pretrained_model_path)

    clouds_model.train(dataloaders=dataloaders, epochs=args.epochs)
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