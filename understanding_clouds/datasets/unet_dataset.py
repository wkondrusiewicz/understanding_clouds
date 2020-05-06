import os
import pandas as pd
import numpy as np
import cv2

import torch
import torchvision.transforms as T

from torch.utils.data import Dataset

from understanding_clouds.utils import preproces_dataframe_all_masks, get_all_masks_and_img
from understanding_clouds.constants import LABELS_MAPPING

class UnetDataset(Dataset):
    def __init__(self, images_dirpath, transforms=None, img_scale_factor=4, subsample=None):
        self.images_dirpath = images_dirpath
        self._img_scale_factor = 4
        df = pd.read_csv(os.path.join(images_dirpath, 'train.csv'))
        df = preproces_dataframe_all_masks(df)
        self.df = df.iloc[::subsample] if subsample is not None else df
        self.transforms = transforms

    def __getitem__ (self, index):
        masks, img, labels = get_all_masks_and_img(
            self.df, index, os.path.join(self.images_dirpath, 'train_images'), scale_factor=self._img_scale_factor)

        # size reduced, divisible by 16, left hardcoded for now
        img = cv2.resize(img, (352,528), interpolation=cv2.INTER_AREA)
        masks = [cv2.resize(mask, (352,528), interpolation=cv2.INTER_AREA) for mask in masks]

        labels = [LABELS_MAPPING[l] for l in labels]
        labels = [l for l in labels if l > 0]

        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8) / 255

        img = T.ToTensor()(img)
        if self.transforms is not None:
            img, masks = self.transforms(img, masks)

        img = img.to('cuda')
        masks = masks.to('cuda').float()

        return img, masks

    def __len__(self):
        return len(self.df)
