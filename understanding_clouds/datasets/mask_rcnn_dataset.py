import os
import pandas as pd
import numpy as np

from PIL import Image

import torch
import torchvision.transforms as T

from torch.utils.data import Dataset


from understanding_clouds.utils import preproces_dataframe_all_masks, get_all_masks_and_img


class MaskRCNNDataset(Dataset):
    def __init__(self, images_dirpath, transforms=None):
        self.images_dirpath = images_dirpath
        self.df = preproces_dataframe_all_masks(pd.read_csv(
            os.path.join(images_dirpath, 'train_small.csv')))
        self.transforms = transforms

    def __getitem__(self, index):

        masks, img, labels = get_all_masks_and_img(
            self.df, index, os.path.join(self.images_dirpath, 'train_images'))

        # it return tuple of indices where elements are not zero, it is almost an alias for np.asarray(condition).nonzero(), see np.nonzero documentation for more details
        bboxes, masks_not_empty = [], []
        for mask in filter(lambda x: np.sum(x) > 0, masks):
            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            bbox = [xmin, ymin, xmax, ymax]
            bboxes.append(bbox)
            masks_not_empty.append(mask)
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        masks = torch.as_tensor(masks_not_empty, dtype=torch.uint8)
        target = {'boxes': bboxes, 'masks': masks}
        img = T.ToTensor()(img)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.df)
