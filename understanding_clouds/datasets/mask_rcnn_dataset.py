import os
import pandas as pd
import numpy as np

import torch
import torchvision.transforms as T

from torch.utils.data import Dataset

from understanding_clouds.utils import preproces_dataframe_all_masks, get_all_masks_and_img
from understanding_clouds.constants import LABELS_MAPPING


class MaskRCNNDataset(Dataset):
    def __init__(self, images_dirpath, transforms=None, img_scale_factor=4):
        self.images_dirpath = images_dirpath
        self.df = preproces_dataframe_all_masks(pd.read_csv(
            os.path.join(images_dirpath, 'train_small.csv')))
        self.transforms = transforms
        self._img_scale_factor = img_scale_factor

    def __getitem__(self, index):

        masks, img, labels = get_all_masks_and_img(
            self.df, index, os.path.join(self.images_dirpath, 'train_images'), scale_factor=self._img_scale_factor)

        # img_id = self.df.iloc[index]['filename']
        labels = [LABELS_MAPPING[l] for l in labels]

        bboxes, masks_not_empty = [], []
        for mask in filter(lambda x: np.sum(x) > 0, masks):
            # it return tuple of indices where elements are not zero, it is almost an alias for np.asarray(condition).nonzero(), see np.nonzero documentation for more details
            pos = np.where(mask)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            bbox = [xmin, ymin, xmax, ymax]
            bboxes.append(bbox)
            masks_not_empty.append(mask)

        img_id = torch.tensor([index])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)
        area = (bboxes[:, 3] - bboxes[:, 1]) * \
            (bboxes[:, 2] - bboxes[:, 0])
        masks = torch.as_tensor(masks_not_empty, dtype=torch.uint8)
        iscrowd = torch.zeros((len(masks),), dtype=torch.int64)

        target = {'boxes': bboxes,
                  'masks': masks,
                  'labels': labels,
                  'image_id': img_id,
                  'area': area,
                  'iscrowd': iscrowd}

        img = T.ToTensor()(img)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.df)
