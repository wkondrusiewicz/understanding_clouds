
from copy import deepcopy

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from understanding_clouds.constants import REVERSED_LABELS_MAPPING, colorize_mask


class MaskRCNNPrediction:
    def __init__(self, raw_images, raw_outputs, raw_targets=None):
        self._raw_images = raw_images
        self._raw_outputs = raw_outputs
        self._raw_targets = raw_targets
        self.targets = []

    @staticmethod
    def _to_dcn(tensor):
        return tensor.detach().cpu().numpy()

    def filter_outputs(self, mask_thresh=0.5, score_threshold=0.25):
        self.images = []
        self.masks = []
        self.boxes = []
        self.labels = []
        self.scores = []
        for img, output in zip(self._raw_images, self._raw_outputs):
            scores = self._to_dcn(output['scores'])
            masks = self._to_dcn(
                (output['masks'] > mask_thresh).squeeze())
            labels = self._to_dcn(output['labels'])
            boxes = self._to_dcn(output['boxes'])
            to_preserve = scores > score_threshold
            scores, masks, labels, boxes = scores[to_preserve], masks[to_preserve], \
                labels[to_preserve], boxes[to_preserve]
            labels = [REVERSED_LABELS_MAPPING[l] for l in labels]
            self.scores.append(scores)
            self.masks.append(masks)
            self.labels.append(labels)
            self.boxes.append(boxes)
            self.images.append(self._to_dcn(img).transpose((1, 2, 0)))

    def draw_masks_at_img(self, outpath=None, draw_bb=False, transparency=0.3, rect_thickness=1, text_size=1, text_thickness=1):
        fig, axes = plt.subplots(figsize=(
            14 * len(self.images), 21), ncols=1, nrows=len(self.images), squeeze=False)
        for i, (img, masks, boxes, labels) in enumerate(zip(self.images, self.masks, self.boxes, self.labels)):
            img = deepcopy((img * 255).astype(np.uint8))
            for j, (mask, box, label) in enumerate(zip(masks, boxes, labels)):
                rgb_mask = colorize_mask(mask, label)
                img = cv2.addWeighted(img, 1, rgb_mask, transparency, 0)
                if draw_bb:
                    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=(
                        127, 127, 127), thickness=rect_thickness)
                cv2.putText(img, label, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX,
                            text_size, (127, 127, 127), thickness=text_thickness)
            axes[i, 0].imshow(img)
        if outpath is not None:
            plt.savefig(outpath)
        else:
            plt.show()
