
from copy import deepcopy

import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from understanding_clouds.constants import LABELS_MAPPING, REVERSED_LABELS_MAPPING, colorize_mask, BACKGROUND_CLASSNAME


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

    def merge_predicted_masks_per_class(self):
        # First you must use `filter_outputs` function
        new_masks = []
        new_boxes = []
        new_labels = []
        for masks, boxes, labels in zip(self.masks, self.boxes, self.labels):
            concat_masks = []
            concat_labels = []
            concat_bbox = []
            for k in LABELS_MAPPING.keys():
                if k in labels:
                    lab_mask = np.array(labels) == k
                    merged_mask = masks[lab_mask].sum(axis=0)
                    merged_mask = (merged_mask >= 1).astype(np.uint8)
                    xmin = boxes[lab_mask][:, 0].min()
                    ymin = boxes[lab_mask][:, 1].min()
                    xmax = boxes[lab_mask][:, 2].max()
                    ymax = boxes[lab_mask][:, 3].max()
                    concat_bbox.append([xmin, ymin, xmax, ymax])
                    concat_masks.append(merged_mask)
                    concat_labels.append(k)
                elif k != BACKGROUND_CLASSNAME:
                    concat_labels.append(k)
                    concat_masks.append(np.zeros(masks.shape[1:]))
            new_boxes.append(np.array(concat_bbox))
            new_masks.append(np.array(concat_masks))
            new_labels.append(np.array(concat_labels))
        self.masks = new_masks
        self.boxes = new_boxes
        self.labels = new_labels

    def add_empty_masks_to_gt_data(self):
        all_gt_masks = []
        all_gt_boxes = []
        all_gt_labels = []
        for t in self._raw_targets:
            gt_masks = []
            gt_boxes = []
            gt_labels = []
            labels, masks, boxes = t['labels'], t['masks'], t['boxes']
            labels, masks, boxes = map(
                self._to_dcn, [labels, masks, boxes])
            for k in REVERSED_LABELS_MAPPING.keys():
                if k in labels:
                    lab_mask = np.array(labels) == k
                    lab_mask_ind = np.argmax(lab_mask)
                    gt_masks.append(
                        masks[lab_mask_ind][np.newaxis, :, :])
                    gt_boxes.append(boxes[lab_mask_ind])
                    gt_labels.append(k)
                elif k != 0:
                    gt_masks.append(
                        np.zeros((1, masks.shape[1], masks.shape[2])))
                    gt_boxes.append(np.array([0., 0., 350., 525.]))
                    gt_labels.append(k)
            all_gt_boxes.append(gt_boxes)
            all_gt_labels.append(gt_labels)
            all_gt_masks.append(np.concatenate(gt_masks, axis=0))
        self.gt_masks = all_gt_masks
        self.gt_boxes = all_gt_boxes
        self.gt_labels = all_gt_labels

    @staticmethod
    def _get_dice(prediction, target, eps=0.0001):
        overlap = np.multiply(prediction, target)
        overlap_sum = np.sum(overlap)
        pred_sum = np.sum(prediction)
        targ_sum = np.sum(target)

        coeff = (2 * overlap_sum + eps) / (pred_sum + targ_sum + eps)

        return coeff

    def get_dice_for_all_targets(self):
        # Use after applying `merge_predicted_masks_per_class`
        results = {}
        for masks, gt_masks, t in zip(self.masks, self.gt_masks, self._raw_targets):
            # TODO change image_id to filename
            gt_indices = [i for i, m in enumerate(gt_masks) if m.sum()>0]
            gt_masks_for_coeff = gt_masks[gt_indices]
            masks_for_coeff = masks[gt_indices]
            score = np.mean([self._get_dice(pred, gt) for pred, gt in zip(masks_for_coeff, gt_masks_for_coeff)])
            results[t['image_id'].item()] = score
        self.results = results
        return results

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
