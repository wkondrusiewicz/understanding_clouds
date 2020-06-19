from copy import deepcopy
import numpy as np
import matplotlib as plt


class UnetPrediction:
    def __init__(self, pred_masks, gt_masks, img,  filename, thresh: float=0.5):
        self.pred_masks = deepcopy(pred_masks)
        self.gt_masks = deepcopy(gt_masks)
        self.img = deepcopy(img)
        self.filename = filename
        self.postprocess(thresh)

    @staticmethod
    def _get_dice(prediction, target, eps=0.0001):
        overlap = np.multiply(prediction, target)
        overlap_sum = np.sum(overlap)
        pred_sum = np.sum(prediction)
        targ_sum = np.sum(target)

        coeff = (2 * overlap_sum + eps) / (pred_sum + targ_sum + eps)

        return coeff

    def _binarize_masks(self, thresh: float=0.5):
        self.pred_masks = (self.pred_masks > thresh).astype(self.pred_masks.dtype)

    def _get_non_empty_masks_indices(self):
        gt = {i for i, mask in enumerate(self.gt_masks) if np.sum(mask) > 0}
        pred = {i for i, mask in enumerate(self.pred_masks) if np.sum(mask) > 0}
        return list(gt.union(pred))


    def draw_masks(self, outpath=None):
        assert self.score is not None, 'please first calculate dice score'
        fig, axes = plt.subplots(figsize=(5 * len(self.gt_masks), 25), ncols=2, nrows=len(self.gt_masks) + 1, squeeze=False)
        plt.suptitle(f'Results for {self.filename} with dice score of {self.score}', fontsize=20, y=0.92)
        for i, (pred, gt) in enumerate(zip(self.pred_masks, self.gt_masks)):
            name = REVERSED_LABELS_MAPPING[i]
            axes[i][0].imshow(pred)
            axes[i][0].set_title(f'Pred for class: {name}')
            axes[i][1].imshow(gt)
            axes[i][0].set_title(f'Ground truth for class: {name}')
        axes[i + 1][0].imshow(self.img)
        axes[i + 1][0].set_title('Image')
        axes[i + 1][1].imshow(self.img)
        axes[i + 1][1].set_title('Image')
        plt.show()

    def compute_dice_score_for_all_targets(self):
        results = {}
        indices = self._get_non_empty_masks_indices()
        gt, pred = self.gt_masks[indices], self.pred_masks[indices]
        scores = [self._get_dice(p, g) for p,g in zip(gt, pred)]
        score = np.mean(scores)
        self.score = score

    def postprocess(self, thresh):
        self._binarize_masks(thresh)
        self.compute_dice_score_for_all_targets()
