import numpy as np
import torch


class DiceLoss(torch.nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation=torch.nn.Sigmoid()):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, pred, targ):
        return 1 - get_DiceCoefficient(pred, targ, eps=self.eps, activation=self.activation)


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation=torch.nn.Sigmoid(), lambda_dice=1.0, lambda_bce=1.0):
        super().__init__(eps, activation)
        if activation == None:
            self.bce = torch.nn.BCELoss(reduction='mean')
        else:
            self.bce = torch.nn.BCEWithLogitsLoss(reduction='mean')
        self.lambda_dice = lambda_dice
        self.lambda_bce = lambda_bce

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return (self.lambda_dice * dice) + (self.lambda_bce * bce)


def get_DiceCoefficient(prediction, target, eps, activation):

    if activation is not None:
        prediction = activation(prediction)

    overlap = prediction * target
    overlap_sum = torch.sum(overlap)
    pred_sum = torch.sum(prediction)
    targ_sum = torch.sum(target)

    coeff = (2 * overlap_sum + eps) / (pred_sum + targ_sum + eps)

    return coeff
