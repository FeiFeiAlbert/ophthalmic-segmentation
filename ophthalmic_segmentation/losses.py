# -*- coding: utf-8 -*-
"""
Fundus image segmentation loss functions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        num_classes = pred.size(1)
        pred_softmax = F.softmax(pred, dim=1)
        target_onehot = torch.zeros_like(pred)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)

        dice_loss = 0
        for c in range(num_classes):
            pred_c = pred_softmax[:, c]
            target_c = target_onehot[:, c]
            intersection = (pred_c * target_c).sum(dim=(1, 2))
            union = pred_c.sum(dim=(1, 2)) + target_c.sum(dim=(1, 2))
            dice = (2 * intersection + self.smooth) / (union + self.smooth)
            dice_loss += (1 - dice).mean()
        return dice_loss / num_classes


class LovaszLoss(nn.Module):
    def forward(self, probas, labels):
        probas = F.softmax(probas, dim=1)
        probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, probas.size(1))
        labels = labels.contiguous().view(-1)
        return lovasz_softmax_flat(probas, labels)


def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.cumsum(0)
    union = gts + (1 - gt_sorted).cumsum(0)
    jaccard = 1. - intersection / (union + 1e-6)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels):
    num_classes = probas.size(1)
    losses = []
    for c in range(num_classes):
        fg = (labels == c).float()
        if fg.sum() == 0:
            continue
        class_pred = probas[:, c]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        fg_sorted = fg[perm]
        grad = lovasz_grad(fg_sorted)
        losses.append(torch.dot(errors_sorted, grad))
    return torch.stack(losses).mean() if losses else probas.sum() * 0


class CombinedLoss(nn.Module):
    """
    Combined loss for segmentation: 0.3*Focal + 0.4*Dice + 0.3*Lovasz
    Used in V16 to achieve Val Dice 0.9554
    """
    def __init__(self, focal_weight=0.3, dice_weight=0.4, lovasz_weight=0.3):
        super().__init__()
        self.focal = FocalLoss(alpha=1, gamma=2)
        self.dice = DiceLoss()
        self.lovasz = LovaszLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight

    def forward(self, pred, target):
        focal = self.focal(pred, target)
        dice = self.dice(pred, target)
        lovasz = self.lovasz(pred, target)
        return self.focal_weight * focal + self.dice_weight * dice + self.lovasz_weight * lovasz
