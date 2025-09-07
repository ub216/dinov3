import torch
import torch.nn.functional as F

def iou_multiclass_torch(pred_logits, target, eps=1e-6):
    """
    Multi-class IoU (Jaccard index) for segmentation.

    Args:
        pred_logits: Tensor of shape (B, C, H, W) — raw model outputs
        target: Tensor of shape (B, H, W) (or B, 1, H, W) with class indices [0..C-1]
        eps: small constant for numerical stability

    Returns:
        iou: Tensor of shape (B, C) with IoU per class
    """
    num_classes = pred_logits.shape[1]

    # One-hot encode target: (B, C, H, W)
    if target.dim() == 4:
        if target.shape[1] != 1:
            assert(False), "Expected target with shape (B, 1, H, W) when 4D"
        target = target.squeeze(1)
    target_onehot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()

    # Predictions: choose highest scoring class per pixel
    pred_classes = torch.argmax(pred_logits, dim=1)  # (B, H, W)
    pred_onehot = F.one_hot(pred_classes, num_classes).permute(0, 3, 1, 2).float()

    # Intersection and union per class
    inter = (pred_onehot * target_onehot).sum(dim=(2, 3))
    union = pred_onehot.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3)) - inter + eps

    iou = (inter + eps) / union  # (B, C)

    return iou

def dice_multiclass_torch(pred_logits, target, eps=1e-6):
    """
    Multi-class Dice score for segmentation.
    
    Args:
        pred_logits: Tensor of shape (B, C, H, W) — raw model outputs
        target: Tensor of shape (B, H, W) with class indices [0..C-1]
        eps: small value to avoid division by zero
    
    Returns:
        dice: Tensor of shape (B, C) with dice per class
    """
    num_classes = pred_logits.shape[1]

    # One-hot encode target: (B, C, H, W)
    if target.dim() == 4:
        if target.shape[1] != 1:
            assert(False), "Expected target with shape (B, 1, H, W) when 4D"
        target = target.squeeze(1)
    target_onehot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).float()

    # Probabilities
    prob = torch.softmax(pred_logits, dim=1)

    # Intersection and union per class
    inter = (prob * target_onehot).sum(dim=(2, 3))
    union = prob.sum(dim=(2, 3)) + target_onehot.sum(dim=(2, 3)) + eps

    dice = (2 * inter + eps) / union  # (B, C)

    return dice

def iou_binary_torch(pred_logits, target, eps=1e-6, thresh=0.5):
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = (target > 0.5).float()
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - inter + eps
    iou = (inter + eps) / union
    return iou.view(-1)

def dice_binary_torch(pred_logits, target, eps=1e-6, thresh=0.5):
    prob = torch.sigmoid(pred_logits)
    pred = (prob > thresh).float()
    target = target.float().clamp(0, 1)
    inter = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps
    dice = (2 * inter + eps) / union
    return dice