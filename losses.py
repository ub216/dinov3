import torch
from torchvision.ops import sigmoid_focal_loss
from metrics import dice_multiclass_torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=-1, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, target):
        """
        logits: (B, C) raw outputs
        target: (B,) class indices
        """
        target_onehot = F.one_hot(target, num_classes=logits.shape[1])  # (B, H, W, C)
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()       # (B, C, H, W)

        loss = sigmoid_focal_loss(logits, target_onehot, alpha=self.alpha, gamma=self.gamma, reduction=self.reduction)
        return loss
        
def dice_loss_multiclass(pred_logits, target, eps=1e-6):
    dice = dice_multiclass_torch(pred_logits, target, eps)  # (B, C)
    return 1 - dice.mean()

class DicePlusFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, dice_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction='mean')
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, logits, target):
        focal = self.focal_loss(logits, target)
        dice = dice_loss_multiclass(logits, target)
        return self.focal_weight * focal + self.dice_weight * dice