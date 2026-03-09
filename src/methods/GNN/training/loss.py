import torch
import torch.nn.functional as F

def binary_loss(logits, labels, pos_weight=10.0):
    """
    Weighted BCE loss for binary classification (Section 5.1.2).
    """
    weight_tensor = torch.tensor([pos_weight], device=logits.device)
    return F.binary_cross_entropy_with_logits(
        logits.squeeze(-1),
        labels.float(),
        pos_weight=weight_tensor
    )