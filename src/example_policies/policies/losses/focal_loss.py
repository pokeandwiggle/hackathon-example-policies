import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTerminationLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, focal_idx=-1):
        super(FocalTerminationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.focal_idx = focal_idx

    def forward(self, inputs, targets):
        """
        Compute the focal loss between `inputs` and the ground truth `targets`.

        Args:
            inputs (torch.Tensor): Predicted logits with shape (B, T, A).
            targets (torch.Tensor): Ground truth binary labels with shape (B, T, A).

        Returns:
            torch.Tensor: Computed focal loss.
        """

        term_index = self.focal_idx
        assert targets.shape[-1] > term_index, "Targets do not have termination signal."

        inputs_term = inputs[..., term_index]
        targets_term = targets[..., term_index]

        probas = torch.sigmoid(inputs_term)
        ce_loss = F.binary_cross_entropy_with_logits(
            inputs_term, targets_term, reduction="none"
        )

        p_t = probas * targets_term + (1 - probas) * (1 - targets_term)
        focal_factor = (1 - p_t) ** self.gamma

        loss = self.alpha * focal_factor * ce_loss

        # Populate dictionary for logging
        loss_dict = {
            "term_loss": loss.mean().item(),
        }

        return loss.unsqueeze(-1), loss_dict
