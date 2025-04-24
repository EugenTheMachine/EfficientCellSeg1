import torch
import torch.nn.functional as F


def cross_entropy_loss(
    true_masks: torch.Tensor, pred_logits: torch.Tensor, true_cell_prob: torch.Tensor
):
    selection_mask = true_cell_prob == 1
    if selection_mask.sum() == 0:
        return torch.tensor(0.0, device=true_masks.device)

    selected_true_masks = true_masks[selection_mask]
    selected_pred_logits = pred_logits[selection_mask]

    loss = F.binary_cross_entropy_with_logits(
        selected_pred_logits, selected_true_masks, reduction="mean"
    )
    return loss


def cell_prob_mse_loss(
    true_cell_prob: torch.Tensor,
    pred_cell_prob: torch.Tensor,
) -> torch.Tensor:
    loss = F.mse_loss(
        input=pred_cell_prob,
        target=true_cell_prob,
        reduction="mean",
    )
    return loss
