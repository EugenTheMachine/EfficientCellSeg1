import torch
import torch.nn.functional as F


# def cross_entropy_loss(
#     true_masks: torch.Tensor, pred_logits: torch.Tensor, true_cell_prob: torch.Tensor
# ):
#     selection_mask = true_cell_prob == 1
#     if selection_mask.sum() == 0:
#         return torch.tensor(0.0, device=true_masks.device)

#     selected_true_masks = true_masks[selection_mask]
#     selected_pred_logits = pred_logits[selection_mask]

#     loss = F.binary_cross_entropy_with_logits(
#         selected_pred_logits, selected_true_masks, reduction="mean"
#     )
#     return loss


def cross_entropy_loss(
    true_masks: torch.Tensor, pred_logits: torch.Tensor,
    true_cell_prob: torch.Tensor, alpha: float = 0.25,
    gamma: float = 2, smooth: float = 0.1
):
    """
    Modification of original CE-loss with added Focal Loss and Dice Loss.
    This is supposed to better handle our cells by applying stricter penalties.
    """

    selection_mask = true_cell_prob == 1
    if selection_mask.sum() == 0:
        return torch.tensor(0.0, device=true_masks.device)

    selected_true_masks = true_masks[selection_mask]
    selected_pred_logits = pred_logits[selection_mask]

    ce_loss = F.binary_cross_entropy_with_logits(
        selected_pred_logits, selected_true_masks, reduction="mean"
    )

    selected_pred = torch.sigmoid(selected_pred_logits)
    p_t = selected_pred * true_masks + (1 - selected_pred) * (1 - true_masks)
    focal_loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * true_masks + (1 - alpha) * (1 - true_masks)
        focal_loss = alpha_t * focal_loss

    intersection = (selected_pred * true_masks).sum(dim=(1,2))
    union = (selected_pred + true_masks).sum(dim=(1,2))
    dice = (2. * intersection + smooth) / (union + smooth)
    dice_loss = 1 - dice.mean()

    loss = ce_loss + focal_loss.mean() + dice_loss

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
