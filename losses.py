import torch
import torch.nn as nn


def clip_contrastive_loss(logits_per_text: torch.Tensor,
                          logits_per_image: torch.Tensor,
                          temperature: float = 1.0) -> torch.Tensor:
    """CLIP / ITC 스타일 contrastive loss.

    logits_per_text: [B, B]
    logits_per_image: [B, B]
    """
    batch_size = logits_per_text.size(0)
    labels = torch.arange(batch_size, device=logits_per_text.device)
    loss_fct = nn.CrossEntropyLoss()

    loss_text = loss_fct(logits_per_text / temperature, labels)
    loss_image = loss_fct(logits_per_image / temperature, labels)
    return (loss_text + loss_image) / 2.0


def itm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """ITM 분류 loss.

    logits: [B, num_labels]
    labels: [B]
    """
    loss_fct = nn.CrossEntropyLoss()
    return loss_fct(logits, labels)


def lm_loss(logits: torch.Tensor,
            labels: torch.Tensor,
            ignore_index: int = -100) -> torch.Tensor:
    """Language modeling loss.

    logits: [B, T, V]
    labels: [B, T]
    """
    loss_fct = nn.CrossEntropyLoss(ignore_index=ignore_index)
    return loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
