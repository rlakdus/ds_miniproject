import torch
from torch.optim import AdamW

from losses import clip_contrastive_loss, itm_loss


def make_optimizer(model: torch.nn.Module,
                   lr: float = 1e-4,
                   weight_decay: float = 0.01) -> torch.optim.Optimizer:
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def clip_train_step(model: torch.nn.Module,
                    input_ids: torch.Tensor,
                    attention_mask: torch.Tensor,
                    image_features: torch.Tensor,
                    optimizer: torch.optim.Optimizer) -> float:
    model.train()
    outputs = model(input_ids, attention_mask, image_features)
    loss = clip_contrastive_loss(outputs['logits_per_text'], outputs['logits_per_image'])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def blip_itm_train_step(text_model: torch.nn.Module,
                        classifier: torch.nn.Module,
                        input_ids: torch.Tensor,
                        attention_mask: torch.Tensor,
                        encoder_hidden_states: torch.Tensor,
                        encoder_attention_mask: torch.Tensor,
                        labels: torch.Tensor,
                        optimizer: torch.optim.Optimizer) -> float:
    text_model.train()
    classifier.train()

    outputs = text_model(
        input_ids,
        attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
    )
    pooled = outputs['pooler_output']
    logits = classifier(pooled)

    loss = itm_loss(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
