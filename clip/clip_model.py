import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig
from bert.bert_model import BertModel


class ClipTextEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.text_encoder = BertModel(config, is_decoder=False)
        self.projection = nn.Linear(config.text.hidden_size, config.projection_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.text_encoder(input_ids, attention_mask)
        pooled = outputs["pooler_output"]
        return self.projection(pooled)


class ClipImageEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.vision.encoder_width != config.projection_dim:
            self.projection = nn.Linear(config.vision.encoder_width, config.projection_dim)
        else:
            self.projection = nn.Identity()

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        # image_features: [B, I, encoder_width]
        pooled = image_features.mean(dim=1)
        return self.projection(pooled)


class ClipModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.text_encoder = ClipTextEncoder(config)
        self.image_encoder = ClipImageEncoder(config)
        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_features: torch.Tensor,
    ) -> dict:
        text_features = self.text_encoder(input_ids, attention_mask)
        image_features = self.image_encoder(image_features)

        text_features = F.normalize(text_features, dim=-1)
        image_features = F.normalize(image_features, dim=-1)
        logit_scale = self.logit_scale.exp()

        return {
            "text_embeddings": text_features,
            "image_embeddings": image_features,
            "logit_scale": logit_scale,
            "logits_per_text": logit_scale * torch.matmul(text_features, image_features.t()),
            "logits_per_image": logit_scale * torch.matmul(image_features, text_features.t()),
        }
