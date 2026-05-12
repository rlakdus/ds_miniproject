import torch
import torch.nn as nn

from config import ModelConfig
from bert.bert_model import BertModel


class BlipVisionProjection(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.vision.encoder_width != config.text.hidden_size:
            self.projection = nn.Linear(config.vision.encoder_width, config.text.hidden_size)
        else:
            self.projection = nn.Identity()

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.projection(image_features)


class BlipModel(nn.Module):
    def __init__(self, config: ModelConfig, is_decoder: bool = False):
        super().__init__()
        self.config = config
        self.is_decoder = is_decoder
        self.text_module = BertModel(config, is_decoder=is_decoder)
        self.vision_projection = BlipVisionProjection(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
    ) -> dict:
        if encoder_hidden_states is not None:
            encoder_hidden_states = self.vision_projection(encoder_hidden_states)

        return self.text_module(
            input_ids,
            attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )
