import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class BertIntermediate(nn.Module):
    """
    FFN 첫 번째 단계.
    hidden_size → intermediate_size, GELU 활성화.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense = nn.Linear(config.text.hidden_size, config.text.intermediate_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, seq, 768] → [B, seq, 3072]
        return F.gelu(self.dense(x))


class BertOutput(nn.Module):
    """
    FFN 두 번째 단계.
    intermediate_size → hidden_size, Residual + LayerNorm.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense     = nn.Linear(config.text.intermediate_size, config.text.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.text.hidden_size, eps=config.text.layer_norm_eps)
        self.dropout   = nn.Dropout(config.text.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # [B, seq, 3072] → [B, seq, 768]
        return self.LayerNorm(self.dropout(self.dense(hidden_states)) + residual)
