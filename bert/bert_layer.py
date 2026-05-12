import torch
import torch.nn as nn

from config import ModelConfig
from .attention import BertAttention
from .ffn import BertIntermediate, BertOutput


class BertLayer(nn.Module):
    """
    논문 Figure 1의 단일 블록.

    모드별 동작:
    ┌─────────────────────────────────────────────────────┐
    │ Mode         │ attention_mask │ encoder_hidden_states│
    ├─────────────────────────────────────────────────────┤
    │ ITC (Encoder)│ bi-directional │ None                 │
    │ ITM (Encoder)│ bi-directional │ image features       │
    │ LM  (Decoder)│ causal (하삼각)│ image features       │
    └─────────────────────────────────────────────────────┘

    Causal vs Bi는 BertModel에서 mask를 만들어서 넘겨줌.
    Cross-Attention 유무는 config.add_cross_attention으로 결정.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = BertAttention(config)   # Self-Attention

        # Cross-Attention: ITM/LM 모드에서만 레이어 자체가 생성됨
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            self.crossattention = BertAttention(config, is_cross_attention=True)

        self.intermediate = BertIntermediate(config)
        self.output       = BertOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,                    # [B, T, hidden]
        attention_mask: torch.Tensor = None,            # [B, 1, T, T] or [B, 1, 1, T]
        encoder_hidden_states: torch.Tensor = None,     # [B, I, hidden]
        encoder_attention_mask: torch.Tensor = None,    # [B, 1, 1, I]
    ) -> torch.Tensor:

        # ① Self-Attention (Bi or Causal — mask로 결정)
        hidden_states = self.attention(hidden_states, attention_mask)

        # ② Cross-Attention (이미지 feature가 있고, 레이어도 있을 때만)
        if encoder_hidden_states is not None and self.add_cross_attention:
            hidden_states = self.crossattention(
                hidden_states,
                attention_mask=encoder_attention_mask,
                encoder_hidden_states=encoder_hidden_states,
            )

        # ③ FFN
        hidden_states = self.output(
            self.intermediate(hidden_states),
            hidden_states   # residual
        )
        return hidden_states
