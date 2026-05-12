import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class BertSelfAttention(nn.Module):
    """
    Multi-head Attention.

    - Self-Attention:  Q/K/V 모두 텍스트 hidden_states
    - Cross-Attention: Q = 텍스트, K/V = 이미지 encoder_hidden_states

    is_cross_attention=True이면 K/V Linear 입력 차원이
    hidden_size가 아닌 encoder_width로 바뀜.
    """
    def __init__(self, config: ModelConfig, is_cross_attention: bool = False):
        super().__init__()
        assert config.text.hidden_size % config.text.num_attention_heads == 0

        self.num_heads     = config.text.num_attention_heads
        self.head_dim      = config.text.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim

        # Q는 항상 텍스트 차원
        self.query = nn.Linear(config.text.hidden_size, self.all_head_size)

        # K/V: BlipVisionProjection이 encoder_width → hidden_size로 투영 후 전달하므로
        # cross-attention이어도 hidden_size 기준
        kv_dim = config.text.hidden_size
        self.key   = nn.Linear(kv_dim, self.all_head_size)
        self.value = nn.Linear(kv_dim, self.all_head_size)

        self.dropout = nn.Dropout(config.text.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """[B, seq, hidden] → [B, heads, seq, head_dim]"""
        B, S, _ = x.size()
        x = x.view(B, S, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,                   # [B, T, hidden]
        attention_mask: torch.Tensor = None,           # [B, 1, ?, ?] — 0 or -10000
        encoder_hidden_states: torch.Tensor = None,    # [B, I, hidden]
    ) -> torch.Tensor:

        Q = self.transpose_for_scores(self.query(hidden_states))

        # K/V 소스 결정
        kv_src = encoder_hidden_states if encoder_hidden_states is not None \
                 else hidden_states

        K = self.transpose_for_scores(self.key(kv_src))
        V = self.transpose_for_scores(self.value(kv_src))

        # Scaled dot-product
        # Self-Attn:  scores [B, H, T, T]
        # Cross-Attn: scores [B, H, T, I]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask

        probs  = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(probs, V)                # [B, H, T, head_dim]

        # [B, T, hidden]으로 복원
        output = output.permute(0, 2, 1, 3).contiguous()
        return output.view(output.size(0), output.size(1), self.all_head_size)


class BertSelfOutput(nn.Module):
    """Linear + Dropout + Residual + LayerNorm"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.dense     = nn.Linear(config.text.hidden_size, config.text.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.text.hidden_size, eps=config.text.layer_norm_eps)
        self.dropout   = nn.Dropout(config.text.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.LayerNorm(self.dropout(self.dense(hidden_states)) + residual)


class BertAttention(nn.Module):
    """BertSelfAttention + BertSelfOutput 묶음"""
    def __init__(self, config: ModelConfig, is_cross_attention: bool = False):
        super().__init__()
        self.self   = BertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
    ) -> torch.Tensor:
        attn_out = self.self(hidden_states, attention_mask, encoder_hidden_states)
        return self.output(attn_out, hidden_states)   # residual = 입력 원본
