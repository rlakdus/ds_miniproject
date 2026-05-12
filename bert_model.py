import torch
import torch.nn as nn

from config import MedConfig
from bert_layer import BertLayer


class BertEmbeddings(nn.Module):
    """Token ID → 벡터 변환 (word embedding + position embedding)"""
    def __init__(self, config: MedConfig):
        super().__init__()
        self.word_embeddings     = nn.Embedding(config.vocab_size, config.hidden_size,
                                                padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(512, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout   = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, S = input_ids.shape
        pos  = torch.arange(S, device=input_ids.device).unsqueeze(0)
        emb  = self.word_embeddings(input_ids) + self.position_embeddings(pos)
        return self.dropout(self.LayerNorm(emb))   # [B, seq, hidden]


class BertEncoder(nn.Module):
    """BertLayer를 num_hidden_layers개 쌓은 것"""
    def __init__(self, config: MedConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        encoder_hidden_states: torch.Tensor = None,
        encoder_attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
            )
        return hidden_states


class BertModel(nn.Module):
    """
    MED의 실제 텍스트 모듈.

    is_decoder=False → Encoder (ITC / ITM)
      - bi-directional attention mask
      - pooler_output: [CLS] 토큰 → ITC 유사도 / ITM 분류에 사용

    is_decoder=True  → Decoder (LM)
      - causal attention mask (하삼각 행렬)
      - last_hidden_state: 각 토큰 → lm_head로 next token 예측
    """
    def __init__(self, config: MedConfig, is_decoder: bool = False):
        super().__init__()
        self.config     = config
        self.is_decoder = is_decoder
        self.embeddings = BertEmbeddings(config)
        self.encoder    = BertEncoder(config)
        self.pooler     = nn.Linear(config.hidden_size, config.hidden_size)

    def get_extended_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        [B, seq] → [B, 1, seq, seq] 형태로 확장.

        Encoder: bi-directional — padding 위치만 -10000
        Decoder: causal — 미래 토큰 + padding 모두 -10000

        Shape 정리:
          Encoder mask: [B, 1, 1, T]  → [B, H, T, T]에 broadcast
          Decoder mask: [B, 1, T, T]  → [B, H, T, T]에 broadcast
        """
        B, S   = attention_mask.shape
        device = attention_mask.device

        if self.is_decoder:
            # 하삼각 행렬: i번째 토큰은 0~i까지만 볼 수 있음
            causal = torch.tril(torch.ones(S, S, device=device))   # [S, S]
            mask   = causal.unsqueeze(0) * attention_mask[:, None, None, :]  # [B, 1, S, S]
        else:
            mask = attention_mask[:, None, None, :]  # [B, 1, 1, S]

        # 0인 위치 → -10000 (softmax 후 ≈ 0)
        return (1.0 - mask.float()) * -10000.0

    def forward(
        self,
        input_ids: torch.Tensor,                        # [B, seq]
        attention_mask: torch.Tensor = None,            # [B, seq]
        encoder_hidden_states: torch.Tensor = None,     # [B, I, encoder_width]
        encoder_attention_mask: torch.Tensor = None,    # [B, I]
    ) -> dict:

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # [B, seq] → [B, 1, seq, seq] (Bi or Causal)
        extended_mask = self.get_extended_attention_mask(attention_mask)

        # 이미지 mask도 [B, I] → [B, 1, 1, I]로 확장
        ext_enc_mask = None
        if encoder_attention_mask is not None:
            ext_enc_mask = (1.0 - encoder_attention_mask[:, None, None, :].float()) * -10000.0

        # Embedding → BertLayer × N
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.encoder(
            hidden_states,
            attention_mask=extended_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=ext_enc_mask,
        )

        # [CLS] 토큰 pooling
        pooled = torch.tanh(self.pooler(hidden_states[:, 0]))

        return {
            "last_hidden_state": hidden_states,   # [B, seq, hidden] — LM에서 사용
            "pooler_output":     pooled,           # [B, hidden]      — ITC/ITM에서 사용
        }
