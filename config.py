from dataclasses import dataclass, field


@dataclass
class TextConfig:
    """BERT 텍스트 인코더 설정. BERT-base 기본값 기준."""
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-12
    vocab_size: int = 30522
    pad_token_id: int = 0


@dataclass
class VisionConfig:
    """비전 인코더(ViT) 설정."""
    encoder_width: int = 768


@dataclass
class ModelConfig:
    """CLIP / BLIP 모델 최상위 설정."""
    text: TextConfig = field(default_factory=TextConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    projection_dim: int = 512        # CLIP ITC 임베딩 차원
    add_cross_attention: bool = False  # True → ITM/LM 모드 (BertLayer에 cross-attn 추가)
