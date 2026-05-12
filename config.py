from dataclasses import dataclass


@dataclass
class MedConfig:
    """
    BLIP MED (Multimodal Mixture of Encoder-Decoder) 설정값.
    BERT-base 기본값 기준.
    """
    # 텍스트 모듈 설정
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.0
    attention_probs_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-12
    vocab_size: int = 30522
    pad_token_id: int = 0

    # BLIP 전용 설정
    encoder_width: int = 768        # ViT 출력 차원 (cross-attn K/V 입력)
    add_cross_attention: bool = False  # True → ITM/LM 모드
