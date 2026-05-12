"""
Dummy input으로 각 모듈을 단계별로 검증.
전체 학습 코드 붙이기 전에 shape와 출력이 맞는지 확인.
"""

import torch

from config import MedConfig
from attention import BertAttention
from bert_layer import BertLayer
from bert_model import BertModel


# ── 공통 더미 설정 ────────────────────────────────────────────
B        = 2      # batch size
T        = 16     # text length
I        = 197    # image patch 수 (ViT: 196 patches + 1 CLS)
hidden   = 768

# 경량화 config (테스트용)
config_enc = MedConfig(num_hidden_layers=2, intermediate_size=1024,
                       add_cross_attention=False)
config_mm  = MedConfig(num_hidden_layers=2, intermediate_size=1024,
                       add_cross_attention=True)


def test_self_attention():
    """Self-Attention: [B, T, hidden] → [B, T, hidden]"""
    print("── Test 1: BertAttention (Self) ──")

    hidden_states = torch.randn(B, T, hidden)
    layer = BertAttention(config_enc, is_cross_attention=False)
    out   = layer(hidden_states)

    assert out.shape == (B, T, hidden), f"Expected {(B, T, hidden)}, got {out.shape}"
    print(f"  입력: {tuple(hidden_states.shape)}")
    print(f"  출력: {tuple(out.shape)}  ✓\n")


def test_cross_attention():
    """
    Cross-Attention: Q=텍스트, K/V=이미지
    scores shape: [B, H, T, I]
    출력: [B, T, hidden]  ← 텍스트 길이 기준
    """
    print("── Test 2: BertAttention (Cross) ──")

    hidden_states         = torch.randn(B, T, hidden)
    encoder_hidden_states = torch.randn(B, I, hidden)  # 이미지 feature

    layer = BertAttention(config_mm, is_cross_attention=True)
    out   = layer(hidden_states, encoder_hidden_states=encoder_hidden_states)

    assert out.shape == (B, T, hidden), f"Expected {(B, T, hidden)}, got {out.shape}"
    print(f"  텍스트 입력: {tuple(hidden_states.shape)}")
    print(f"  이미지 입력: {tuple(encoder_hidden_states.shape)}")
    print(f"  출력:        {tuple(out.shape)}  ← 텍스트 길이 기준 ✓\n")


def test_bert_layer_itc():
    """ITC 모드: Cross-Attention 없음"""
    print("── Test 3: BertLayer (ITC — no cross-attn) ──")

    hidden_states = torch.randn(B, T, hidden)
    layer = BertLayer(config_enc)
    out   = layer(hidden_states)

    assert out.shape == (B, T, hidden)
    print(f"  입력: {tuple(hidden_states.shape)}")
    print(f"  출력: {tuple(out.shape)}  ✓\n")


def test_bert_layer_itm():
    """ITM 모드: Bi Self-Attn + Cross-Attn"""
    print("── Test 4: BertLayer (ITM — with cross-attn) ──")

    hidden_states         = torch.randn(B, T, hidden)
    encoder_hidden_states = torch.randn(B, I, hidden)

    layer = BertLayer(config_mm)
    out   = layer(hidden_states, encoder_hidden_states=encoder_hidden_states)

    assert out.shape == (B, T, hidden)
    print(f"  텍스트 입력: {tuple(hidden_states.shape)}")
    print(f"  이미지 입력: {tuple(encoder_hidden_states.shape)}")
    print(f"  출력:        {tuple(out.shape)}  ✓\n")


def test_bert_model_encoder():
    """BertModel Encoder (ITC/ITM): pooler_output = [CLS] 토큰"""
    print("── Test 5: BertModel (Encoder — ITC) ──")

    input_ids   = torch.randint(100, 30000, (B, T))
    attn_mask   = torch.ones(B, T, dtype=torch.long)

    model = BertModel(config_enc, is_decoder=False)
    out   = model(input_ids, attn_mask)

    assert out["last_hidden_state"].shape == (B, T, hidden)
    assert out["pooler_output"].shape     == (B, hidden)
    print(f"  입력 토큰:        {tuple(input_ids.shape)}")
    print(f"  last_hidden_state:{tuple(out['last_hidden_state'].shape)}")
    print(f"  pooler_output:    {tuple(out['pooler_output'].shape)}  ← ITC에서 사용 ✓\n")


def test_bert_model_decoder():
    """BertModel Decoder (LM): causal mask + cross-attn"""
    print("── Test 6: BertModel (Decoder — LM) ──")

    input_ids             = torch.randint(100, 30000, (B, T))
    attn_mask             = torch.ones(B, T, dtype=torch.long)
    encoder_hidden_states = torch.randn(B, I, hidden)
    encoder_attn_mask     = torch.ones(B, I, dtype=torch.long)

    model = BertModel(config_mm, is_decoder=True)
    out   = model(input_ids, attn_mask,
                  encoder_hidden_states=encoder_hidden_states,
                  encoder_attention_mask=encoder_attn_mask)

    assert out["last_hidden_state"].shape == (B, T, hidden)
    print(f"  입력 토큰:        {tuple(input_ids.shape)}")
    print(f"  이미지 feature:   {tuple(encoder_hidden_states.shape)}")
    print(f"  last_hidden_state:{tuple(out['last_hidden_state'].shape)}  ← LM head 입력 ✓\n")


if __name__ == "__main__":
    print("=" * 50)
    print("  BLIP MED Dummy Input 단위 테스트")
    print("=" * 50 + "\n")

    test_self_attention()
    test_cross_attention()
    test_bert_layer_itc()
    test_bert_layer_itm()
    test_bert_model_encoder()
    test_bert_model_decoder()

    print("=" * 50)
    print("  전체 테스트 통과 ✓")
    print("=" * 50)
