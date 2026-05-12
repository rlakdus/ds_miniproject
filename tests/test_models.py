import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from config import ModelConfig, TextConfig, VisionConfig
from clip.clip_model import ClipModel
from blip.blip_model import BlipModel


def test_clip_model():
    B, T, I = 2, 16, 197
    config = ModelConfig(
        text=TextConfig(num_hidden_layers=2, intermediate_size=1024),
        projection_dim=512,
        add_cross_attention=False,
    )

    input_ids = torch.randint(0, 1000, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    image_features = torch.randn(B, I, config.vision.encoder_width)

    model = ClipModel(config)
    out = model(input_ids, attention_mask, image_features)

    assert out["text_embeddings"].shape == (B, config.projection_dim)
    assert out["image_embeddings"].shape == (B, config.projection_dim)
    assert out["logits_per_text"].shape == (B, B)
    assert out["logits_per_image"].shape == (B, B)
    print("ClipModel forward OK")


def test_blip_model():
    B, T, I, hidden = 2, 16, 197, 768
    config = ModelConfig(
        text=TextConfig(num_hidden_layers=2, intermediate_size=1024),
        add_cross_attention=True,
    )

    input_ids = torch.randint(0, 1000, (B, T))
    attention_mask = torch.ones(B, T, dtype=torch.long)
    image_features = torch.randn(B, I, config.vision.encoder_width)
    image_attention_mask = torch.ones(B, I, dtype=torch.long)

    model = BlipModel(config, is_decoder=False)
    out = model(input_ids, attention_mask,
                encoder_hidden_states=image_features,
                encoder_attention_mask=image_attention_mask)

    assert out["last_hidden_state"].shape == (B, T, hidden)
    assert out["pooler_output"].shape == (B, hidden)
    print("BlipModel encoder forward OK")


if __name__ == "__main__":
    test_clip_model()
    test_blip_model()
    print("All model tests passed.")
