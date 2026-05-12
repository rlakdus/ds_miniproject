import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from losses import clip_contrastive_loss, itm_loss, lm_loss


def test_clip_contrastive_loss():
    logits = torch.randn(2, 2)
    loss = clip_contrastive_loss(logits, logits)
    assert loss.item() > 0.0
    print('clip_contrastive_loss OK')


def test_itm_loss():
    logits = torch.randn(2, 2)
    labels = torch.tensor([0, 1])
    loss = itm_loss(logits, labels)
    assert loss.item() > 0.0
    print('itm_loss OK')


def test_lm_loss():
    logits = torch.randn(2, 4, 10)
    labels = torch.tensor([[1, 2, 3, -100], [0, 1, 2, 3]])
    loss = lm_loss(logits, labels)
    assert loss.item() > 0.0
    print('lm_loss OK')


if __name__ == '__main__':
    test_clip_contrastive_loss()
    test_itm_loss()
    test_lm_loss()
    print('All loss tests passed.')
