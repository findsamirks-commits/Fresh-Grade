"""Deep learning unit tests for Fresh-Grade Multi-Task ViT and Attention Rollout."""

import os
import sys
import pytest
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.model import FreshGradeMultiTaskViT, MultiTaskLoss, compute_attention_rollout


def test_model_forward_pass_and_multitask_loss():
    """Validates multi-task output dimensions, non-negative days, and SmoothL1 loss."""
    model = FreshGradeMultiTaskViT(num_classes=4, hidden_dim=128, pretrained=False)
    criterion = MultiTaskLoss(alpha=1.0, beta=0.5, label_smoothing=0.05)

    batch_size = 2
    dummy_imgs = torch.randn(batch_size, 3, 224, 224)
    target_cls = torch.tensor([1, 3], dtype=torch.long)
    target_days = torch.tensor([3.2, 0.4], dtype=torch.float32)

    logits, days, repr_embed, attentions = model(dummy_imgs, return_attentions=True)

    assert logits.shape == (batch_size, 4), "Logits must match (batch_size, num_classes)"
    assert days.shape == (batch_size, 1), "Days must match (batch_size, 1)"
    assert torch.all(days >= 0.0), "Shelf-life days must be non-negative via Softplus"
    assert repr_embed.shape == (batch_size, 128), "Projected embedding dimension mismatch"

    loss, l_cls, l_reg = criterion(logits, days, target_cls, target_days)
    assert loss.item() > 0.0
    assert l_cls.item() > 0.0
    assert l_reg.item() >= 0.0


def test_attention_rollout_computation():
    """Validates that attention rollout produces normalized 14x14 spatial patch heatmap."""
    # Test fallback mode
    rollout_fallback = compute_attention_rollout(None)
    assert rollout_fallback.shape == (14, 14)
    assert rollout_fallback.min() >= 0.0
    assert rollout_fallback.max() <= 1.0

    # Test synthetic layer attention tensors (2 layers, 4 heads, 197 tokens)
    num_tokens = 197
    attn_layer1 = torch.softmax(torch.randn(1, 4, num_tokens, num_tokens), dim=-1)
    attn_layer2 = torch.softmax(torch.randn(1, 4, num_tokens, num_tokens), dim=-1)

    rollout_tensor = compute_attention_rollout((attn_layer1, attn_layer2))
    assert rollout_tensor.shape == (14, 14)
    assert rollout_tensor.min() >= 0.0
    assert rollout_tensor.max() <= 1.0
