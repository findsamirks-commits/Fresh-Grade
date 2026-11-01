"""Unit tests for Fresh-Grade Markdown Bridge and Model dimensions."""

import os
import sys
import pytest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.markdown_bridge import MarkdownBridge
from src.model import FreshGradeMultiTaskViT, JointLoss


def test_markdown_bridge_rules():
    bridge = MarkdownBridge(base_price_inr=50.0)

    # 1. Fresh -> HOLD, 0% discount
    d_fresh = bridge.evaluate(grade="Fresh", days_to_spoil=6.5)
    assert d_fresh.action == "HOLD"
    assert d_fresh.suggested_discount_pct == 0
    assert d_fresh.suggested_price_inr == 50.0

    # 2. Ripe -> SELL_FIRST, 15% discount
    d_ripe = bridge.evaluate(grade="Ripe", days_to_spoil=2.8)
    assert d_ripe.action == "SELL_FIRST"
    assert d_ripe.suggested_discount_pct == 15
    assert d_ripe.suggested_price_inr == 42.50

    # 3. Overripe -> DISCOUNT_40, 40% discount
    d_overripe = bridge.evaluate(grade="Overripe", days_to_spoil=1.2)
    assert d_overripe.action == "DISCOUNT_40"
    assert d_overripe.suggested_discount_pct == 40
    assert d_overripe.suggested_price_inr == 30.00

    # 4. Spoiled -> DUMP_SALVAGE, 100% discount
    d_spoiled = bridge.evaluate(grade="Spoiled", days_to_spoil=0.2)
    assert d_spoiled.action == "DUMP_SALVAGE"
    assert d_spoiled.suggested_discount_pct == 100
    assert d_spoiled.suggested_price_inr == 0.0


def test_model_forward_pass_and_loss():
    model = FreshGradeMultiTaskViT(num_classes=4, hidden_dim=128, pretrained=False)
    criterion = JointLoss(alpha=1.0, beta=0.5)

    batch_size = 2
    dummy_imgs = torch.randn(batch_size, 3, 224, 224)
    target_cls = torch.tensor([1, 3], dtype=torch.long)
    target_days = torch.tensor([3.2, 0.4], dtype=torch.float32)

    logits, days, repr_embed = model(dummy_imgs)

    assert logits.shape == (batch_size, 4)
    assert days.shape == (batch_size, 1)
    assert torch.all(days >= 0.0), "Shelf-life days must be non-negative via Softplus"

    loss, l_cls, l_reg = criterion(logits, days, target_cls, target_days)
    assert loss.item() > 0.0
    assert not torch.isnan(loss)
