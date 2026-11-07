"""Decoupled Business Unit Tests for Fresh-Grade Markdown Bridge.

Runs 100% without PyTorch installed to provide instantaneous feedback on
merchandising pricing ladders, FEFO picking logic, and edge-case governance.
"""

import os
import sys
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.markdown_bridge import MarkdownBridge, MarkdownDecision


def test_1_fresh_produce_holding():
    """Fresh produce with >4 days remaining maintains full price with zero markdown."""
    bridge = MarkdownBridge(base_price_inr=60.0)
    decision = bridge.evaluate(grade="Fresh", days_to_spoil=6.5)

    assert decision.action == "HOLD"
    assert decision.suggested_discount_pct == 0
    assert decision.suggested_price_inr == 60.00
    assert decision.shelf_life_band == ">96h (Fresh)"


def test_2_ripe_fefo_priority():
    """Ripe produce with 2-4 days remaining triggers FEFO picking priority with 15% markdown."""
    bridge = MarkdownBridge(base_price_inr=100.0)
    decision = bridge.evaluate(grade="Ripe", days_to_spoil=2.8)

    assert decision.action == "SELL_FIRST"
    assert decision.suggested_discount_pct == 15
    assert decision.suggested_price_inr == 85.00
    assert decision.shelf_life_band == "48–96h"


def test_3_overripe_clearance():
    """Overripe produce with 1-2 days remaining triggers aggressive 40% clearance discount."""
    bridge = MarkdownBridge(base_price_inr=50.0)
    decision = bridge.evaluate(grade="Overripe", days_to_spoil=1.2)

    assert decision.action == "DISCOUNT_40"
    assert decision.suggested_discount_pct == 40
    assert decision.suggested_price_inr == 30.00
    assert decision.shelf_life_band == "12–48h"


def test_4_spoiled_dump_salvage():
    """Spoiled produce with <1 day remaining triggers zero-recovery disposal."""
    bridge = MarkdownBridge(base_price_inr=80.0)
    decision = bridge.evaluate(grade="Spoiled", days_to_spoil=0.2)

    assert decision.action == "DUMP_SALVAGE"
    assert decision.suggested_discount_pct == 100
    assert decision.suggested_price_inr == 0.00
    assert decision.shelf_life_band == "<12h (Spoiled)"


def test_5_forced_dump_override_on_spoiled_grade():
    """Even if sensor/regression reports high days (e.g. 5.5d), Spoiled grade overrides to DUMP."""
    bridge = MarkdownBridge(base_price_inr=50.0)
    decision = bridge.evaluate(grade="Spoiled", days_to_spoil=5.5)

    assert decision.action == "DUMP_SALVAGE"
    assert decision.suggested_discount_pct == 100
    assert decision.suggested_price_inr == 0.00


def test_6_forced_overripe_minimum_discount_override():
    """Even if regression outputs 4.5 days, Overripe grade forces at least 40% discount."""
    bridge = MarkdownBridge(base_price_inr=100.0)
    decision = bridge.evaluate(grade="Overripe", days_to_spoil=4.5)

    assert decision.action == "DISCOUNT_40"
    assert decision.suggested_discount_pct == 40
    assert decision.suggested_price_inr == 60.00


def test_7_base_price_fallback_on_null_or_invalid():
    """Falls back to default base price when None, zero, or negative price is passed."""
    bridge = MarkdownBridge(base_price_inr=50.0)

    # Test None
    d_none = bridge.evaluate(grade="Fresh", days_to_spoil=5.0, base_price_inr=None)
    assert d_none.base_price_inr == 50.0
    assert d_none.suggested_price_inr == 50.0

    # Test zero
    d_zero = bridge.evaluate(grade="Fresh", days_to_spoil=5.0, base_price_inr=0.0)
    assert d_zero.base_price_inr == 50.0

    # Test negative
    d_neg = bridge.evaluate(grade="Ripe", days_to_spoil=3.0, base_price_inr=-10.0)
    assert d_neg.base_price_inr == 50.0
    assert d_neg.suggested_price_inr == 42.50


def test_8_negative_days_protection():
    """Negative estimated days are safely clamped to 0.0 and treated as DUMP_SALVAGE."""
    bridge = MarkdownBridge(base_price_inr=40.0)
    decision = bridge.evaluate(grade="Spoiled", days_to_spoil=-2.5)

    assert decision.days_to_spoil == 0.0
    assert decision.action == "DUMP_SALVAGE"
    assert decision.suggested_price_inr == 0.0


def test_9_custom_ladder_config_injection():
    """Verifies ladder can be instantiated with custom dictionary or YAML config."""
    custom_ladder = {
        "markdown_ladder": {
            "fresh": {"max_days": float("inf"), "discount_pct": 5, "action": "HOLD"},
            "ripe": {"max_days": 3.0, "discount_pct": 20, "action": "SELL_FIRST"},
        }
    }
    bridge = MarkdownBridge(base_price_inr=100.0, config=custom_ladder)
    assert bridge.ladder_config == custom_ladder
    decision = bridge.evaluate(grade="Fresh", days_to_spoil=7.0)
    assert decision.action == "HOLD"
    assert decision.base_price_inr == 100.0
