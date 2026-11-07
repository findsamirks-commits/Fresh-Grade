"""Markdown Bridge: Translates produce quality grade & days-to-spoil into dynamic markdown pricing.

Chains computer vision predictions into operational quick-commerce merchandising actions,
directly interfacing with Perishable-Markdown-Optimizer and FEFO dark-store picking logic.
"""

import os
from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field
import yaml


class MarkdownDecision(BaseModel):
    """Structured commercial pricing decision produced from perishable vision inference."""

    quality_grade: str = Field(description="Primary predicted grade: Fresh, Ripe, Overripe, Spoiled")
    days_to_spoil: float = Field(description="Estimated continuous days before complete spoilage")
    shelf_life_band: str = Field(description="Operational timeframe: '>96h', '48-96h', '24-48h', '<24h'")
    action: Literal["HOLD", "SELL_FIRST", "DISCOUNT_40", "DUMP_SALVAGE"] = Field(
        description="Merchandising action code"
    )
    suggested_discount_pct: int = Field(description="Percentage price markdown: 0 to 100%")
    base_price_inr: float = Field(description="Original base price before discount in INR")
    suggested_price_inr: float = Field(description="Discounted checkout price in INR")
    reason: str = Field(description="Commercial and inventory rationale explaining the markdown")


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "fresh_grade.yaml")


class MarkdownBridge:
    """Evaluates estimated days-to-spoil and quality grade against configurable markdown ladders."""

    def __init__(
        self,
        base_price_inr: float = 50.0,
        config: Optional[Union[str, Dict[str, Any]]] = None,
    ):
        self.default_base_price_inr = base_price_inr if base_price_inr and base_price_inr > 0 else 50.0
        self.ladder_config = self._load_config(config)

    def _load_config(self, config: Optional[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Loads ladder configuration from dict, file path, or default YAML."""
        if isinstance(config, dict):
            return config
        if isinstance(config, str) and os.path.exists(config):
            with open(config, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        if os.path.exists(DEFAULT_CONFIG_PATH):
            try:
                with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f)
            except Exception:
                pass
        # Hardcoded safe fallback defaults
        return {
            "markdown_ladder": {
                "fresh": {"max_days": float("inf"), "discount_pct": 0, "action": "HOLD"},
                "ripe": {"max_days": 4.0, "discount_pct": 15, "action": "SELL_FIRST"},
                "overripe": {"max_days": 1.8, "discount_pct": 40, "action": "DISCOUNT_40"},
                "spoiled": {"max_days": 0.5, "discount_pct": 100, "action": "DUMP_SALVAGE"},
            }
        }

    def evaluate(
        self,
        grade: str,
        days_to_spoil: float,
        base_price_inr: Optional[float] = None,
        sku_name: Optional[str] = None,
    ) -> MarkdownDecision:
        """Determines price markdown and FEFO inventory picking status.

        Applies strict business governance:
        1. Grade override: 'Spoiled' ALWAYS triggers DUMP_SALVAGE regardless of days.
        2. Grade override: 'Overripe' ALWAYS enforces at least 40% discount.
        3. Negative days protection: Clamped safely to 0.0.
        4. Base price fallback: Uses default price if null, non-positive, or missing.
        """
        base_p = (
            base_price_inr
            if base_price_inr is not None and base_price_inr > 0
            else self.default_base_price_inr
        )
        days = max(0.0, float(days_to_spoil))
        grade_clean = str(grade).strip().capitalize()

        # Grade-level hard business guardrails & overrides
        if grade_clean == "Spoiled" or days < 0.5:
            action = "DUMP_SALVAGE"
            discount_pct = 100
            band = "<12h (Spoiled)"
            price = 0.0
            reason = "Produce reached spoilage threshold. Remove from dark store inventory immediately for zero-recovery disposal."

        elif grade_clean == "Overripe" or days < 1.8:
            action = "DISCOUNT_40"
            discount_pct = 40
            band = "12–48h"
            price = round(base_p * 0.60, 2)
            reason = f"Accelerated decay risk ({days:.1f} days remaining). Aggressive 40% clearance discount to stimulate volume sellout."

        elif grade_clean == "Ripe" or days < 4.0:
            action = "SELL_FIRST"
            discount_pct = 15
            band = "48–96h"
            price = round(base_p * 0.85, 2)
            reason = f"Optimal consumption window ({days:.1f} days remaining). Prioritize FEFO pick tote order with 15% markdown."

        else:  # Fresh
            action = "HOLD"
            discount_pct = 0
            band = ">96h (Fresh)"
            price = round(base_p, 2)
            reason = f"Peak freshness ({days:.1f} days remaining). Maintain full gross margin at base price without discount."

        return MarkdownDecision(
            quality_grade=grade_clean,
            days_to_spoil=round(days, 2),
            shelf_life_band=band,
            action=action,
            suggested_discount_pct=discount_pct,
            base_price_inr=base_p,
            suggested_price_inr=price,
            reason=reason,
        )
