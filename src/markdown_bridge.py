"""Markdown Bridge: Translates quality grade & days-to-spoil into dynamic markdown pricing.

Chains vision predictions into operational retail merchandising actions,
directly interfacing with Perishable-Markdown-Optimizer.
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


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


class MarkdownBridge:
    """Evaluates estimated days-to-spoil against perishable markdown ladders."""

    def __init__(self, base_price_inr: float = 50.0):
        self.base_price_inr = base_price_inr

    def evaluate(self, grade: str, days_to_spoil: float, base_price_inr: Optional[float] = None) -> MarkdownDecision:
        """Determines price markdown and FEFO inventory picking status."""
        base_p = base_price_inr if base_price_inr is not None else self.base_price_inr
        days = max(0.0, float(days_to_spoil))

        # Rule Ladder
        if grade == "Spoiled" or days < 0.5:
            action = "DUMP_SALVAGE"
            discount_pct = 100
            band = "<12h (Spoiled)"
            price = 0.0
            reason = "Produce reached spoilage threshold. Remove from dark store inventory immediately for zero-recovery disposal."

        elif grade == "Overripe" or days < 1.8:
            action = "DISCOUNT_40"
            discount_pct = 40
            band = "12–48h"
            price = round(base_p * 0.60, 2)
            reason = f"Accelerated decay risk ({days:.1f} days remaining). Aggressive 40% clearance discount to stimulate volume sellout."

        elif grade == "Ripe" or days < 4.0:
            action = "SELL_FIRST"
            discount_pct = 15
            band = "48–96h"
            price = round(base_p * 0.85, 2)
            reason = f"Optimal consumption window ({days:.1f} days remaining). Prioritize FEFO pick tote order with 15% markdown."

        else: # Fresh
            action = "HOLD"
            discount_pct = 0
            band = ">96h (Fresh)"
            price = round(base_p, 2)
            reason = f"Peak freshness ({days:.1f} days remaining). Maintain full gross margin at base price without discount."

        return MarkdownDecision(
            quality_grade=grade,
            days_to_spoil=round(days, 2),
            shelf_life_band=band,
            action=action,
            suggested_discount_pct=discount_pct,
            base_price_inr=base_p,
            suggested_price_inr=price,
            reason=reason,
        )
