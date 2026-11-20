"""Transaction costs for delivery trades in NSE cash equities.

Statutory charges (rates as of FY2025-26, delivery segment):
    STT              0.100% of value, both buy and sell
    Stamp duty       0.015% of value, buy only
    Exchange txn     0.00297% of value (NSE), both sides
    SEBI fee         Rs 10 per crore = 0.0001%, both sides
    GST              18% on (brokerage + exchange + SEBI fees)
Brokerage is an input because institutional rates are negotiated.

Market impact uses the square-root law:
    impact = k * sigma_daily * sqrt(order_value / ADV_value)
with k ~ 1 (Torre 1997; Almgren et al. 2005 report the exponent close to 0.5
across markets). Impact is charged on each side of a trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

STT = 0.001
STAMP_BUY = 0.00015
EXCHANGE = 0.0000297
SEBI = 0.000001
GST = 0.18


@dataclass(frozen=True)
class CostModel:
    aum: float = 5e9  # Rs 500 crore
    brokerage: float = 0.0002  # 2 bps institutional
    impact_k: float = 1.0
    include_impact: bool = True

    def trade_cost(
        self,
        trades: np.ndarray,
        daily_vol: np.ndarray,
        adv_value: np.ndarray,
    ) -> tuple[float, float]:
        """Cost of a rebalance as a fraction of AUM.

        ``trades`` are weight changes (positive = buy). Returns
        (explicit_cost, impact_cost).
        """
        buys = np.clip(trades, 0, None)
        sells = np.clip(-trades, 0, None)
        gross = buys + sells
        fees = self.brokerage + EXCHANGE + SEBI
        explicit = (
            gross.sum() * (STT + fees * (1 + GST)) + buys.sum() * STAMP_BUY
        )
        impact = 0.0
        if self.include_impact:
            order_value = gross * self.aum
            with np.errstate(divide="ignore", invalid="ignore"):
                participation = np.where(adv_value > 0, order_value / adv_value, 0.0)
            impact = float(np.sum(gross * self.impact_k * daily_vol * np.sqrt(participation)))
        return float(explicit), impact
