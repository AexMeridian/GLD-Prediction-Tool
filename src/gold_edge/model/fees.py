"""Kalshi's per-order trading fee.

fee = ceil_to_cent(fee_multiplier * base_rate * contracts * price * (1 - price))

Confirmed against a live KXGOLD15M series (fee_multiplier=1, fee_type=
"quadratic") — see docs/contract_notes.md. Maker (resting) orders have paid
the same formula since 2026-08-19; before that they paid nothing, which is
why `maker_fee` takes an explicit enabled flag instead of assuming it's
always on (useful for backtesting periods before that date).
"""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gold_edge.config import FeesConfig
    from gold_edge.models import Position

CENT = Decimal("0.01")


def _validate_price(price: Decimal) -> None:
    if price < 0 or price > 1:
        raise ValueError(f"price must be in [0, 1] dollars, got {price}")


def taker_fee(
    contracts: Decimal,
    price: Decimal,
    fee_multiplier: Decimal,
    base_rate: Decimal = Decimal("0.07"),
) -> Decimal:
    _validate_price(price)
    raw = fee_multiplier * base_rate * contracts * price * (Decimal(1) - price)
    return raw.quantize(CENT, rounding=ROUND_CEILING)


def maker_fee(
    contracts: Decimal,
    price: Decimal,
    fee_multiplier: Decimal,
    base_rate: Decimal = Decimal("0.07"),
    maker_fees_enabled: bool = True,
) -> Decimal:
    if not maker_fees_enabled:
        _validate_price(price)
        return Decimal("0.00")
    return taker_fee(contracts, price, fee_multiplier, base_rate)


def settle_position_pnl(position: Position, result: str, fees_cfg: FeesConfig) -> Decimal:
    """P&L for a position carried into settlement (win pays $1/contract,
    loss pays $0 — no settlement fee, only the entry fee already paid).
    `result` is Kalshi's market result field, "yes" or "no".

    Lives here (not in server.py, where it was originally defined) so
    backtest/replay.py and learning/opportunities.py can import it without
    depending on server.py at all -- server.py itself now imports learning
    modules for the Review/Learning tab endpoints, and server -> ... ->
    backtest.replay -> server would otherwise be a circular import.
    """
    from gold_edge.models import Side

    won = (result == "yes" and position.side is Side.YES) or (
        result == "no" and position.side is Side.NO
    )
    settlement_value = Decimal(1) if won else Decimal(0)
    entry_fee = taker_fee(
        position.size,
        position.entry_price,
        Decimal(str(fees_cfg.fee_multiplier)),
        Decimal(str(fees_cfg.base_rate)),
    )
    return (settlement_value - position.entry_price) * position.size - entry_fee
