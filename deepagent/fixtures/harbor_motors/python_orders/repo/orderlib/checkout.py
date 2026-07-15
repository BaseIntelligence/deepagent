"""Checkout orchestration across pricing and inventory modules."""

from __future__ import annotations

from dataclasses import dataclass

from orderlib.inventory import Inventory
from orderlib.pricing import PricingEngine


@dataclass(frozen=True)
class LineItem:
    sku: str
    unit_price: float
    quantity: int


@dataclass(frozen=True)
class OrderResult:
    accepted: bool
    total: float
    remaining: dict[str, int]
    reason: str = ""


class CheckoutService:
    """Coordinate reservation + multi-line pricing into an order result."""

    def __init__(self, inventory: Inventory, pricing: PricingEngine | None = None) -> None:
        self.inventory = inventory
        self.pricing = pricing or PricingEngine()

    def place_order(self, items: list[LineItem]) -> OrderResult:
        if not items:
            return OrderResult(
                accepted=False,
                total=0.0,
                remaining=self.inventory.snapshot(),
                reason="empty_cart",
            )
        requests = {item.sku: item.quantity for item in items}
        # Aggregate duplicate SKUs
        aggregated: dict[str, int] = {}
        for sku, qty in requests.items():
            aggregated[sku] = aggregated.get(sku, 0) + int(qty)

        if not self.inventory.can_reserve(aggregated):
            return OrderResult(
                accepted=False,
                total=0.0,
                remaining=self.inventory.snapshot(),
                reason="insufficient_stock",
            )
        reserved = self.inventory.reserve(aggregated)
        if not reserved:
            return OrderResult(
                accepted=False,
                total=0.0,
                remaining=self.inventory.snapshot(),
                reason="reserve_failed",
            )
        lines = [(item.unit_price, item.quantity) for item in items]
        total = self.pricing.cart_total(lines)
        return OrderResult(
            accepted=True,
            total=total,
            remaining=self.inventory.snapshot(),
            reason="ok",
        )
