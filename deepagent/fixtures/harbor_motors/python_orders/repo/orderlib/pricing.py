"""Pricing engine with discount bands and tax application."""

from __future__ import annotations


class PricingEngine:
    """Compute subtotal, discount, and tax for cart lines."""

    def __init__(self, tax_rate: float = 0.10, discount_threshold: float = 100.0) -> None:
        if tax_rate < 0:
            raise ValueError("tax_rate must be non-negative")
        self.tax_rate = float(tax_rate)
        self.discount_threshold = float(discount_threshold)

    def line_total(self, unit_price: float, quantity: int) -> float:
        if quantity < 0:
            raise ValueError("quantity must be non-negative")
        return float(unit_price) * int(quantity)

    def apply_discount(self, subtotal: float) -> float:
        """10% off when subtotal reaches the configured threshold; else identity."""
        sub = float(subtotal)
        if sub >= self.discount_threshold:
            return round(sub * 0.9, 2)
        return round(sub, 2)

    def apply_tax(self, amount: float) -> float:
        return round(float(amount) * (1.0 + self.tax_rate), 2)

    def cart_total(self, lines: list[tuple[float, int]]) -> float:
        subtotal = sum(self.line_total(price, qty) for price, qty in lines)
        discounted = self.apply_discount(subtotal)
        return self.apply_tax(discounted)
