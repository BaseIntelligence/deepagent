"""Inventory reservation with optimistic multi-sku accounting."""

from __future__ import annotations

from copy import deepcopy


class Inventory:
    """Track available stock and support multi-item reservations."""

    def __init__(self, stock: dict[str, int] | None = None) -> None:
        self._stock: dict[str, int] = dict(stock or {})

    def available(self, sku: str) -> int:
        return int(self._stock.get(sku, 0))

    def can_reserve(self, requests: dict[str, int]) -> bool:
        for sku, qty in requests.items():
            if qty < 0:
                return False
            if self.available(sku) < qty:
                return False
        return True

    def reserve(self, requests: dict[str, int]) -> bool:
        """Atomically reserve all requested SKUs or none."""
        if not self.can_reserve(requests):
            return False
        for sku, qty in requests.items():
            self._stock[sku] = self.available(sku) - int(qty)
        return True

    def release(self, requests: dict[str, int]) -> None:
        for sku, qty in requests.items():
            if qty < 0:
                raise ValueError("release quantity must be non-negative")
            self._stock[sku] = self.available(sku) + int(qty)

    def snapshot(self) -> dict[str, int]:
        return deepcopy(self._stock)
