from orderlib.checkout import CheckoutService, LineItem
from orderlib.inventory import Inventory
from orderlib.pricing import PricingEngine


def test_place_order_success() -> None:
    inv = Inventory({"widget": 10})
    svc = CheckoutService(inv, PricingEngine(tax_rate=0.10, discount_threshold=50.0))
    result = svc.place_order([LineItem(sku="widget", unit_price=20.0, quantity=3)])
    assert result.accepted is True
    assert result.reason == "ok"
    # subtotal 60 → discount 54 → tax 59.4
    assert result.total == 59.4
    assert result.remaining["widget"] == 7


def test_place_order_insufficient_stock() -> None:
    inv = Inventory({"widget": 1})
    svc = CheckoutService(inv)
    result = svc.place_order([LineItem(sku="widget", unit_price=5.0, quantity=3)])
    assert result.accepted is False
    assert result.reason == "insufficient_stock"
