from orderlib.inventory import Inventory


def test_reserve_atomic() -> None:
    inv = Inventory({"sku-a": 5, "sku-b": 2})
    assert inv.can_reserve({"sku-a": 3, "sku-b": 2})
    assert inv.reserve({"sku-a": 3, "sku-b": 2}) is True
    assert inv.available("sku-a") == 2
    assert inv.available("sku-b") == 0


def test_reserve_rejects_partial() -> None:
    inv = Inventory({"sku-a": 1})
    assert inv.reserve({"sku-a": 2}) is False
    assert inv.available("sku-a") == 1
