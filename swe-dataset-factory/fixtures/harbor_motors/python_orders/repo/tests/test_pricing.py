from orderlib.pricing import PricingEngine


def test_line_total_basic() -> None:
    engine = PricingEngine()
    assert engine.line_total(10.0, 3) == 30.0


def test_discount_threshold() -> None:
    engine = PricingEngine(tax_rate=0.0, discount_threshold=100.0)
    assert engine.apply_discount(100.0) == 90.0
    assert engine.apply_discount(99.0) == 99.0
