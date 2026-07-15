"""PASS_TO_PASS regression: pure helper that stays green without gold."""


def test_always_ok() -> None:
    assert 1 + 1 == 2
