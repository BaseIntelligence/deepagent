from demo_pkg.text_ops import reverse_words


def test_reverse_words() -> None:
    assert reverse_words("hello world") == "world hello"
