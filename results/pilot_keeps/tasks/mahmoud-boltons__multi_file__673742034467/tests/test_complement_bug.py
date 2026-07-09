from boltons.setutils import complement


def test_double_complement_returns_original_set():
    s = set(range(5))
    result = complement(complement(s))
    assert result == s
