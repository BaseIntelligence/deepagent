from boltons.setutils import complement


def test_double_complement_returns_original():
    s = set(range(5))
    # Complementing a complement set must give back the original set.
    result = complement(complement(s))
    assert result == s
