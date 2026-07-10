import pytest

from boltons.dictutils import OneToOne
from boltons.statsutils import Stats


def test_one_to_one_basic():
    oto = OneToOne({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto['b'] == 2
    assert oto.inv[1] == 'a'
    assert oto.inv[2] == 'b'
    assert len(oto) == 2
    assert len(oto.inv) == 2


def test_one_to_one_with_dupe_values_keeps_last():
    # Values are duplicated: len(self) != len(self.inv), so it rebuilds
    # from inv, effectively deduping so only one key survives per value.
    oto = OneToOne({'a': 1, 'b': 1})
    # After rebuild, inv has {1: <one of a/b>}, forward is inverse of that
    assert len(oto) == 1
    assert len(oto.inv) == 1
    # the surviving value must be 1
    assert list(oto.values()) == [1]
    # and it must round-trip
    only_key = list(oto.keys())[0]
    assert only_key in ('a', 'b')
    assert oto.inv[1] == only_key
    assert oto[only_key] == 1


def test_one_to_one_unique_raises_on_dupe():
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 1, 'b': 1})


def test_one_to_one_unique_ok():
    oto = OneToOne.unique({'a': 1, 'b': 2})
    assert oto['a'] == 1
    assert oto.inv[2] == 'b'


def test_one_to_one_unique_across_inputs():
    with pytest.raises(ValueError):
        OneToOne.unique({'a': 2}, b=2)


def test_one_to_one_empty():
    oto = OneToOne()
    assert len(oto) == 0
    assert len(oto.inv) == 0
    oto['x'] = 10
    assert oto['x'] == 10
    assert oto.inv[10] == 'x'


def test_one_to_one_inv_is_populated_when_unique():
    # len check must be == to skip rebuild for unique data
    oto = OneToOne({'a': 1, 'b': 2, 'c': 3})
    assert dict(oto) == {'a': 1, 'b': 2, 'c': 3}
    assert dict(oto.inv) == {1: 'a', 2: 'b', 3: 'c'}


def test_one_to_one_overwrite_both_directions():
    oto = OneToOne({'a': 1, 'b': 2})
    oto.inv[1] = 'c'
    assert oto.get('a') is None
    assert oto['c'] == 1
    assert oto.inv[1] == 'c'
    assert len(oto) == 2


def test_stats_describe_text_format():
    stats = Stats(range(1, 8))
    text = stats.describe(format='text')
    expected = (
        "count:    7\n"
        "mean:     4.0\n"
        "std_dev:  2.0\n"
        "mad:      2.0\n"
        "min:      1\n"
        "0.25:     2.5\n"
        "0.5:      4\n"
        "0.75:     5.5\n"
        "max:      7"
    )
    assert text == expected


def test_stats_describe_ljust_padding():
    # verify labels are ljust to 10 chars (the '%' mutation would change padding)
    stats = Stats(range(1, 8))
    lines = stats.describe(format='text').split('\n')
    for line in lines:
        label, sep, rest = line.partition(':')
        # the "label:" portion is ljust(10)
        prefix_len = len(label) + 1  # +1 for ':'
        # everything up to the value is padded to at least 10
        # find where value starts
        padded = (label + ':').ljust(10)
        assert line.startswith(padded)


def test_stats_describe_dict_format():
    stats = Stats(range(1, 8))
    d = stats.describe(format='dict')
    assert d['count'] == 7
    assert d['mean'] == 4.0
    assert d['std_dev'] == 2.0
    assert d['mad'] == 2.0
    assert d['min'] == 1
    assert d['max'] == 7
    assert d['0.25'] == 2.5
    assert d['0.5'] == 4
    assert d['0.75'] == 5.5


def test_stats_describe_list_format():
    stats = Stats(range(1, 8))
    lst = stats.describe(format='list')
    keys = [k for k, v in lst]
    assert keys == ['count', 'mean', 'std_dev', 'mad', 'min',
                    '0.25', '0.5', '0.75', 'max']


def test_stats_describe_default_is_dict():
    stats = Stats(range(1, 8))
    ret = stats.describe()
    assert isinstance(ret, dict)


def test_stats_describe_invalid_format():
    stats = Stats(range(1, 8))
    with pytest.raises(ValueError):
        stats.describe(format='bogus')


def test_stats_describe_custom_quantiles():
    stats = Stats(range(1, 8))
    d = stats.describe(quantiles=[0.1, 0.9], format='dict')
    assert '0.1' in d
    assert '0.9' in d
    assert '0.25' not in d
