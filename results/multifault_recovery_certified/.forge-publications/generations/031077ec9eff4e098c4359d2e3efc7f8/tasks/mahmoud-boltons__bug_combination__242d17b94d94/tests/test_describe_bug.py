from boltons.dictutils import OneToOne
from boltons.statsutils import Stats, describe


def test_describe_default_quantiles():
    result = describe(range(7), format='dict')
    assert result['count'] == 7
    assert result['mean'] == 3.0
    assert result['0.25'] == 1.5
    assert result['0.5'] == 3
    assert result['0.75'] == 4.5


def test_stats_describe_default_quantiles_list():
    items = Stats(range(1, 8)).describe(format='list')
    labels = [key for key, _value in items]
    assert '0.25' in labels
    assert '0.5' in labels
    assert '0.75' in labels


def test_one_to_one_initializes_mapping_and_keywords():
    one_to_one = OneToOne({'alpha': 1}, beta=2)
    assert one_to_one == {'alpha': 1, 'beta': 2}
    assert one_to_one.inv == {1: 'alpha', 2: 'beta'}
