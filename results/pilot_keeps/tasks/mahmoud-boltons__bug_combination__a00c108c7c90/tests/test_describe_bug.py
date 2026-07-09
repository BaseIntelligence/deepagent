from boltons.statsutils import describe, Stats


def test_describe_default_quantiles():
    result = describe(range(7), format='dict')
    assert result['count'] == 7
    assert result['mean'] == 3.0
    # default quantiles [0.25, 0.5, 0.75] should be present
    assert result['0.25'] == 1.5
    assert result['0.5'] == 3
    assert result['0.75'] == 4.5


def test_stats_describe_default_quantiles_list():
    items = Stats(range(1, 8)).describe(format='list')
    labels = [k for k, v in items]
    assert '0.25' in labels
    assert '0.5' in labels
    assert '0.75' in labels
