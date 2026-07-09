from boltons.statsutils import describe
from boltons.formatutils import get_format_args


def test_describe_default_quantiles():
    result = describe(range(7))
    assert result['count'] == 7
    assert result['min'] == 0
    assert result['max'] == 6
    assert result['0.5'] == 3


def test_get_format_args_simple():
    fargs, fkwargs = get_format_args("{noun} is {1:d} years old{punct}")
    assert fargs == [(1, int)]
    assert fkwargs == [('noun', str), ('punct', str)]
