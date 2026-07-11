from boltons.funcutils import wraps


def test_wraps_basic_regular_function_preserves_metadata_and_wrapped():
    def source(value):
        '''Return a value through the original callable.'''
        return value * 2

    @wraps(source)
    def wrapped(*args, **kwargs):
        return source(*args, **kwargs)

    assert wrapped(3) == 6
    assert wrapped.__name__ == source.__name__
    assert wrapped.__doc__ == source.__doc__
    assert wrapped.__wrapped__ is source
