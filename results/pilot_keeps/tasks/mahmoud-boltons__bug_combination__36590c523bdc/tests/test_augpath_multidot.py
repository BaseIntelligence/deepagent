from boltons.pathutils import augpath


def test_augpath_multidot_suffix():
    # With multidot=True, everything after the first dot is the extension.
    # foo.tar.gz -> base='foo', ext='.tar.gz'; adding suffix keeps the ext.
    assert augpath('foo.tar.gz', suffix='_new', multidot=True) == 'foo_new.tar.gz'
