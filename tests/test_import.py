import pytest


def test_import():
    with pytest.raises(ImportError):
        import foobar  # noqa
