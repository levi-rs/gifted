# import pytest

# from gifted import cli
from gifted.cli import PNG, png, JPG, jpg, GIF, gif
from gifted.cli import OUTPUT_FILE, DEFAULT_DURATION


def test_strings():
    """
    Verify the global strings exist
    """
    assert PNG == 'PNG'
    assert png == 'png'
    assert JPG == 'JPG'
    assert jpg == 'jpg'
    assert GIF == 'GIF'
    assert gif == 'gif'
    assert OUTPUT_FILE == 'output.gif'
    assert DEFAULT_DURATION == 0.2
