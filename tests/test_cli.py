import pytest
from unittest.mock import MagicMock, patch

# from gifted import cli
from gifted.cli import main
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


def test_main():
    """
    Verify functionality of main()
    """
    args_mock = MagicMock()
    args_mock.directory = "/thisdoesnotexist"
    get_args_mock = MagicMock()
    get_args_mock.return_value = args_mock

    with patch('gifted.cli.get_args', get_args_mock):
        with pytest.raises(ValueError) as excstr:
            main()
        assert "/thisdoesnotexist" in str(excstr.value)
