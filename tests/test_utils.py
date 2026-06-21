from pathlib import Path

from src.utils import is_valid_file


def test_is_valid_file_returns_false_for_missing_file(tmp_path):
    missing_path = tmp_path / "missing.pkl"

    assert is_valid_file(missing_path) is False


def test_is_valid_file_returns_false_for_empty_file(tmp_path):
    empty_path = tmp_path / "empty.pkl"
    empty_path.touch()

    assert is_valid_file(empty_path) is False


def test_is_valid_file_returns_true_for_non_empty_file(tmp_path):
    model_path = tmp_path / "model.pkl"
    model_path.write_bytes(b"not empty")

    assert is_valid_file(model_path) is True


def test_is_valid_file_returns_false_for_directory(tmp_path):
    directory = tmp_path / "models"
    directory.mkdir()

    assert is_valid_file(Path(directory)) is False
