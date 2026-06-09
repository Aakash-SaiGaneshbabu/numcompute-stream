import numpy as np
import pytest

from numcompute.io import load_array, save_array, load_csv, save_csv


def test_save_and_load_array(tmp_path):
    arr = np.array([[1.5, 2.5], [3.5, 4.5]])
    path = tmp_path / "array.txt"

    save_array(path, arr)
    loaded = load_array(path)

    assert np.allclose(loaded, arr)


def test_save_and_load_csv(tmp_path):
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    path = tmp_path / "data.csv"

    save_csv(path, arr)
    loaded = load_csv(path)

    assert np.allclose(loaded, arr)


def test_load_array_file_not_found(tmp_path):
    path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        load_array(path)


def test_load_array_with_skiprows(tmp_path):
    path = tmp_path / "with_header.txt"
    path.write_text("header line\n1,2\n3,4\n")

    loaded = load_array(path, skiprows=1)

    assert np.allclose(loaded, np.array([[1, 2], [3, 4]]))