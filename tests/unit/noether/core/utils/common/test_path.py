#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from pathlib import Path
from types import SimpleNamespace

import pytest

from noether.core.utils.common.path import select_with_path, validate_path


def test_validate_path_exists_must_pass(tmp_path):
    """Test exists='must' with an existing file."""
    test_file = tmp_path / "test_file.txt"
    test_file.touch()

    result = validate_path(test_file, exists="must")
    assert isinstance(result, Path)
    assert result == test_file


def test_validate_path_exists_must_fail(tmp_path):
    """Test exists='must' with a missing file raises FileNotFoundError."""
    missing_file = tmp_path / "ghost.txt"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        validate_path(missing_file, exists="must")


def test_validate_path_exists_must_not_pass(tmp_path):
    """Test exists='must_not' with a missing file passes."""
    missing_file = tmp_path / "new_file.txt"

    result = validate_path(missing_file, exists="must_not")
    assert result == missing_file


def test_validate_path_exists_must_not_fail(tmp_path):
    """Test exists='must_not' with an existing file raises FileExistsError."""
    existing_file = tmp_path / "here.txt"
    existing_file.touch()

    with pytest.raises(FileExistsError, match="already exists"):
        validate_path(existing_file, exists="must_not")


def test_validate_path_exists_any(tmp_path):
    """Test exists='any' allows both existing and missing files."""
    # 1. Existing
    f1 = tmp_path / "real.txt"
    f1.touch()
    assert validate_path(f1, exists="any") == f1

    # 2. Missing
    f2 = tmp_path / "fake.txt"
    assert validate_path(f2, exists="any") == f2


def test_validate_path_suffix_check(tmp_path):
    """Test suffix validation."""
    # Note: We pass exists="any" so we don't have to create files for simple string checks
    path = tmp_path / "image.png"

    # Correct suffix
    assert validate_path(path, suffix=".png", exists="any") == path

    # Incorrect suffix
    with pytest.raises(ValueError, match="doesn't end with '.jpg'"):
        validate_path(path, suffix=".jpg", exists="any")


def test_validate_path_mkdir(tmp_path):
    """Test that mkdir=True creates directories recursively."""
    target_dir = tmp_path / "parent" / "child"

    # Ensure it doesn't exist yet
    assert not target_dir.exists()

    result = validate_path(target_dir, mkdir=True, exists="must")

    assert result.exists()
    assert result.is_dir()


def test_validate_path_string_input(tmp_path):
    """Test that string inputs are correctly converted to Path objects."""
    path_str = str(tmp_path / "test.txt")

    # Create it so exists="must" works
    Path(path_str).touch()

    result = validate_path(path_str)
    assert isinstance(result, Path)
    assert result.as_posix() == Path(path_str).as_posix()


class MockObject:
    def __init__(self):
        self.value = 123
        self.nested = SimpleNamespace(inner="hello")


@pytest.fixture
def complex_data():
    return {"a": 10, "b": {"c": 20, "d": {"e": 30}}, "list": [100, 200, {"nested_in_list": 300}], "obj": MockObject()}


def test_select_with_path_dict(complex_data):
    """Test retrieving values from nested dictionaries."""
    assert select_with_path(complex_data, "a") == 10
    assert select_with_path(complex_data, "b.c") == 20
    assert select_with_path(complex_data, "b.d.e") == 30


def test_select_with_path_list(complex_data):
    """
    Test retrieving values from lists using integer indices.
    Note: The implementation uses split('.') so the path is "list.0", not "list[0]"
    """
    assert select_with_path(complex_data, "list.0") == 100
    assert select_with_path(complex_data, "list.1") == 200


def test_select_with_path_mixed(complex_data):
    """Test traversing a mix of lists and dicts."""
    # access complex_data['list'][2]['nested_in_list']
    assert select_with_path(complex_data, "list.2.nested_in_list") == 300


def test_select_with_path_object_attributes(complex_data):
    """Test traversing object attributes (getattr)."""
    # access complex_data['obj'].value
    assert select_with_path(complex_data, "obj.value") == 123
    # access complex_data['obj'].nested.inner
    assert select_with_path(complex_data, "obj.nested.inner") == "hello"


def test_select_with_path_empty_or_none(complex_data):
    """Test that empty path returns the original object."""
    assert select_with_path(complex_data, None) == complex_data
    assert select_with_path(complex_data, "") == complex_data


def test_select_with_path_missing_keys(complex_data):
    """Test that standard Python exceptions are raised for missing keys/indices."""
    with pytest.raises(KeyError):
        select_with_path(complex_data, "b.z")

    with pytest.raises(IndexError):
        select_with_path(complex_data, "list.99")

    with pytest.raises(AttributeError):
        select_with_path(complex_data, "obj.missing_attr")
