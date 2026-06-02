#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

from unittest.mock import MagicMock

import pytest

from noether.core.utils.common import lower_type_name, pascal_to_snake, snake_type_name


@pytest.mark.parametrize(
    ("input_str", "expected"),
    [
        # --- Standard Cases ---
        ("PascalCase", "pascal_case"),
        ("MyFunctionName", "my_function_name"),
        ("camelCase", "camel_case"),
        ("Simple", "simple"),
        ("", ""),
        # --- Digit Handling ---
        ("Model123", "model123"),
        ("V2Model", "v2_model"),
        # --- Acronym Handling ---
        ("ID", "id"),
        ("JSON", "json"),
        ("HTTPClient", "http_client"),
        ("XMLParser", "xml_parser"),
        ("SSLLibrary", "ssl_library"),
    ],
)
def test_pascal_to_snake(input_str, expected):
    assert pascal_to_snake(input_str) == expected


class TestClass:
    pass


def test_lower_type_name_custom_class():
    obj = TestClass()
    assert lower_type_name(obj) == "testclass"


def test_lower_type_name_builtin():
    obj = "I am a string"
    assert lower_type_name(obj) == "str"
    obj_int = 123
    assert lower_type_name(obj_int) == "int"


def test_snake_type_name_no_module_match():
    """
    Case 1: The module name does NOT resemble the class name.
    Expected: Returns the class name converted to snake_case.
    """
    obj = MagicMock()
    obj.__class__.__name__ = "SuperMetric"
    obj.__class__.__module__ = "my_library.metrics"

    # "super_metric" != "metrics", so it returns the snake case name
    assert snake_type_name(obj) == "super_metric"


def test_snake_type_name_with_module_match():
    """
    Case 2: The module name resembles the class name (ignoring underscores).
    Expected: Returns the module name (preferred).
    """
    obj = MagicMock()
    obj.__class__.__name__ = "KoLeoLoss"
    # Logic: snake("KoLeoLoss") -> "ko_leo_loss"
    # Stripped: "koleoloss"

    # If module is "koleo_loss", stripped it is "koleoloss" -> Match!
    obj.__class__.__module__ = "custom_lib.losses.koleo_loss"

    # Should prefer the module name over the auto-generated snake case:
    assert snake_type_name(obj) == "koleo_loss"


def test_snake_type_name_exact_match():
    """
    Case 3: The snake case name is exactly the same as the module name.
    """
    obj = MagicMock()
    obj.__class__.__name__ = "SimpleThing"
    obj.__class__.__module__ = "simple_thing"

    # snake("SimpleThing") -> "simple_thing"
    # module -> "simple_thing"
    # They match, returns module (which is identical to snake)
    assert snake_type_name(obj) == "simple_thing"
