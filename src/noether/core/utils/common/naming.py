#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import re


def pascal_to_snake(pascal_case: str) -> str:
    """Convert pascal/camel to snake case using Regex.

    Handles acronyms and numbers correctly.
    Example:
        XMLParser -> xml_parser
        HTTPClient -> http_client
        V2Model -> v2_model
    """
    if not pascal_case:
        return ""

    # --- Handle acronyms followed by a capitalized word (e.g., XMLParser -> XML_Parser):
    # We look for a character followed by Upper then Lower.
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", pascal_case)

    # --- Handle lower/digit followed by Upper (e.g., camelCase -> camel_case):
    s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)

    return s2.lower()


def lower_type_name(obj: object) -> str:
    """Return the type name of an object in lowercase."""
    return type(obj).__name__.lower()


def snake_type_name(obj: object) -> str:
    """Return the type name of an object in snake case."""
    # convert a type name to snake case
    # preferably use module name as class names are often custom names (e.g. KoLeoLoss)
    # e.g. KoLeoLoss would be converted to ko_leo_loss but if the module is called koleo_loss it will be preferred
    snake = pascal_to_snake(type(obj).__name__)
    module = type(obj).__module__.split(".")[-1]
    if snake.replace("_", "") == module.replace("_", ""):
        return module
    return snake
