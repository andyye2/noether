#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import os

from aistore.sdk.obj.object import Object  # type: ignore[import-untyped]


def group_data_by_paths(samples: list[Object]) -> dict[str, dict[str, Object]]:
    """Groups a list of all aistore objects by their path such that those in the same folder belong together.
    Args:
        samples: List of all aistore objects of type 'Object'.
    Returns:
        A map from paths to grouped data, where grouped data is represented by a dictionary, mapping the filename
        to the respective aistore object.
    """
    path_to_samples_map: dict[str, dict[str, Object]] = {}
    for sample in samples:
        base_path = os.path.dirname(sample.name)
        filename = os.path.basename(sample.name)
        if base_path not in path_to_samples_map:
            path_to_samples_map[base_path] = {}
        assert filename not in path_to_samples_map[base_path], f"Duplicate filename '{filename}' in path '{base_path}'"
        path_to_samples_map[base_path][filename] = sample
    return path_to_samples_map
