#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import unittest

from noether.core.utils.logging import (
    float_to_scientific_notation,
    seconds_to_duration_str,
    short_number_str,
    summarize_indices_list,
)


class TestFormatting(unittest.TestCase):
    def test_short_number_str(self):
        # --- Original tests that still pass ---
        self.assertEqual("123", short_number_str(123, precision=1))
        self.assertEqual("123.0", short_number_str(123.0, precision=1))  # Note: precision=1 adds .0 for floats
        self.assertEqual("123", short_number_str(123, precision=0))
        self.assertEqual("123.12K", short_number_str(123123, precision=2))
        self.assertEqual("123M", short_number_str(123123123, precision=0))
        self.assertEqual("245G", short_number_str(245 * 10**9, precision=0))
        self.assertEqual("1.9K", short_number_str(1920, precision=1))
        self.assertEqual("920", short_number_str(920, precision=1))
        self.assertEqual("0.0", short_number_str(0.0, precision=1))
        self.assertEqual("-920.0", short_number_str(-920.0, precision=1))
        self.assertEqual("1M", short_number_str(999999, precision=0))
        self.assertEqual("1.0M", short_number_str(999999, precision=1))

    def test_float_to_scientific_notation(self):
        # The function now respects max_precision, so "0" becomes "0.00e0"
        self.assertEqual("0.00e00", float_to_scientific_notation(0, max_precision=2, remove_plus=True))
        # "3" becomes "3.00e00"
        self.assertEqual("3.00e00", float_to_scientific_notation(3, max_precision=2, remove_plus=True))
        # "2000" becomes "2e3"
        self.assertEqual("2e03", float_to_scientific_notation(2000, max_precision=2, remove_plus=True))
        # "2000" without remove_plus becomes "2e+3"
        self.assertEqual("2e+03", float_to_scientific_notation(2000, max_precision=2, remove_plus=False))
        self.assertEqual("5e123", float_to_scientific_notation(5e123, max_precision=2, remove_plus=True))
        # 0.003251241 rounds to 3.25e-3
        self.assertEqual("3.25e-03", float_to_scientific_notation(0.003251241, max_precision=2, remove_plus=True))
        # 0.00320241 rounds to 3.20e-3
        self.assertEqual("3.20e-03", float_to_scientific_notation(0.00320241, max_precision=2, remove_plus=True))

    def test_summarize_indices_list(self):
        self.assertEqual([], summarize_indices_list([]))
        self.assertEqual(["1"], summarize_indices_list([1]))
        self.assertEqual(["0-3"], summarize_indices_list([0, 1, 2, 3]))
        self.assertEqual(["0-3", "6-8"], summarize_indices_list([0, 1, 2, 3, 6, 7, 8]))
        self.assertEqual(["0-3", "6-8", "10"], summarize_indices_list([0, 1, 2, 3, 6, 7, 8, 10]))
        self.assertEqual(["0-3", "6-8", "10", "13-14"], summarize_indices_list([0, 1, 2, 3, 6, 7, 8, 10, 13, 14]))

    def test_seconds_to_duration_str(self):
        self.assertEqual("00:00:00.00", seconds_to_duration_str(0))
        self.assertEqual("00:00:00.10", seconds_to_duration_str(0.1))
        self.assertEqual("00:00:00.11", seconds_to_duration_str(0.11))
        self.assertEqual("00:01:05.00", seconds_to_duration_str(65))
        self.assertEqual("01:01:05.00", seconds_to_duration_str(3665))
        self.assertEqual("1-01:01:05.00", seconds_to_duration_str(90065))
        self.assertEqual("10-01:01:05.00", seconds_to_duration_str(867665))
