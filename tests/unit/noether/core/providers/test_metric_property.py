#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import logging
import unittest
from io import StringIO

from noether.core.providers.metric_property import MetricPropertyProvider, Ordinality


class TestMetricPropertyProvider(unittest.TestCase):
    def _register_all_defaults(self):
        """Helper to ensure a clean slate and default patterns are set correctly."""
        MetricPropertyProvider._PATTERNS = []
        MetricPropertyProvider._register_defaults()

    def setUp(self):
        # 1. Reset static state before every test
        self._register_all_defaults()

        # 2. Setup Logging Capture for the named logger
        self.log_stream = StringIO()
        self.handler = logging.StreamHandler(self.log_stream)

        self.provider = MetricPropertyProvider()

        # MUST target the logger used by the provider instance
        self.logger = logging.getLogger(type(self.provider).__name__)

        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.WARNING)
        self.logger.propagate = False

    def tearDown(self):
        self.logger.removeHandler(self.handler)
        self.log_stream.close()

    def test_default_patterns(self):
        """Test default patterns for all three ordinalities and priority logic."""

        # Format: (key, expected_ordinality, expected_higher_is_better_bool)
        test_cases: list[tuple[str, Ordinality, bool]] = [
            # Lower is Better
            ("val/loss/total", Ordinality.LOWER_IS_BETTER, False),
            # Higher is Better
            ("train_accuracy", Ordinality.HIGHER_IS_BETTER, True),
            # Neutral (Priority check: Should match 'loss_weight/*' (Neutral) over '*loss*' (Lower))
            ("loss_weight/decay", Ordinality.NEUTRAL, False),
            # Neutral (Namespace match - highest priority)
            ("optim/lr/encoder", Ordinality.NEUTRAL, False),
        ]

        for key, expected_ordinality, expected_higher_is_better in test_cases:
            with self.subTest(key=key):
                self.assertEqual(MetricPropertyProvider.get_ordinality(key), expected_ordinality)
                self.assertEqual(self.provider.higher_is_better(key), expected_higher_is_better)

    def test_unmatched_key_logs_warning(self):
        """Test that an unmatched key defaults to True and correctly logs a warning."""
        key = "unmatched_custom_metric"

        # 1. Check the return value
        self.assertTrue(self.provider.higher_is_better(key))
        self.assertEqual(MetricPropertyProvider.get_ordinality(key), None)

        # 2. Check the warning log (This is the previously failing assertion)
        log_content = self.log_stream.getvalue()
        self.assertIn(f"Metric '{key}' had no defined pattern", log_content)

    def test_custom_pattern_registration_priority(self):
        """Test that a custom, later-registered pattern takes precedence (FIFO priority)."""

        # Clear defaults to test custom registration strictly
        MetricPropertyProvider._PATTERNS = []

        # 1. Register the SPECIFIC, HIGH-PRIORITY rule first
        # We want val/loss/* to be NEUTRAL (most specific intent)
        MetricPropertyProvider.register_pattern("val/loss/*", Ordinality.NEUTRAL)

        # 2. Register the GENERAL, LOW-PRIORITY rule second
        # Everything else should be LOWER_IS_BETTER
        MetricPropertyProvider.register_pattern("*", Ordinality.LOWER_IS_BETTER)

        # Test 1 (High Priority): Hit 'val/loss/*' -> Should be NEUTRAL
        # This now passes because NEUTRAL is encountered first in the pattern list.
        self.assertEqual(MetricPropertyProvider.get_ordinality("val/loss/epoch"), Ordinality.NEUTRAL)
        self.assertFalse(self.provider.higher_is_better("val/loss/epoch"))

        # Test 2 (Low Priority): Skip 'val/loss/*', Hit '*' -> Should be LOWER_IS_BETTER
        self.assertEqual(MetricPropertyProvider.get_ordinality("train/score"), Ordinality.LOWER_IS_BETTER)
        self.assertFalse(self.provider.higher_is_better("train/score"))
