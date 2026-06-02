#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import string
from datetime import date

from noether.core.providers.path import PathProvider


def test_generate_run_id_format():
    """Test that generate_run_id returns a string with the expected format: YYYY-MM-DD_xxxxx"""
    run_id = PathProvider.generate_run_id()

    # Should be 16 characters total: 10 (date) + 1 (underscore) + 5 (random chars)
    assert len(run_id) == 16

    # Should start with today's date in YYYY-MM-DD format
    today_str = date.today().strftime("%Y-%m-%d_")
    assert run_id.startswith(today_str)

    # The last 5 characters should be alphanumeric (lowercase letters and digits)
    suffix = run_id[-5:]
    assert suffix.isalnum()
    assert suffix.islower() or suffix.isdigit()


def test_generate_run_id_suffix_characters():
    """Test that the suffix only contains lowercase letters and digits."""
    allowed_chars = set(string.ascii_lowercase + string.digits)

    # Test multiple run_ids to ensure consistency
    for seed in range(10):
        run_id = PathProvider.generate_run_id(seed=seed)
        suffix = run_id[-5:]

        # All characters in suffix should be in allowed set
        for char in suffix:
            assert char in allowed_chars
