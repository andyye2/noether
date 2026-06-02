#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import unittest

from noether.core.distributed.utils import parse_devices


class TestRun(unittest.TestCase):
    def test_get_devices(self):
        for accelerator in ["cpu", "gpu"]:
            self.assertEqual((1, ["0"]), parse_devices(accelerator, "0"))
            self.assertEqual((1, ["1"]), parse_devices(accelerator, "1"))
            self.assertEqual((2, ["0", "1"]), parse_devices(accelerator, "0,1"))
            self.assertEqual((3, ["1", "2", "3"]), parse_devices(accelerator, "1,2,3"))
            self.assertEqual((3, ["3", "2", "1"]), parse_devices(accelerator, "3,2,1"))
            self.assertEqual((3, ["1", "2", "3"]), parse_devices(accelerator, "1,2,3"))
