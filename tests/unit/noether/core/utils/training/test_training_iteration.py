#  Copyright © 2025 Emmi AI GmbH. All rights reserved.

import unittest
from unittest.mock import patch

from noether.core.utils.training.training_iteration import TrainingIteration


class TestCheckpoint(unittest.TestCase):
    def test_add_sub(self):
        self.assertEqual(TrainingIteration(3, 6, 9), TrainingIteration(1, 2, 3) + TrainingIteration(2, 4, 6))
        self.assertEqual(TrainingIteration(1, 2, 3), TrainingIteration(2, 4, 6) - TrainingIteration(1, 2, 3))
        with self.assertRaises(RuntimeError):
            TrainingIteration(2, 4, 6) - TrainingIteration(1, 2)

    def test_is_fully_specified(self):
        self.assertTrue(TrainingIteration(1, 2, 3).is_fully_specified)
        self.assertFalse(TrainingIteration(1, 2).is_fully_specified)
        self.assertFalse(TrainingIteration(1).is_fully_specified)
        self.assertFalse(TrainingIteration().is_fully_specified)

    def test_is_minimally_specified(self):
        self.assertFalse(TrainingIteration(1, 2, 3).is_minimally_specified)
        self.assertFalse(TrainingIteration(1, 2).is_minimally_specified)
        self.assertTrue(TrainingIteration(1).is_minimally_specified)
        self.assertFalse(TrainingIteration().is_minimally_specified)

    def test_get_n_equal_properties(self):
        self.assertEqual(3, TrainingIteration().get_n_equal_properties(TrainingIteration()))
        self.assertEqual(3, TrainingIteration(epoch=1).get_n_equal_properties(TrainingIteration(epoch=1)))
        self.assertEqual(3, TrainingIteration(update=1).get_n_equal_properties(TrainingIteration(update=1)))
        self.assertEqual(3, TrainingIteration(sample=1).get_n_equal_properties(TrainingIteration(sample=1)))
        self.assertEqual(1, TrainingIteration(sample=1).get_n_equal_properties(TrainingIteration(epoch=1)))
        self.assertEqual(2, TrainingIteration(0, 1, 2).get_n_equal_properties(TrainingIteration(0, 1, 4)))
        self.assertEqual(0, TrainingIteration(2, 4, 6).get_n_equal_properties(TrainingIteration(0, 1, 4)))
        self.assertEqual(1, TrainingIteration(2, 4, 6).get_n_equal_properties(TrainingIteration(0, 1, 6)))

    def test_find_checkpoint_string(self):
        self.assertEqual("E5_U342_S123155", TrainingIteration.find_string("someasdfa E5_U342_S123155.optim.th"))
        self.assertEqual("E5_U342_S123155", TrainingIteration.find_string("someasdfa cp=E5_U342_S123155.optim.th"))

    def test_from_checkpoint_string(self):
        actual = TrainingIteration.from_string("E5_U342_S123155")
        self.assertEqual(TrainingIteration(epoch=5, update=342, sample=123155), actual)

    def test_equally_specified(self):
        self.assertTrue(TrainingIteration(epoch=5).has_same_specified_properties(TrainingIteration(epoch=6)))
        self.assertTrue(TrainingIteration(update=5).has_same_specified_properties(TrainingIteration(update=6)))
        self.assertTrue(TrainingIteration(sample=5).has_same_specified_properties(TrainingIteration(sample=6)))
        self.assertFalse(TrainingIteration(epoch=5).has_same_specified_properties(TrainingIteration()))
        self.assertFalse(TrainingIteration(update=5).has_same_specified_properties(TrainingIteration()))
        self.assertFalse(TrainingIteration(sample=5).has_same_specified_properties(TrainingIteration()))
        self.assertFalse(TrainingIteration(epoch=5).has_same_specified_properties(TrainingIteration(update=6)))
        self.assertFalse(TrainingIteration(epoch=5).has_same_specified_properties(TrainingIteration(sample=6)))
        self.assertFalse(TrainingIteration(update=5).has_same_specified_properties(TrainingIteration(epoch=6)))
        self.assertFalse(TrainingIteration(update=5).has_same_specified_properties(TrainingIteration(sample=6)))
        self.assertFalse(TrainingIteration(sample=5).has_same_specified_properties(TrainingIteration(epoch=6)))
        self.assertFalse(TrainingIteration(sample=5).has_same_specified_properties(TrainingIteration(update=6)))
        self.assertFalse(
            TrainingIteration(epoch=5).has_same_specified_properties(TrainingIteration(epoch=6, update=12))
        )
        self.assertFalse(
            TrainingIteration(epoch=5).has_same_specified_properties(TrainingIteration(epoch=6, update=12, sample=78))
        )
        self.assertFalse(
            TrainingIteration(update=5).has_same_specified_properties(TrainingIteration(epoch=6, update=12))
        )
        self.assertFalse(
            TrainingIteration(update=5).has_same_specified_properties(TrainingIteration(epoch=6, update=12, sample=78))
        )
        self.assertFalse(
            TrainingIteration(epoch=5).has_same_specified_properties(TrainingIteration(epoch=6, update=12))
        )
        self.assertFalse(
            TrainingIteration(epoch=5).has_same_specified_properties(TrainingIteration(epoch=6, update=12, sample=78))
        )

    def test_ge(self):
        self.assertFalse(
            TrainingIteration(epoch=5, update=342, sample=1231) >= TrainingIteration(epoch=6, update=344, sample=1232)
        )
        self.assertTrue(
            TrainingIteration(epoch=6, update=344, sample=1232) >= TrainingIteration(epoch=5, update=342, sample=1231)
        )
        self.assertTrue(
            TrainingIteration(epoch=5, update=342, sample=1231) >= TrainingIteration(epoch=5, update=342, sample=1231)
        )
        self.assertTrue(
            TrainingIteration(epoch=5, update=344, sample=1232) >= TrainingIteration(epoch=5, update=342, sample=1231)
        )
        self.assertFalse(
            TrainingIteration(epoch=5, update=342, sample=1231) >= TrainingIteration(epoch=5, update=341, sample=1232)
        )

    def test_to_target_specification(self):
        target = TrainingIteration(epoch=5)
        self.assertEqual(target, TrainingIteration(epoch=5, update=12, sample=123).to_target_specification(target))
        self.assertEqual(
            TrainingIteration(epoch=12),
            TrainingIteration(epoch=12, update=14, sample=163).to_target_specification(target),
        )
        target = TrainingIteration(update=12)
        self.assertEqual(target, TrainingIteration(epoch=5, update=12, sample=123).to_target_specification(target))
        self.assertEqual(
            TrainingIteration(update=17),
            TrainingIteration(epoch=12, update=17, sample=23).to_target_specification(target),
        )
        target = TrainingIteration(sample=123)
        self.assertEqual(target, TrainingIteration(epoch=5, update=12, sample=123).to_target_specification(target))
        self.assertEqual(
            TrainingIteration(sample=147),
            TrainingIteration(epoch=2, update=67, sample=147).to_target_specification(target),
        )

    def test_to_fully_specified_from_filenames(self):
        with patch(
            "os.listdir",
            lambda _: [
                "mae_vit.encoder cp=E1_U2_S4 model.th",
                "mae_vit.encoder cp=E2_U4_S8 model.th",
                "mae_vit.encoder cp=E5_U10_S20 model.th",
                "mae_vit.encoder cp=E1_U2_S4 optim.th",
                "mae_vit.encoder cp=E2_U4_S8 optim.th",
                "mae_vit.encoder cp=E5_U10_S20 optim.th",
                "mae_vit.head cp=E5_U10_S20 queue.th",
                "mae_vit.head cp=E6_U12_S24 queue.th",
            ],
        ):
            self.assertEqual(
                TrainingIteration.to_fully_specified_from_filenames(
                    directory="dummy_folder",
                    training_iteration=TrainingIteration(epoch=1),
                    prefix="mae_vit.encoder cp=",
                    suffix=" model.th",
                ),
                TrainingIteration(1, 2, 4),
            )
            self.assertEqual(
                TrainingIteration.to_fully_specified_from_filenames(
                    directory="dummy_folder",
                    training_iteration=TrainingIteration(update=2),
                    prefix="mae_vit.encoder cp=",
                    suffix=" model.th",
                ),
                TrainingIteration(1, 2, 4),
            )
            self.assertEqual(
                TrainingIteration.to_fully_specified_from_filenames(
                    directory="dummy_folder",
                    training_iteration=TrainingIteration(sample=4),
                    prefix="mae_vit.encoder cp=",
                    suffix=" model.th",
                ),
                TrainingIteration(1, 2, 4),
            )
            self.assertEqual(
                TrainingIteration.to_fully_specified_from_filenames(
                    directory="dummy_folder",
                    training_iteration=TrainingIteration(epoch=1),
                    prefix="mae_vit.encoder cp=",
                    suffix=" optim.th",
                ),
                TrainingIteration(1, 2, 4),
            )
            self.assertRaises(
                FileNotFoundError,
                lambda: TrainingIteration.to_fully_specified_from_filenames(
                    directory="dummy_folder",
                    training_iteration=TrainingIteration(epoch=10),
                    prefix="mae_vit.encoder cp=",
                    suffix=" optim.th",
                ),
            )
            self.assertRaises(
                FileNotFoundError,
                lambda: TrainingIteration.to_fully_specified_from_filenames(
                    directory="dummy_folder",
                    training_iteration=TrainingIteration(epoch=6),
                    prefix="mae_vit.encoder cp=",
                    suffix=" optim.th",
                ),
            )
            self.assertEqual(
                TrainingIteration.to_fully_specified_from_filenames(
                    directory="dummy_folder",
                    training_iteration=TrainingIteration(epoch=6),
                    prefix="mae_vit.head cp=",
                ),
                TrainingIteration(6, 12, 24),
            )
            self.assertEqual(
                TrainingIteration.to_fully_specified_from_filenames(
                    directory="dummy_folder",
                    training_iteration=TrainingIteration(epoch=6),
                ),
                TrainingIteration(6, 12, 24),
            )

    def test_cast_to_dict(self):
        self.assertEqual({}, dict(TrainingIteration()))
        self.assertEqual(dict(epoch=5), dict(TrainingIteration(epoch=5)))
        self.assertEqual(dict(update=5), dict(TrainingIteration(update=5)))
        self.assertEqual(dict(sample=5), dict(TrainingIteration(sample=5)))
        self.assertEqual(dict(epoch=5, update=10), dict(TrainingIteration(epoch=5, update=10)))
        self.assertEqual(dict(epoch=5, sample=20), dict(TrainingIteration(epoch=5, sample=20)))
        self.assertEqual(dict(update=10, sample=20), dict(TrainingIteration(update=10, sample=20)))
        self.assertEqual(dict(epoch=5, update=10, sample=20), dict(TrainingIteration(epoch=5, update=10, sample=20)))
