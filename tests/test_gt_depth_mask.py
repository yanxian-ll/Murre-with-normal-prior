import unittest

import numpy as np

from murre.training_dataset import _normal_valid_from_depth_valid


class GTDepthMaskTest(unittest.TestCase):
    def test_all_valid_keeps_only_pixels_with_full_cross_neighborhood(self):
        valid = np.ones((5, 6), dtype=bool)
        normal_valid = _normal_valid_from_depth_valid(valid)

        expected = np.zeros_like(valid)
        expected[1:-1, 1:-1] = True
        np.testing.assert_array_equal(normal_valid, expected)

    def test_depth_hole_invalidates_center_and_four_neighbors(self):
        valid = np.ones((7, 7), dtype=bool)
        valid[3, 3] = False

        normal_valid = _normal_valid_from_depth_valid(valid)

        self.assertFalse(normal_valid[3, 3])
        self.assertFalse(normal_valid[2, 3])
        self.assertFalse(normal_valid[4, 3])
        self.assertFalse(normal_valid[3, 2])
        self.assertFalse(normal_valid[3, 4])
        self.assertTrue(normal_valid[2, 2])
        self.assertTrue(normal_valid[4, 4])


if __name__ == "__main__":
    unittest.main()
