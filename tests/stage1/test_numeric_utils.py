import math
import unittest


class NumericUtilsTest(unittest.TestCase):
    def test_finite_or_default_replaces_nan_and_infinity(self):
        from src.stage1.numeric_utils import finite_or_default

        self.assertEqual(finite_or_default(float("nan"), 0.25), 0.25)
        self.assertEqual(finite_or_default(float("inf"), 0.25), 0.25)
        self.assertEqual(finite_or_default(-float("inf"), 0.25), 0.25)
        self.assertTrue(math.isclose(finite_or_default(0.75, 0.25), 0.75))


if __name__ == "__main__":
    unittest.main()
