import unittest

from DanRL_retrieval.training.schema import derive_uniform_action_seed


class UniformActionSeedTest(unittest.TestCase):
    def test_action_replicates_are_distinct_and_reproducible(self):
        values = {
            seed: derive_uniform_action_seed(seed, 5106072301, 0)
            for seed in (2026, 2027, 2028)
        }
        self.assertEqual(len(set(values.values())), 3)
        self.assertEqual(values[2026], derive_uniform_action_seed(2026, 5106072301, 0))

    def test_deal_and_side_are_part_of_the_seed(self):
        reference = derive_uniform_action_seed(2026, 5106072301, 0)
        self.assertNotEqual(reference, derive_uniform_action_seed(2026, 5106072302, 0))
        self.assertNotEqual(reference, derive_uniform_action_seed(2026, 5106072301, 1))

    def test_invalid_values_fail_closed(self):
        for args in ((True, 1, 0), (1, True, 0), (1, 1, True), (-1, 1, 0), (1, -1, 0), (1, 1, 2)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                derive_uniform_action_seed(*args)


if __name__ == "__main__":
    unittest.main()
