import unittest

from DanRL_retrieval.retrieval.context import RetrievalContext
from DanRL_retrieval.retrieval.pressure import pressure


class PressureTest(unittest.TestCase):
    def test_single_pressure_high_for_next_opponent_one_card(self):
        ctx = RetrievalContext(my_seat=0, public_counts=(11, 1, 10, 5))
        self.assertGreater(pressure(ctx, "Single"), 0.9)
        self.assertLess(pressure(ctx, "Pair"), pressure(ctx, "Single"))

    def test_pair_pressure_high_for_two_card_opponent(self):
        ctx = RetrievalContext(my_seat=0, public_counts=(11, 2, 10, 5))
        self.assertGreater(pressure(ctx, "Pair"), 0.9)
        self.assertLess(pressure(ctx, "Single"), pressure(ctx, "Pair"))

    def test_opening_pressure_low(self):
        ctx = RetrievalContext(my_seat=0, public_counts=(27, 27, 27, 27))
        self.assertLess(pressure(ctx, "Single"), 0.01)


if __name__ == "__main__":
    unittest.main()
