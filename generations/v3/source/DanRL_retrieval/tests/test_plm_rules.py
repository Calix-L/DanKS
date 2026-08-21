import unittest

from DanRL_retrieval.retrieval.models import ActionCandidate
from DanRL_retrieval.retrieval.cards import rank_strength
from DanRL_retrieval.retrieval.plm_rules import POINTS, RANK_TO_VALUE, can_follow_action, follow_rank, plain_points


class PLMRulesTest(unittest.TestCase):
    def test_plm_tile_values_keep_two_below_three_and_four(self):
        self.assertLess(RANK_TO_VALUE["A"], RANK_TO_VALUE["2"])
        self.assertLess(RANK_TO_VALUE["2"], RANK_TO_VALUE["3"])
        self.assertLess(RANK_TO_VALUE["2"], RANK_TO_VALUE["4"])

    def test_plain_points_keep_plain_two_below_a(self):
        self.assertEqual(POINTS["2"], plain_points("2"))
        self.assertLess(plain_points("2"), plain_points("3"))
        self.assertLess(plain_points("2"), plain_points("4"))
        self.assertLess(plain_points("2"), plain_points("A"))

    def test_follow_rank_keeps_plain_two_low_unless_level(self):
        self.assertTrue(follow_rank("T", "2", "3"))
        self.assertTrue(follow_rank("T", "2", "A"))
        self.assertFalse(follow_rank("T", "A", "2"))
        self.assertTrue(follow_rank("2", "A", "2"))
        self.assertFalse(follow_rank("2", "2", "A"))

    def test_rank_strength_keeps_plain_two_low_and_level_two_high(self):
        self.assertLess(rank_strength("2", "T"), rank_strength("3", "T"))
        self.assertLess(rank_strength("2", "T"), rank_strength("A", "T"))
        self.assertGreater(rank_strength("2", "2"), rank_strength("A", "2"))
        self.assertLess(rank_strength("2", "2"), rank_strength("BJ", "2"))

    def test_level_rank_above_a_below_jokers(self):
        self.assertTrue(follow_rank("T", "A", "T"))
        self.assertFalse(follow_rank("T", "T", "A"))
        self.assertTrue(follow_rank("T", "T", "BJ"))

    def test_bomb_order_matches_plm_tiers(self):
        four = ActionCandidate(0, "Bomb", ("S9", "H9", "C9", "D9"), "9")
        five = ActionCandidate(1, "Bomb", ("S3", "H3", "C3", "D3", "S3"), "3")
        flush = ActionCandidate(2, "StraightFlush", ("S3", "S4", "S5", "S6", "S7"), "7")
        six = ActionCandidate(3, "Bomb", ("S4", "H4", "C4", "D4", "S4", "H4"), "4")
        kings = ActionCandidate(4, "FourKings", ("BJ", "BJ", "RJ", "RJ"), "RJ")

        self.assertTrue(can_follow_action(five, four, "T"))
        self.assertTrue(can_follow_action(flush, five, "T"))
        self.assertTrue(can_follow_action(six, flush, "T"))
        self.assertTrue(can_follow_action(kings, six, "T"))
        self.assertFalse(can_follow_action(flush, six, "T"))

    def test_sequence_compare_ignores_level(self):
        low = ActionCandidate(0, "Straight", ("S9", "ST", "SJ", "SQ", "SK"), "K")
        level_a = ActionCandidate(1, "Straight", ("ST", "SJ", "SQ", "SK", "SA"), "A")
        self.assertTrue(can_follow_action(level_a, low, "A"))


if __name__ == "__main__":
    unittest.main()
