import unittest

from DanRL_retrieval.retrieval.context import RetrievalContext
from DanRL_retrieval.retrieval.diagnostics import (
    action_structure_diagnostic,
    card_opportunity_map,
    structure_opportunities,
)
from DanRL_retrieval.retrieval.models import ActionCandidate


class DiagnosticsTest(unittest.TestCase):
    def test_card_has_multiple_structure_identities(self):
        hand = ["S3", "S4", "S5", "S6", "S7", "H3", "C3", "D3"]
        opportunities = structure_opportunities(hand, RetrievalContext())
        by_card = card_opportunity_map(opportunities)
        s3_kinds = {group.kind for group in by_card["S3"]}

        self.assertIn("StraightFlush", s3_kinds)
        self.assertIn("Bomb", s3_kinds)
        self.assertNotIn("Straight", s3_kinds)

    def test_bomb_action_reports_broken_straight_flush(self):
        hand = ["S3", "S4", "S5", "S6", "S7", "H3", "C3", "D3"]
        opportunities = structure_opportunities(hand, RetrievalContext())
        action = ActionCandidate(0, "Bomb", ("S3", "H3", "C3", "D3"), "3")
        diagnostic = action_structure_diagnostic(action, opportunities)
        broken = {(group.kind, group.cards) for group in diagnostic.broken_groups}
        complete = {(group.kind, group.cards) for group in diagnostic.complete_groups}

        self.assertIn(("StraightFlush", ("S3", "S4", "S5", "S6", "S7")), broken)
        self.assertIn(("Bomb", ("C3", "D3", "H3", "S3")), complete)

    def test_wild_pair_complete_respects_action_rank(self):
        hand = ["H8", "H8"]
        opportunities = structure_opportunities(hand, RetrievalContext(cur_rank="8"))
        action = ActionCandidate(0, "Pair", ("H8", "H8"), "8")
        diagnostic = action_structure_diagnostic(action, opportunities)
        complete_ranks = {group.rank for group in diagnostic.complete_groups}

        self.assertEqual(complete_ranks, {"8"})


if __name__ == "__main__":
    unittest.main()
