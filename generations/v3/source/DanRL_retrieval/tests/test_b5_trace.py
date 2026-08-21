import os
import unittest
from unittest.mock import patch

from DanRL_retrieval.retrieval.b5_trace import B5TraceRanker


class B5TraceRankerTest(unittest.TestCase):
    def test_offline_trace_preserves_production_topk_and_stage_containment(self):
        hand = ["S3", "H3", "C3", "D3", "S4", "H4", "S5", "S6", "S7", "BJ"]
        actions = [
            {"index": 10, "kind": "Single", "rank": "3", "cards": ["S3"]},
            {"index": 11, "kind": "Single", "rank": "BJ", "cards": ["BJ"]},
            {"index": 12, "kind": "Pair", "rank": "4", "cards": ["S4", "H4"]},
            {"index": 13, "kind": "PASS", "rank": "PASS", "cards": []},
        ]
        ctx = {
            "my_seat": 0,
            "curRank": "A",
            "public_counts": [10, 8, 12, 7],
            "current_kind": "Single",
            "current_rank": "2",
            "known_hand_cards": {"0": hand},
        }
        reference_ranker = B5TraceRanker(max_partitions=None)
        reference = reference_ranker.rank(hand, actions, ctx, top_k=3, approximate_top_k=True)
        traced_ranker = B5TraceRanker(max_partitions=None)
        with patch.dict(os.environ, {"DANRL_APPROX_ACTION_LIMIT": "2"}, clear=False):
            traced, trace = traced_ranker.rank_with_b5_trace(
                hand, actions, ctx, top_k=3, approximate_top_k=True
            )
            self.assertEqual(os.environ["DANRL_APPROX_ACTION_LIMIT"], "2")
        self.assertEqual([row.action.index for row in traced], [row.action.index for row in reference])
        self.assertEqual(trace.legal_action_indices, (10, 11, 12, 13))
        self.assertEqual(set(trace.prefilter_action_indices), {10, 11, 12, 13})
        self.assertEqual(set(trace.cover_discovered_action_indices), {10, 11, 12, 13})
        self.assertEqual(trace.top_k_action_indices, tuple(row.action.index for row in traced))

    def test_shadow_cover_contract_preserves_topk_and_records_score_gaps(self):
        hand = ["S3", "H3", "C3", "D3", "S4", "H4", "S5", "S6", "S7", "BJ"]
        actions = [
            {"index": 10, "kind": "Single", "rank": "3", "cards": ["S3"]},
            {"index": 11, "kind": "Single", "rank": "BJ", "cards": ["BJ"]},
            {"index": 12, "kind": "Pair", "rank": "4", "cards": ["S4", "H4"]},
            {"index": 13, "kind": "PASS", "rank": "PASS", "cards": []},
        ]
        ctx = {
            "my_seat": 0,
            "curRank": "A",
            "public_counts": [10, 8, 12, 7],
            "current_kind": "Single",
            "current_rank": "2",
            "known_hand_cards": {"0": hand},
        }
        ranker = B5TraceRanker(
            max_partitions=1,
            shadow_oracle_budget=16,
            cover_score_epsilon=0.0,
        )
        _, trace = ranker.rank_with_b5_trace(
            hand, actions, ctx, top_k=3, approximate_top_k=True,
        )
        self.assertEqual(trace.shadow_oracle_budget, 16)
        self.assertEqual(trace.cover_score_epsilon, 0.0)
        self.assertEqual({item[0] for item in trace.cover_score_gaps}, {10, 11, 12, 13})
        self.assertLessEqual(
            set(trace.top_k_action_indices),
            set(trace.cover_discovered_action_indices),
        )


if __name__ == "__main__":
    unittest.main()
