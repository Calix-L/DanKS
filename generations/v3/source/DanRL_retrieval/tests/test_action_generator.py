import unittest

from DanRL_retrieval.retrieval.action_generator import PLMActionGenerator
from DanRL_retrieval.retrieval.context import RetrievalContext


class PLMActionGeneratorTest(unittest.TestCase):
    def test_generates_follow_actions_with_pass(self):
        ctx = RetrievalContext(cur_rank="T", current_kind="Single", current_rank="9", current_size=1)
        actions = PLMActionGenerator().generate(["S8", "SK", "BJ"], ctx)
        sigs = {(action.kind, action.rank, action.cards) for action in actions}

        self.assertIn(("PASS", "PASS", ()), sigs)
        self.assertIn(("Single", "K", ("SK",)), sigs)
        self.assertIn(("Single", "BJ", ("BJ",)), sigs)
        self.assertNotIn(("Single", "8", ("S8",)), sigs)

    def test_generates_bomb_over_single(self):
        ctx = RetrievalContext(cur_rank="T", current_kind="Single", current_rank="A", current_size=1)
        actions = PLMActionGenerator().generate(["S3", "H3", "C3", "D3"], ctx)
        self.assertTrue(any(action.kind == "Bomb" and action.rank == "3" for action in actions))

    def test_generates_all_large_bomb_action_sizes(self):
        natural = PLMActionGenerator().generate(
            ["S9", "S9", "H9", "H9", "C9", "C9", "D9", "D9"],
            RetrievalContext(cur_rank="A", current_kind="Lead"),
        )
        self.assertEqual(
            {action.size for action in natural if action.kind == "Bomb" and action.rank == "9"},
            {4, 5, 6, 7, 8},
        )

        wild = PLMActionGenerator().generate(
            ["S9", "H9", "C9", "D9", "H2", "H2"],
            RetrievalContext(cur_rank="2", current_kind="Lead"),
        )
        wild_sizes = {action.size for action in wild if action.kind == "Bomb" and action.rank == "9"}
        self.assertIn(5, wild_sizes)
        self.assertIn(6, wild_sizes)


if __name__ == "__main__":
    unittest.main()
