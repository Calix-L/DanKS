import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from DanRL_retrieval.openguandan_adapter.eval_manifest import load_guandan_paired_split


def _row(index: int, seed: int) -> dict:
    return {
        "assignments": ["A", "B"],
        "deal_seed": seed,
        "game": "guandan",
        "games_per_pair": 2,
        "pair_index": index,
        "protocol_id": "cardks-plan-b-20260722-v1",
        "schema_version": "cardks.ablation.split_seed.v1",
        "split": "validation",
    }


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class EvalManifestTest(unittest.TestCase):
    def test_load_guandan_paired_split_locks_identity(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "split.jsonl"
            _write(path, [_row(0, 11), _row(1, 12)])
            pairs, identity = load_guandan_paired_split(path)
        self.assertEqual(pairs, [(0, 11), (1, 12)])
        self.assertEqual(identity["pairs"], 2)
        self.assertEqual(identity["split"], "validation")
        self.assertEqual(len(identity["sha256"]), 64)

    def test_load_guandan_paired_split_fails_closed(self) -> None:
        cases = [
            ([_row(1, 11)], "contiguous"),
            ([_row(0, 11), _row(1, 11)], "duplicate"),
            ([{**_row(0, 11), "game": "doudizhu"}], "not for guandan"),
            ([{**_row(0, 11), "assignments": ["B", "A"]}], "paired deal"),
        ]
        for rows, message in cases:
            with self.subTest(message=message), TemporaryDirectory() as directory:
                path = Path(directory) / "split.jsonl"
                _write(path, rows)
                with self.assertRaisesRegex(ValueError, message):
                    load_guandan_paired_split(path)


if __name__ == "__main__":
    unittest.main()
