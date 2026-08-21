from __future__ import annotations

import unittest

import torch

from DanRL_retrieval.training.accelerator import (
    cpu_state_dict,
    initialize_device,
    is_accelerator,
    resolve_device,
)


class AcceleratorTest(unittest.TestCase):
    def test_explicit_cpu_does_not_require_an_accelerator_backend(self) -> None:
        device = initialize_device("cpu")
        self.assertEqual(device.type, "cpu")
        self.assertFalse(is_accelerator(device))

    def test_invalid_device_is_rejected_by_torch(self) -> None:
        with self.assertRaises((RuntimeError, ValueError)):
            resolve_device("not-a-device")

    def test_cpu_state_dict_is_portable_and_is_a_stable_snapshot(self) -> None:
        model = torch.nn.Linear(3, 2)
        snapshot = cpu_state_dict(model)
        self.assertTrue(all(value.device.type == "cpu" for value in snapshot.values()))
        before = snapshot["weight"].clone()
        with torch.no_grad():
            model.weight.add_(1.0)
        self.assertTrue(torch.equal(snapshot["weight"], before))


if __name__ == "__main__":
    unittest.main()
