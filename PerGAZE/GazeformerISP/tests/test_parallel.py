import sys
import importlib
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from parallel import resolve_devices, wrap_model, model_state_dict, load_model_state, restore_cuda_rng


class ParallelTest(unittest.TestCase):
    def test_select_two_visible_gpus(self):
        device, ids = resolve_devices("cuda:0", [0, 1], 2)
        self.assertEqual(str(device), "cuda:0")
        self.assertEqual(ids, [0, 1])
        self.assertEqual(resolve_devices("cuda:0", None, 2)[1], [0, 1])

    def test_reject_missing_or_duplicate_gpu(self):
        for ids in ([0, 1], [0, 0]):
            with self.assertRaisesRegex(ValueError, "Invalid"):
                resolve_devices("cuda:0", ids, 1)
        with self.assertRaises(ValueError):
            resolve_devices("cpu", [0, 1], 0)

    def test_primary_gpu_order(self):
        self.assertEqual(resolve_devices("cuda:1", None, 2)[1], [1, 0])
        with self.assertRaisesRegex(ValueError, "primary"):
            resolve_devices("cuda:0", [1, 0], 2)

    def test_cpu_and_single_gpu_fallback(self):
        self.assertEqual(resolve_devices("cpu", None, 0)[1], [])
        network = torch.nn.Linear(3, 2)
        self.assertIs(wrap_model(network, torch.device("cpu"), []), network)
        with patch.object(network, "to", return_value=network), patch("torch.nn.DataParallel") as wrapper:
            wrap_model(network, torch.device("cuda:0"), [0, 1])
            wrapper.assert_called_once_with(network, device_ids=[0, 1], output_device=0)

    def test_checkpoint_works_with_wrapped_and_unwrapped_models(self):
        source = torch.nn.Linear(3, 2)
        parallel_module = importlib.import_module("torch.nn.parallel.data_parallel")
        with patch.object(parallel_module, "_get_available_device_type", return_value=None):
            wrapped = torch.nn.DataParallel(source)
        state = model_state_dict(wrapped)
        self.assertEqual(set(state), {"weight", "bias"})
        target = torch.nn.Linear(3, 2)
        load_model_state(target, state)
        torch.testing.assert_close(source.weight, target.weight)
        with patch.object(parallel_module, "_get_available_device_type", return_value=None):
            wrapped_target = torch.nn.DataParallel(target)
        load_model_state(wrapped_target, {"module." + key: value for key, value in state.items()})
        torch.testing.assert_close(source.bias, target.bias)

    def test_resume_rng_on_fewer_gpus(self):
        with patch("torch.cuda.device_count", return_value=1), patch("torch.cuda.set_rng_state") as setter:
            states = [torch.zeros(4, dtype=torch.uint8), torch.ones(4, dtype=torch.uint8)]
            restore_cuda_rng(states)
            self.assertEqual(setter.call_count, 1)


if __name__ == "__main__":
    unittest.main()
