import sys
import unittest
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dataset.tasks import OBJECT_NAMES, task_head
from models.baseline_attention import baseline


class ObjectHeadTest(unittest.TestCase):
    def test_mapping_shared_by_tp_ta_and_separate_for_vqa(self):
        self.assertEqual(len(OBJECT_NAMES), 18)
        for i, name in enumerate(OBJECT_NAMES):
            self.assertEqual(task_head({"condition": "present", "task": name}), i)
            self.assertEqual(task_head({"condition": "absent", "task": name}), i)
        self.assertEqual(task_head({"condition": "vqa", "task": "What is on the table?"}), -1)
        self.assertEqual(task_head({"condition": "present", "task": "potted_plant"}), 13)
        with self.assertRaisesRegex(ValueError, "Unknown TP/TA"):
            task_head({"condition": "present", "task": "not_an_object"})

    def test_mixed_batch_preserves_order_and_only_selected_heads_get_gradients(self):
        # Small convolutions exercise the same routing function without 18 large allocations.
        network = baseline.__new__(baseline)
        nn.Module.__init__(network)
        network.object_name = list(OBJECT_NAMES)
        network.int2object = dict(enumerate(OBJECT_NAMES))
        network.object_sal_layer = nn.ModuleDict({name: nn.Conv2d(2, 2, 1) for name in OBJECT_NAMES})
        network.performance_sal_layer = nn.Conv2d(2, 2, 1)
        features = torch.randn(5, 2, 4, 4, requires_grad=True)
        indices = torch.tensor([2, -1, 0, 2, -1])
        output = network.route_task_features(features, indices)
        with torch.no_grad():
            expected = torch.cat([network.object_sal_layer["car"](features[0:1]),
                network.performance_sal_layer(features[1:2]),
                network.object_sal_layer["bottle"](features[2:3]),
                network.object_sal_layer["car"](features[3:4]),
                network.performance_sal_layer(features[4:5])])
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        for name, layer in network.object_sal_layer.items():
            self.assertEqual(layer.bias.grad is not None, name in ("car", "bottle"))
        self.assertIsNotNone(network.performance_sal_layer.bias.grad)
        torch.testing.assert_close(network.object_sal_layer["car"].bias.grad, torch.full((2,), 32.))
        torch.testing.assert_close(network.performance_sal_layer.bias.grad, torch.full((2,), 32.))

    def test_old_shared_head_checkpoint_is_rejected(self):
        from runtime import restore_config
        with self.assertRaisesRegex(ValueError, "18 TP/TA"):
            restore_config(None, {"manifest": {}}, {})
