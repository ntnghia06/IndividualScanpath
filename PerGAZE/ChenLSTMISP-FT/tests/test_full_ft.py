import sys
import unittest
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import bootstrap
from adaptation import configure_full_finetuning, select_support

class FullFTTest(unittest.TestCase):
    def test_all_parameters_and_bn_are_trainable_and_update(self):
        torch.manual_seed(3)
        network = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.ReLU(), nn.Linear(4, 2))
        network.requires_grad_(False).eval()
        manifest = {}
        count = configure_full_finetuning(network, manifest)
        self.assertTrue(all(p.requires_grad for p in network.parameters()))
        self.assertTrue(network[1].training)
        self.assertEqual(count, sum(p.numel() for p in network.parameters()))
        self.assertEqual(manifest["adaptation_mode"], "full_model_v1")
        before = network[0].weight.detach().clone()
        optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
        optimizer.zero_grad(); network(torch.randn(4, 4)).square().mean().backward(); optimizer.step()
        self.assertFalse(torch.equal(network[0].weight, before))
        self.assertGreater(network[1].num_batches_tracked.item(), 0)

    def test_support_separates_same_id_across_conditions(self):
        rows=[{"condition":c,"subject":7,"record_id":f"{c}-{i}"} for c in ("present","absent","vqa") for i in range(10)]
        selected, mapping=select_support(rows, 2, 10)
        self.assertEqual(len(selected), 6)
        self.assertEqual(set(mapping), {"tp:7", "ta:7", "air:7"})
        self.assertEqual(mapping, select_support(list(reversed(rows)), 2, 10)[1])

    def test_embedding_only_checkpoint_cannot_resume_full_ft(self):
        from adaptation import require_full_checkpoint
        with self.assertRaisesRegex(ValueError, "embedding-only"):
            require_full_checkpoint({"adaptation_subjects": ["ta:7"]})
        require_full_checkpoint({"adaptation_mode": "full_model_v1"})
