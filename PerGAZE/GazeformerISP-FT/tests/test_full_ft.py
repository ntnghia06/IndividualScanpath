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

    def test_backbone_gradients_flow_and_train_override_keeps_bn_active(self):
        from full_model import GazeformerFullFT
        class TinyFT(GazeformerFullFT):
            def __init__(self):
                nn.Module.__init__(self)
                self.backbone = nn.Sequential(nn.Conv2d(3, 2048, 1), nn.BatchNorm2d(2048))
                self.head = nn.Linear(2048, 1)
            def forward_features(self, features, *args):
                return self.head(features.mean(1))
        model=TinyFT()
        configure_full_finetuning(model,{})
        before=model.backbone[0].weight.detach().clone()
        optimizer=torch.optim.Adam(model.parameters(),lr=1e-4)
        model(torch.randn(2,3,24,32),None,None,None,None).square().mean().backward()
        self.assertIsNotNone(model.backbone[0].weight.grad)
        self.assertTrue(torch.isfinite(model.backbone[0].weight.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(before,model.backbone[0].weight))
        self.assertTrue(model.backbone[1].training)
        model.eval(); self.assertFalse(model.backbone[1].training)

    def test_frozen_coco_bn_is_converted_to_trainable_bn(self):
        from full_model import unfreeze_batch_norm
        from torchvision.ops.misc import FrozenBatchNorm2d
        model=nn.Sequential(FrozenBatchNorm2d(3))
        model[0].running_mean.copy_(torch.tensor([1.,2.,3.]))
        unfreeze_batch_norm(model)
        self.assertIsInstance(model[0],nn.BatchNorm2d)
        self.assertTrue(model[0].weight.requires_grad)
        torch.testing.assert_close(model[0].running_mean,torch.tensor([1.,2.,3.]))
        model.train(); model(torch.randn(2,3,4,4))
        self.assertEqual(model[0].num_batches_tracked.item(),1)

    def test_embedding_only_checkpoint_cannot_resume_full_ft(self):
        from adaptation import require_full_checkpoint
        with self.assertRaisesRegex(ValueError, "embedding-only"):
            require_full_checkpoint({"adaptation_subjects": ["ta:7"]})
        require_full_checkpoint({"adaptation_mode": "full_model_v1"})
