import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import bootstrap

import torch
from torch import nn
from adaptation import select_support, build_manifest, expand_subject_embeddings, freeze_for_adaptation


def row(subject, index, condition="vqa"):
    return {"record_id": f"{condition}:{subject}:{index}", "subject": subject, "condition": condition,
            "name": f"{index}.jpg", "qid": str(index), "task": "question", "X": [1], "Y": [2], "T": [100]}


class AdaptationTest(unittest.TestCase):
    def test_exactly_k_per_user_and_reproducible_under_reordering(self):
        rows = [row(subject, i) for subject in ("A", "B") for i in range(20)]
        selected, selection = select_support(rows, 3, 10)
        reversed_selected, reversed_selection = select_support(list(reversed(rows)), 3, 10)
        self.assertEqual(selection, reversed_selection)
        self.assertEqual(selected, reversed_selected)
        self.assertEqual(len(selected), 6)
        self.assertTrue(all(user["selected_count"] == 3 for user in selection.values()))
        self.assertNotEqual(selection, select_support(rows, 3, 11)[1])

    def test_insufficient_pool_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "requested"):
            select_support([row("A", 1)], 2, 10)

    def test_manifest_extends_observers_without_changing_vocab_or_old_indices(self):
        base = {"subjects": {"air:OLD": 0}, "text_vocabulary": {"<pad>": 0, "<unk>": 1}}
        result = build_manifest(base, [row("NEW", 1)], [row("NEW", 2)])
        self.assertEqual(result["subjects"], {"air:OLD": 0, "air:NEW": 1})
        self.assertEqual(base["subjects"], {"air:OLD": 0})
        self.assertEqual(result["text_vocabulary"], base["text_vocabulary"])
        self.assertEqual(result["record_splits"], ["train", "validation"])

    def test_overlap_and_missing_support_are_rejected(self):
        base = {"subjects": {"air:OLD": 0}, "text_vocabulary": {}}
        with self.assertRaisesRegex(ValueError, "overlapping"):
            build_manifest(base, [row("NEW", 1)], [row("NEW", 1)])
        with self.assertRaisesRegex(ValueError, "no selected support"):
            build_manifest(base, [row("NEW", 1)], [row("OTHER", 2)])

    def test_only_new_rows_change_and_frozen_layers_and_buffers_stay_fixed(self):
        network = nn.Module()
        network.subject_embed = nn.Embedding(2, 3)
        network.frozen = nn.Linear(3, 2)
        network.bn = nn.BatchNorm1d(3)
        network.dropout = nn.Dropout(.5)
        with torch.no_grad():
            network.subject_embed.weight.copy_(torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
        old_mapping = {"air:OLD": 0, "coco:1": 1}
        new_mapping = {**old_mapping, "air:NEW": 2, "coco:9": 3}
        expand_subject_embeddings(network, old_mapping, new_mapping)
        torch.testing.assert_close(network.subject_embed.weight[2], torch.tensor([1., 2., 3.]))
        torch.testing.assert_close(network.subject_embed.weight[3], torch.tensor([4., 5., 6.]))
        manifest = {"subjects": new_mapping, "adaptation_subjects": ["air:NEW", "coco:9"]}
        _, handle = freeze_for_adaptation(network, manifest)
        before = {key: value.clone() for key, value in network.state_dict().items()}
        optimizer = torch.optim.Adam([network.subject_embed.weight], lr=.01, weight_decay=0.)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            vectors = network.subject_embed(torch.tensor([2, 3]))
            loss = network.frozen(network.dropout(network.bn(vectors))).square().mean()
            loss.backward()
            optimizer.step()
        self.assertFalse(network.bn.training)
        self.assertFalse(network.dropout.training)
        self.assertEqual([n for n, p in network.named_parameters() if p.requires_grad], ["subject_embed.weight"])
        torch.testing.assert_close(network.subject_embed.weight[:2], before["subject_embed.weight"][:2], rtol=0, atol=0)
        self.assertFalse(torch.equal(network.subject_embed.weight[2:], before["subject_embed.weight"][2:]))
        for key, value in network.state_dict().items():
            if key != "subject_embed.weight":
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        handle.remove()


if __name__ == "__main__":
    unittest.main()
