"""OSIE subject mapping, embedding initialization and embedding-only adaptation."""
import copy

import torch
from torch import nn

IDENTITY_SCHEME = "osie_separate_v1"


from osie_data import subject_key, record_identity


def build_manifest(base_manifest, chosen, tests):
    support_users = {subject_key(row) for row in chosen}
    missing = {subject_key(row) for row in tests} - support_users
    if missing:
        raise ValueError(f"Test observers have no selected support: {sorted(missing)}")
    overlap = {record_identity(row) for row in chosen} & {record_identity(row) for row in tests}
    if overlap:
        raise ValueError("Selected support and test_unseen contain overlapping records")
    manifest = copy.deepcopy(base_manifest)
    subjects = manifest["subjects"]
    if sorted(subjects.values()) != list(range(len(subjects))):
        raise ValueError("Base subject mapping must use consecutive indices")
    for key in sorted(support_users - subjects.keys()):
        subjects[key] = len(subjects)
    manifest.update(version=3, split_mode="explicit_files",
                    record_splits=["train"] * len(chosen) + ["validation"] * len(tests),
                    adaptation_subjects=sorted(support_users),
                    subject_identity_scheme=IDENTITY_SCHEME,
                    adaptation_sampling_protocol="original_sigma2_v1")
    return manifest


def expand_subject_embeddings(network, old_subjects, subjects, initialization="mean"):
    old_weight = network.subject_embedding.weight.detach().clone()
    if old_weight.shape[0] != len(old_subjects):
        raise ValueError("Checkpoint subject embedding and mapping disagree")
    replacement = nn.Embedding(len(subjects), old_weight.shape[1]).to(old_weight.device)
    with torch.no_grad():
        replacement.weight[:len(old_subjects)].copy_(old_weight)
        for key, index in subjects.items():
            if key in old_subjects:
                continue
            if initialization == "mean":
                prefix = key.split(":", 1)[0]
                source_indices = [i for user, i in old_subjects.items()
                                  if user.split(":", 1)[0] == prefix]
                # Legacy pretrained checkpoints combined TA/TP as coco:*.
                if not source_indices and prefix in ("ta", "tp"):
                    source_indices = [i for user, i in old_subjects.items()
                                      if user.split(":", 1)[0] == "coco"]
                if not source_indices:
                    source_indices = list(old_subjects.values())
                replacement.weight[index].copy_(old_weight[source_indices].mean(0))
            else:
                nn.init.normal_(replacement.weight[index], std=.1)
    network.subject_embedding = replacement
    if hasattr(network, "subject_num"):
        network.subject_num = len(subjects)



def freeze_for_adaptation(network, manifest):
    """Only new OSIE rows can receive gradients; dropout/BatchNorm stay in eval."""
    network.requires_grad_(False)
    network.eval()
    weight = network.subject_embedding.weight
    weight.requires_grad_(True)
    indices = [manifest["subjects"][key] for key in manifest["adaptation_subjects"]]
    mask = torch.zeros_like(weight)
    mask[indices] = 1
    handle = weight.register_hook(lambda gradient: gradient * mask)
    manifest["adaptation_mode"] = "osie_subject_embedding_v1"
    return indices, handle


def require_embedding_checkpoint(manifest):
    if manifest.get("adaptation_mode") != "osie_subject_embedding_v1":
        raise ValueError("Expected OSIE embedding-only ISP-S checkpoint; full-FT checkpoints cannot resume ISP-S")
