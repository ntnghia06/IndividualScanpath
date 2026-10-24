import copy

import torch
from torch import nn

from .data import subject_key, record_identity

MODES = {"predictor": "cocofv_gazeformer_predictor_frozen_backbone_v1",
         "embedding": "cocofv_gazeformer_subject_embedding_v1"}
PROTOCOL = "cocofv_test_support_record_split_v1"


def build_manifest(base, chosen, tests, mode):
    support_users = {subject_key(r) for r in chosen}
    if {subject_key(r) for r in tests} - support_users:
        raise ValueError("Every evaluated COCO-FV observer needs support")
    if {record_identity(r) for r in chosen} & {record_identity(r) for r in tests}:
        raise ValueError("Support and evaluation records overlap")
    manifest = copy.deepcopy(base)
    subjects = manifest["subjects"]
    if not subjects or sorted(subjects.values()) != list(range(len(subjects))):
        raise ValueError("Base observer indices must be nonempty and consecutive")
    if support_users & subjects.keys():
        raise ValueError("Expected new COCO-FV subjects absent from the PerGAZE checkpoint")
    for key in sorted(support_users):
        subjects[key] = len(subjects)
    manifest.update(dataset="COCO-FV", subjects=subjects, adaptation_subjects=sorted(support_users),
        record_splits=["train"] * len(chosen) + ["validation"] * len(tests),
        split_mode="explicit_files", subject_identity_scheme="cocofv_separate_v1",
        adaptation_mode=MODES[mode], transfer_protocol=PROTOCOL,
        adaptation_sampling_protocol="original_sigma2_v1", image_backend="online_coco_v1",
        metric_protocol="original_mm_retrieval_duration_v2")
    return manifest


def expand_embeddings(network, old_subjects, subjects, initialization):
    original = network.subject_embed.weight.detach().clone()
    if len(original) != len(old_subjects):
        raise ValueError("Checkpoint embedding table and subject map disagree")
    replacement = nn.Embedding(len(subjects), original.shape[1]).to(original.device)
    with torch.no_grad():
        replacement.weight[:len(original)].copy_(original)
        for key, index in subjects.items():
            if key in old_subjects:
                continue
            if initialization == "mean":
                replacement.weight[index].copy_(original.mean(0))
            else:
                # Match the GazeformerISP-S branch's seeded random initialization.
                nn.init.normal_(replacement.weight[index], std=1.)
    network.subject_embed = replacement


def configure(network, manifest, mode):
    hook = None
    if mode == "embedding":
        network.requires_grad_(False).eval()
        weight = network.subject_embed.weight
        weight.requires_grad_(True)
        indices = [manifest["subjects"][key] for key in manifest["adaptation_subjects"]]
        mask = torch.zeros_like(weight)
        mask[indices] = 1
        hook = weight.register_hook(lambda gradient: gradient * mask)
        elements = len(indices) * weight.shape[1]
    else:
        # Match the current GazeformerISP-FT protocol: predictor only, fixed COCO backbone.
        network.requires_grad_(True)
        network.backbone.requires_grad_(False)
        network.train()
        network.backbone.eval()
        elements = sum(p.numel() for p in network.parameters() if p.requires_grad)
    parameters = [p for p in network.parameters() if p.requires_grad]
    return parameters, elements, hook


def require_mode(manifest, mode):
    if (manifest.get("adaptation_mode") != MODES[mode]
            or manifest.get("transfer_protocol") != PROTOCOL
            or manifest.get("subject_identity_scheme") != "cocofv_separate_v1"
            or manifest.get("adaptation_sampling_protocol") != "original_sigma2_v1"
            or manifest.get("metric_protocol") != "original_mm_retrieval_duration_v2"):
        raise ValueError(f"Expected a compatible COCO-FV Gazeformer {mode} transfer checkpoint")
