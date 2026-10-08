"""Deterministic support sampling and embedding-only adaptation."""
import copy
import hashlib
import json
import random

import torch
from torch import nn

from dataset.schema import subject_key


def record_identity(row):
    if row.get("record_id"):
        return str(row["record_id"])
    fields = {key: row.get(key) for key in ("condition", "subject", "name", "qid", "task", "X", "Y", "T")}
    return hashlib.sha256(json.dumps(fields, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def select_support(rows, k, seed):
    if k < 1:
        raise ValueError("k must be positive")
    pools = {}
    for row in rows:
        pools.setdefault(subject_key(row), []).append(row)
    chosen, selection = [], {}
    for subject, pool in sorted(pools.items()):
        pool = sorted(pool, key=record_identity)
        if len({record_identity(row) for row in pool}) != len(pool):
            raise ValueError(f"Duplicate support records for {subject}")
        if len(pool) < k:
            raise ValueError(f"{subject}: only {len(pool)} support samples, requested k={k}")
        user_seed = int.from_bytes(hashlib.sha256(f"{seed}:{subject}".encode()).digest()[:8], "big")
        sampled = random.Random(user_seed).sample(pool, k)
        chosen.extend(sampled)
        selection[subject] = {"pool_size": len(pool), "selected_count": k,
                              "record_ids": [record_identity(row) for row in sampled]}
    return chosen, selection


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
                    adaptation_subjects=sorted(support_users))
    return manifest


def expand_subject_embeddings(network, old_subjects, subjects, initialization="mean"):
    old_weight = network.subject_embed.weight.detach().clone()
    if old_weight.shape[0] != len(old_subjects):
        raise ValueError("Checkpoint subject embedding and mapping disagree")
    replacement = nn.Embedding(len(subjects), old_weight.shape[1]).to(old_weight.device)
    with torch.no_grad():
        replacement.weight[:len(old_subjects)].copy_(old_weight)
        for key, index in subjects.items():
            if key in old_subjects:
                continue
            if initialization == "mean":
                source_indices = [i for user, i in old_subjects.items() if user.split(":", 1)[0] == key.split(":", 1)[0]]
                if not source_indices:
                    source_indices = list(old_subjects.values())
                replacement.weight[index].copy_(old_weight[source_indices].mean(0))
            else:
                nn.init.normal_(replacement.weight[index], std=1.)
    network.subject_embed = replacement


def freeze_for_adaptation(network, manifest):
    network.requires_grad_(False)
    network.eval()  # Frozen BatchNorm buffers and all dropout remain unchanged.
    weight = network.subject_embed.weight
    weight.requires_grad_(True)
    indices = [manifest["subjects"][key] for key in manifest["adaptation_subjects"]]
    mask = torch.zeros_like(weight)
    mask[indices] = 1
    handle = weight.register_hook(lambda gradient: gradient * mask)
    return indices, handle
