"""Validated OSIE records, stable per-observer support selection, and image loading."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset


def subject_key(row):
    return "osie:" + str(row["subject"])


def record_identity(row):
    # One scanpath per (image, observer); do not allow altered duplicates to cross splits.
    return json.dumps([str(row["name"]), subject_key(row)], separators=(",", ":"))


def read_records(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty OSIE JSON array")
    identities = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Record {index}: expected an object")
        for field in ("name", "subject", "X", "Y", "T", "length", "split"):
            if field not in row:
                raise ValueError(f"Record {index}: missing {field}")
        name = row["name"]
        if not isinstance(name, str) or not name or "/" in name or "\\" in name or ":" in name:
            raise ValueError(f"Record {index}: name must be a flat image filename")
        if not isinstance(row["subject"], int) or isinstance(row["subject"], bool) or row["subject"] < 1:
            raise ValueError(f"Record {index}: OSIE subject must be a positive integer")
        if row["split"] != "test":
            raise ValueError(f"Record {index}: support must be sampled exclusively from OSIE test")
        if not all(isinstance(row[k], list) for k in ("X", "Y", "T")):
            raise ValueError(f"Record {index}: X/Y/T must be lists")
        if not len(row["X"]) == len(row["Y"]) == len(row["T"]) == row["length"] or row["length"] < 1:
            raise ValueError(f"Record {index}: inconsistent fixation lengths")
        coords = np.asarray([row["X"], row["Y"], row["T"]], dtype=float)
        if not np.isfinite(coords).all() or (coords[2] <= 0).any():
            raise ValueError(f"Record {index}: fixation values must be finite and T > 0")
        identity = record_identity(row)
        if identity in identities:
            raise ValueError(f"Duplicate OSIE image/observer record: {identity}")
        identities.add(identity)
    return rows


def split_test(rows, k, seed):
    """Sample exactly k records per subject, holding out all remaining records."""
    if k < 1:
        raise ValueError("k must be positive")
    pools = {}
    for row in rows:
        pools.setdefault(subject_key(row), []).append(row)
    chosen, remaining, selection = [], [], {}
    for observer, pool in sorted(pools.items()):
        pool = sorted(pool, key=record_identity)
        identities = [record_identity(row) for row in pool]
        if len(set(identities)) != len(pool):
            raise ValueError(f"Duplicate records for {observer}")
        if len(pool) <= k:
            raise ValueError(f"{observer}: k={k} must leave at least one evaluation scanpath")
        observer_seed = int.from_bytes(hashlib.sha256(f"{seed}:{observer}".encode()).digest()[:8], "big")
        sampled = random.Random(observer_seed).sample(pool, k)
        selected_ids = {record_identity(row) for row in sampled}
        held = [row for row in pool if record_identity(row) not in selected_ids]
        chosen.extend(sampled)
        remaining.extend(held)
        selection[observer] = {"pool_size": len(pool), "selected_count": k,
                               "record_ids": [record_identity(row) for row in sampled],
                               "evaluation_count": len(held)}
    return chosen, remaining, selection


def prepare_split(args):
    rows = read_records(args.test_file)
    for name in sorted({row["name"] for row in rows}):
        if not (args.img_dir / name).is_file():
            raise FileNotFoundError(args.img_dir / name)
    chosen, remaining, selection = split_test(rows, args.k, args.seed)
    return chosen, remaining, selection


class SubjectAdaptationDataset(Dataset):
    """Keep the PerGAZE ChenLSTM 320x240 preprocessing and 40x30 action grid."""
    def __init__(self, records, manifest, image_dir, attention_dir=None, split="train",
                 max_length=16, blur_sigma=None, task_head=-1):
        self.rows = [(i, row) for i, row in enumerate(records)
                     if manifest["record_splits"][i] == split]
        self.manifest, self.image_dir = manifest, Path(image_dir)
        self.max_length, self.blur_sigma, self.task_head = max_length, blur_sigma, task_head

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        record_index, row = self.rows[index]
        with Image.open(self.image_dir / row["name"]) as source:
            width, height = source.size
            image = np.asarray(source.convert("RGB").resize((320, 240), Image.Resampling.BILINEAR),
                               dtype=np.float32) / 255
        image = torch.from_numpy(image.copy()).permute(2, 0, 1)
        image = (image - torch.tensor([.485, .456, .406])[:, None, None]) / torch.tensor([.229, .224, .225])[:, None, None]
        coords = np.asarray([row["X"], row["Y"], row["T"]], dtype=np.float32).T
        coords[:, 0] = np.clip(coords[:, 0], 0, width - 1e-3) / width * 320
        coords[:, 1] = np.clip(coords[:, 1], 0, height - 1e-3) / height * 240
        coords[:, 2] /= 1000
        targets = np.zeros((self.max_length, 1201), dtype=np.float32)
        durations = np.zeros(self.max_length, dtype=np.float32)
        action_mask = np.zeros(self.max_length, dtype=np.float32)
        duration_mask = np.zeros(self.max_length, dtype=np.float32)
        count = min(len(coords), self.max_length)
        for t, (x, y, duration) in enumerate(coords[:count]):
            fixation = np.zeros((30, 40), dtype=np.float32)
            fixation[min(29, int(y / 240 * 30)), min(39, int(x / 320 * 40))] = 1
            if self.blur_sigma:
                fixation = gaussian_filter(fixation, self.blur_sigma)
            targets[t, 1:] = (fixation / fixation.sum()).ravel()
            durations[t], action_mask[t], duration_mask[t] = duration, 1, 1
        if count < self.max_length:
            targets[count:, 0] = 1
            action_mask[count] = 1
        return {"images": image, "subjects": torch.tensor(self.manifest["subjects"][subject_key(row)]),
                "task_heads": torch.tensor(self.task_head, dtype=torch.long),
                "attention_maps": torch.zeros(1, 30, 40),
                "target_scanpaths": torch.from_numpy(targets), "durations": torch.from_numpy(durations),
                "action_masks": torch.from_numpy(action_mask), "duration_masks": torch.from_numpy(duration_mask),
                "fix_vectors": coords, "metadata": {"record_index": record_index, "name": row["name"],
                    "condition": "freeview", "task": "free-viewing", "qid": None,
                    "subject": row["subject"], "subject_key": subject_key(row),
                    "original_size": [width, height]}}
