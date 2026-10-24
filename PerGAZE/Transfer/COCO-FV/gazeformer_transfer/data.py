"""Validated COCO-FV records, stable per-observer support selection, and image loading."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
import torchvision.transforms as T

from geometry import IMAGE_SIZE, SCANPATH_SIZE, ACTION_GRID, ACTION_COUNT


def subject_key(row):
    return "cocofv:" + str(row["subject"])


def record_identity(row):
    # One scanpath per (image, observer); do not allow altered duplicates to cross splits.
    return json.dumps([str(row["name"]), subject_key(row)], separators=(",", ":"))


def read_records(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        rows = json.load(stream)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty COCO-FV JSON array")
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
            raise ValueError(f"Record {index}: COCO-FV subject must be a positive integer")
        if row.get("condition") != "freeview":
            raise ValueError(f"Record {index}: COCO-FV requires condition=freeview")
        if row["split"] != "test":
            raise ValueError(f"Record {index}: support must be sampled exclusively from COCO-FV test")
        if not all(isinstance(row[k], list) for k in ("X", "Y", "T")):
            raise ValueError(f"Record {index}: X/Y/T must be lists")
        if not len(row["X"]) == len(row["Y"]) == len(row["T"]) == row["length"] or row["length"] < 1:
            raise ValueError(f"Record {index}: inconsistent fixation lengths")
        coords = np.asarray([row["X"], row["Y"], row["T"]], dtype=float)
        if not np.isfinite(coords).all() or (coords[2] <= 0).any():
            raise ValueError(f"Record {index}: fixation values must be finite and T > 0")
        identity = record_identity(row)
        if identity in identities:
            raise ValueError(f"Duplicate COCO-FV image/observer record: {identity}")
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


class COCOFVDataset(Dataset):
    """Preserve the source Gazeformer image preprocessing and geometry-v3 targets."""
    def __init__(self, records, manifest, image_dir, vector, split="train",
                 max_length=16, blur_sigma=None, max_text_length=64):
        self.rows = [(i, row) for i, row in enumerate(records) if manifest["record_splits"][i] == split]
        self.manifest, self.image_dir = manifest, Path(image_dir)
        self.vector = torch.as_tensor(vector, dtype=torch.float32).clone()
        if self.vector.shape != (768,) or not torch.isfinite(self.vector).all():
            raise ValueError("Expected one finite 768-dimensional free-viewing vector")
        self.max_length, self.blur_sigma = max_length, blur_sigma
        self.max_text_length = max_text_length
        self.resize = T.Resize(IMAGE_SIZE)
        self.normalize = T.Normalize([.485, .456, .406], [.229, .224, .225])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        record_index, row = self.rows[index]
        with Image.open(self.image_dir / row["name"]) as source:
            width, height = source.size
            image = T.functional.to_tensor(source.convert("RGB"))
        image = self.normalize(self.resize(image))
        coords = np.asarray([row["X"], row["Y"], row["T"]], dtype=np.float32).T
        coords[:, 0] = np.clip(coords[:, 0], 0, width - 1e-3) / width * SCANPATH_SIZE[1]
        coords[:, 1] = np.clip(coords[:, 1], 0, height - 1e-3) / height * SCANPATH_SIZE[0]
        coords[:, 2] /= 1000
        targets = np.zeros((self.max_length, ACTION_COUNT), dtype=np.float32)
        durations = np.zeros(self.max_length, dtype=np.float32)
        action_mask = np.zeros(self.max_length, dtype=np.float32)
        duration_mask = np.zeros(self.max_length, dtype=np.float32)
        count = min(len(coords), self.max_length)
        for t, (x, y, duration) in enumerate(coords[:count]):
            fixation = np.zeros(ACTION_GRID, dtype=np.float32)
            cell_y = min(ACTION_GRID[0] - 1, int(y / SCANPATH_SIZE[0] * ACTION_GRID[0]))
            cell_x = min(ACTION_GRID[1] - 1, int(x / SCANPATH_SIZE[1] * ACTION_GRID[1]))
            fixation[cell_y, cell_x] = 1
            if self.blur_sigma:
                fixation = gaussian_filter(fixation, self.blur_sigma)
            targets[t, 1:] = (fixation / fixation.sum()).ravel()
            durations[t], action_mask[t], duration_mask[t] = duration, 1, 1
        if count < self.max_length:
            targets[count:, 0] = 1
            action_mask[count] = 1
        return {"images": image, "subjects": torch.tensor(self.manifest["subjects"][subject_key(row)]),
            "attention_maps": torch.zeros(1, *ACTION_GRID),
            "task_embeddings": self.vector.clone(),
            "task_tokens": torch.zeros(self.max_text_length, dtype=torch.long),
            "task_mask": torch.tensor(1.),
            "target_scanpaths": torch.from_numpy(targets), "durations": torch.from_numpy(durations),
            "action_masks": torch.from_numpy(action_mask), "duration_masks": torch.from_numpy(duration_mask),
            "fix_vectors": coords, "metadata": {"record_index": record_index, "name": row["name"],
                "condition": "freeview", "task": "free-viewing", "qid": None,
                "subject": row["subject"], "subject_key": subject_key(row),
                "original_size": [width, height]}}
