"""PerGAZE metadata, subject identities and explicit file membership."""
import hashlib
import json
from collections import Counter
from pathlib import Path

CONDITION_DIR = {"present": "TP", "absent": "TA", "vqa": "VQA"}


def read_records(path):
    with open(path, encoding="utf-8-sig") as stream:
        records = json.load(stream)
    if not isinstance(records, list) or not records:
        raise ValueError("Expected a nonempty JSON array of scanpaths")
    for i, row in enumerate(records):
        if row.get("condition") not in CONDITION_DIR:
            raise ValueError(f"Record {i}: unknown condition")
        for key in ("name", "subject", "task", "X", "Y", "T"):
            if key not in row:
                raise ValueError(f"Record {i}: missing {key}")
        if not len(row["X"]) or not len(row["X"]) == len(row["Y"]) == len(row["T"]):
            raise ValueError(f"Record {i}: empty or inconsistent X/Y/T")
        if row["condition"] == "vqa" and not isinstance(row.get("qid"), str):
            raise ValueError(f"Record {i}: qid must be a string (preserve leading zeroes)")
        if row["condition"] == "present" and not row.get("bbox"):
            raise ValueError(f"Record {i}: present sample requires bbox")
    return records


def subject_key(row):
    # COCO TP and TA share observers; AiR observers form a separate population.
    return ("air:" if row["condition"] == "vqa" else "coco:") + str(row["subject"])


def image_key(row):
    # Also keep different COCO tasks/conditions on the same image in one split.
    return ("air:" if row["condition"] == "vqa" else "coco:") + row["name"]


def make_manifest(records, seed=1, ratios=(0.8, 0.1, 0.1)):
    if len(ratios) != 3 or min(ratios) <= 0 or abs(sum(ratios) - 1) > 1e-8:
        raise ValueError("Three positive split ratios must sum to one")
    subjects = {key: i for i, key in enumerate(sorted({subject_key(r) for r in records}))}
    # Stratify image groups by the set of conditions on that image.
    groups = {}
    for row in records:
        groups.setdefault(image_key(row), set()).add(row["condition"])
    strata = {}
    for key, conditions in groups.items():
        strata.setdefault(tuple(sorted(conditions)), []).append(key)
    splits = {}
    for keys in strata.values():
        keys.sort(key=lambda k: hashlib.sha256(f"{seed}:{k}".encode()).hexdigest())
        ntrain = int(len(keys) * ratios[0])
        nval = int(len(keys) * ratios[1])
        for i, key in enumerate(keys):
            splits[key] = "train" if i < ntrain else "validation" if i < ntrain + nval else "test"
    return {"version": 1, "seed": seed, "ratios": list(ratios), "subjects": subjects, "splits": splits}


def image_path(root, row):
    folder = Path(root) / CONDITION_DIR[row["condition"]]
    candidates = [folder / row["name"]]
    if row["condition"] != "vqa":
        candidates.insert(0, folder / row["task"] / row["name"])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Image not found: {candidates}")


def make_file_manifest(train_records, validation_records):
    subjects = {key: i for i, key in enumerate(sorted({subject_key(row) for row in train_records}))}
    unseen = {subject_key(row) for row in validation_records} - subjects.keys()
    if unseen:
        raise ValueError(f"Validation contains observers missing from training: {sorted(unseen)}")
    return {"version": 2, "split_mode": "explicit_files", "subjects": subjects,
            "record_splits": ["train"] * len(train_records) + ["validation"] * len(validation_records)}


def record_split(manifest, index, row):
    if manifest.get("split_mode") == "explicit_files":
        return manifest["record_splits"][index]
    return manifest["splits"][image_key(row)]


def split_counts(records, manifest):
    splits = ("train", "validation") if manifest.get("split_mode") == "explicit_files" else ("train", "validation", "test")
    return {split: dict(Counter(r["condition"] for index, r in enumerate(records)
                               if record_split(manifest, index, r) == split))
            for split in splits}
