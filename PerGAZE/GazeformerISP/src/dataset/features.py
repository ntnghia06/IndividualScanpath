"""Portable cache keys and fixed COCO image feature contract."""
import hashlib
from pathlib import Path
import torch
from dataset.schema import image_path

FEATURE_FORMAT = "resnet50_coco_tensor_resize_fp32_v2"


def cache_path(feature_dir, image_dir, row):
    relative = image_path(image_dir, row).relative_to(Path(image_dir)).as_posix()
    key = hashlib.sha256(relative.encode("utf-8")).hexdigest()
    return Path(feature_dir) / (key + ".pth"), relative


def load_features(path, relative):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    features = saved["features"]
    if (saved.get("format") != FEATURE_FORMAT or saved.get("source") != relative
            or features.dtype != torch.float32 or features.shape != (2048, 24, 32) or not torch.isfinite(features).all()
            or min(saved["width"], saved["height"]) <= 0):
        raise ValueError(f"Invalid/obsolete COCO feature cache; re-extract with --overwrite: {path}")
    return features.float(), saved["width"], saved["height"]
