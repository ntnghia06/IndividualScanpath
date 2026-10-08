"""Portable cache keys and fixed COCO image feature contract."""
import hashlib
import gzip
import shutil
from pathlib import Path
import torch
from dataset.schema import image_path

FEATURE_FORMAT = "resnet50_coco_tensor_resize_fp32_v2"


def cache_path(feature_dir, image_dir, row):
    relative = image_path(image_dir, row).relative_to(Path(image_dir)).as_posix()
    key = hashlib.sha256(relative.encode("utf-8")).hexdigest()
    return Path(feature_dir) / (key + ".pth"), relative


def is_compressed(path):
    with Path(path).open("rb") as stream:
        return stream.read(2) == b"\x1f\x8b"


def save_features(path, relative, features, width, height, compression="gzip"):
    """Write one independent FP32 storage atomically, optionally losslessly compressed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if features.dtype != torch.float32:
        raise ValueError("Feature cache must retain FP32 values")
    if compression not in ("gzip", "none"):
        raise ValueError(f"Unknown feature compression: {compression}")
    # Slices returned by iterating a batch share the entire batch storage.
    compact = features.detach().cpu().clone(memory_format=torch.contiguous_format)
    free = shutil.disk_usage(path.parent).free
    if free < compact.numel() * compact.element_size() + 1024 * 1024:
        raise OSError(f"Insufficient disk space to write {path}; free={free / 2**20:.1f} MiB. "
                      "Remove old feature caches or use a fresh output directory/session.")
    payload = {"format": FEATURE_FORMAT, "source": relative, "features": compact,
               "width": int(width), "height": int(height)}
    temporary = path.with_suffix(".tmp")
    try:
        if compression == "gzip":
            with gzip.open(temporary, "wb", compresslevel=1) as stream:
                torch.save(payload, stream)
        else:
            torch.save(payload, temporary)
        temporary.replace(path)
    except (OSError, RuntimeError) as error:
        temporary.unlink(missing_ok=True)
        available = shutil.disk_usage(path.parent).free
        raise OSError(f"Could not write feature {path}; free disk={available / 2**30:.2f} GiB. "
                      f"Original write error: {error}") from error


def load_features(path, relative):
    with Path(path).open("rb") as raw:
        magic = raw.read(2)
        raw.seek(0)
        if magic == b"\x1f\x8b":
            with gzip.GzipFile(fileobj=raw, mode="rb") as stream:
                saved = torch.load(stream, map_location="cpu", weights_only=True)
        else:
            saved = torch.load(raw, map_location="cpu", weights_only=True)
    features = saved["features"]
    if (saved.get("format") != FEATURE_FORMAT or saved.get("source") != relative
            or features.dtype != torch.float32 or features.shape != (2048, 24, 32) or not torch.isfinite(features).all()
            or min(saved["width"], saved["height"]) <= 0):
        raise ValueError(f"Invalid/obsolete COCO feature cache; re-extract with --overwrite: {path}")
    if features.untyped_storage().nbytes() != features.numel() * features.element_size():
        features = features.clone()  # Compact old batch-view caches in RAM as well.
    return features, saved["width"], saved["height"]
