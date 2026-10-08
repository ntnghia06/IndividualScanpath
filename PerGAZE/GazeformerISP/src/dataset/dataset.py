"""A single-scanpath loader with TP bbox, VQA maps and zero TA guidance."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset

from dataset.schema import record_split, image_path, subject_key
from geometry import SCANPATH_SIZE, IMAGE_SIZE, ACTION_GRID, ACTION_COUNT


def load_guidance(row, attention_dir, image_size, map_size=ACTION_GRID):
    width, height = image_size
    if row["condition"] == "absent":
        return np.zeros((1, *map_size), dtype=np.float32)
    if row["condition"] == "vqa":
        raw = np.load(Path(attention_dir) / (row["qid"] + ".npy"), allow_pickle=False)
        raw = np.squeeze(raw).astype(np.float32)
        if raw.ndim != 2 or not np.isfinite(raw).all() or raw.min() < 0:
            raise ValueError(f"Invalid attention map for {row['qid']}: shape={raw.shape}")
        raw = F.interpolate(torch.from_numpy(raw.copy())[None, None], size=map_size,
                            mode="bilinear", align_corners=False)[0, 0].numpy()
    else:
        boxes = np.asarray(row["bbox"], dtype=np.float32).reshape(-1, 4)
        if not np.isfinite(boxes).all() or (boxes[:, 2:] <= 0).any():
            raise ValueError(f"Invalid xywh bbox for {row['name']}")
        raw = np.zeros((height, width), dtype=np.float32)
        for x, y, w, h in boxes:
            x0, y0 = max(0, int(np.floor(x))), max(0, int(np.floor(y)))
            x1, y1 = min(width, int(np.ceil(x + w))), min(height, int(np.ceil(y + h)))
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"BBox outside image: {row['name']}")
            raw[y0:y1, x0:x1] = 1
        # Area interpolation preserves small target boxes when reducing resolution.
        raw = F.interpolate(torch.from_numpy(raw)[None, None], size=map_size,
                            mode="area")[0, 0].numpy()
    maximum = float(raw.max())
    if maximum:
        raw /= maximum
    return raw[None].copy()


class PerGAZE(Dataset):
    def __init__(self, records, manifest, image_dir, attention_dir, split="train",
                 resize=SCANPATH_SIZE, max_length=16, blur_sigma=1.0, image_resize=IMAGE_SIZE):
        self.manifest = manifest
        self.records = [(i, r) for i, r in enumerate(records)
                        if split == "all" or record_split(manifest, i, r) == split]
        self.image_dir, self.attention_dir = image_dir, attention_dir
        self.resize, self.max_length, self.blur_sigma = resize, max_length, blur_sigma
        self.image_resize = image_resize or resize
        self.map_size = ACTION_GRID

    def __len__(self):
        return len(self.records)

    def observer_key(self, row):
        return subject_key(row)

    def load_visual(self, row):
        with Image.open(image_path(self.image_dir, row)) as source:
            width, height = source.size
            image = T.functional.to_tensor(source.convert("RGB"))
        image = T.Resize(self.image_resize)(image)
        image = T.Normalize([.485, .456, .406], [.229, .224, .225])(image)
        return image, width, height

    def guidance(self, row, width, height):
        return torch.from_numpy(load_guidance(row, self.attention_dir, (width, height)))

    def __getitem__(self, index):
        record_index, row = self.records[index]
        image, width, height = self.load_visual(row)
        coords = np.asarray([row["X"], row["Y"], row["T"]], dtype=np.float32).T
        if not np.isfinite(coords).all() or (coords[:, 2] <= 0).any():
            raise ValueError(f"Record {record_index}: coordinates/durations must be finite; T > 0")
        # X/Y and bbox are in original image pixels, T is in milliseconds.
        coords[:, 0] = np.clip(coords[:, 0], 0, width - 1e-3) / width * self.resize[1]
        coords[:, 1] = np.clip(coords[:, 1], 0, height - 1e-3) / height * self.resize[0]
        coords[:, 2] /= 1000
        targets = np.zeros((self.max_length, ACTION_COUNT), dtype=np.float32)
        durations = np.zeros(self.max_length, dtype=np.float32)
        action_mask = np.zeros(self.max_length, dtype=np.float32)
        duration_mask = np.zeros(self.max_length, dtype=np.float32)
        count = min(len(coords), self.max_length)
        for t, (x, y, duration) in enumerate(coords[:count]):
            fixation = np.zeros(self.map_size, dtype=np.float32)
            fixation[min(self.map_size[0] - 1, int(y / self.resize[0] * self.map_size[0])),
                     min(self.map_size[1] - 1, int(x / self.resize[1] * self.map_size[1]))] = 1
            if self.blur_sigma:
                fixation = gaussian_filter(fixation, self.blur_sigma)
            targets[t, 1:] = (fixation / fixation.sum()).ravel()
            durations[t], action_mask[t], duration_mask[t] = duration, 1, 1
        if count < self.max_length:
            targets[count:, 0] = 1
            action_mask[count] = 1  # supervise exactly one stop action
        return {"images": image, "subjects": torch.tensor(self.manifest["subjects"][self.observer_key(row)]),
                "attention_maps": self.guidance(row, width, height),
                "target_scanpaths": torch.from_numpy(targets), "durations": torch.from_numpy(durations),
                "action_masks": torch.from_numpy(action_mask), "duration_masks": torch.from_numpy(duration_mask),
                "fix_vectors": coords, "metadata": {"record_index": record_index, "name": row["name"],
                    "condition": row["condition"], "task": row["task"], "qid": row.get("qid"),
                    "subject": row["subject"], "subject_key": self.observer_key(row)}}


def collate_func(batch):
    result = {key: torch.stack([row[key] for row in batch]) for key in batch[0]
              if key not in ("fix_vectors", "metadata")}
    result.update({key: [row[key] for row in batch] for key in ("fix_vectors", "metadata")})
    return result
