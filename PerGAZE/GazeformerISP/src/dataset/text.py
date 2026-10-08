import re
import numpy as np
import torch

from dataset.dataset import PerGAZE
from dataset.schema import record_split
from dataset.features import cache_path, load_features


def task_text(row):
    task = row["task"].strip()
    if row["condition"] == "absent":
        return f"Search for the {task.replace('_', ' ')} in the image."
    return task.replace("_", " ") if row["condition"] == "present" else task


def tokenize(text):
    return re.findall(r"[\w']+|[^\w\s]", text.lower())


def add_text_manifest(records, manifest):
    words = set()
    for index, row in enumerate(records):
        if record_split(manifest, index, row) == "train":
            words.update(tokenize(task_text(row)))
    manifest["text_vocabulary"] = {"<pad>": 0, "<unk>": 1,
                                   **{word: i + 2 for i, word in enumerate(sorted(words))}}


class GazeformerPerGAZE(PerGAZE):
    def __init__(self, *args, text_embeddings=None, feature_dir=None, max_text_length=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_text_length = max_text_length
        self.embeddings = None
        self.feature_dir = feature_dir
        if text_embeddings is None or feature_dir is None:
            raise ValueError("Gazeformer requires precomputed --text_embeddings and --feature_dir")
        if text_embeddings:
            # Safe NPZ of Unicode task strings plus a float matrix; no pickled dict.
            with np.load(text_embeddings, allow_pickle=False) as archive:
                tasks, vectors = archive["tasks"].tolist(), archive["vectors"].astype(np.float32)
            if vectors.ndim != 2 or vectors.shape[1] != 768 or len(tasks) != len(vectors) or not np.isfinite(vectors).all():
                raise ValueError("Invalid task embedding archive")
            self.embeddings = dict(zip(tasks, vectors))
            self.text_dim = vectors.shape[1]
            missing = {task_text(r) for _, r in self.records} - self.embeddings.keys()
            if missing:
                raise ValueError(f"Missing {len(missing)} task embeddings")

    def load_visual(self, row):
        path, relative = cache_path(self.feature_dir, self.image_dir, row)
        return load_features(path, relative)

    def guidance(self, row, width, height):
        # Gazeformer learns visual attention from image features and text.
        return torch.zeros(1, *self.map_size)

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        row = self.records[index][1]
        sample["task_mask"] = torch.tensor(1.)
        # Compatibility fields for the shared runner; no learned word encoder.
        sample["task_tokens"] = torch.zeros(self.max_text_length, dtype=torch.long)
        sample["task_embeddings"] = torch.from_numpy(self.embeddings[task_text(row)].copy())
        return sample
