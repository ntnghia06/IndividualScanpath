import re
import numpy as np
import torch

from dataset.dataset import PerGAZE
from dataset.schema import record_split


def tokenize(text):
    return re.findall(r"[\w']+|[^\w\s]", text.lower())


def add_text_manifest(records, manifest):
    words = set()
    for index, row in enumerate(records):
        if row["condition"] != "absent" and record_split(manifest, index, row) == "train":
            words.update(tokenize(row["task"]))
    manifest["text_vocabulary"] = {"<pad>": 0, "<unk>": 1,
                                   **{word: i + 2 for i, word in enumerate(sorted(words))}}


class GazeformerPerGAZE(PerGAZE):
    def __init__(self, *args, text_embeddings=None, max_text_length=64, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_text_length = max_text_length
        self.embeddings = None
        if text_embeddings:
            # Safe NPZ of Unicode task strings plus a float matrix; no pickled dict.
            with np.load(text_embeddings, allow_pickle=False) as archive:
                tasks, vectors = archive["tasks"].tolist(), archive["vectors"].astype(np.float32)
            if vectors.ndim != 2 or len(tasks) != len(vectors) or not np.isfinite(vectors).all():
                raise ValueError("Invalid task embedding archive")
            self.embeddings = dict(zip(tasks, vectors))
            self.text_dim = vectors.shape[1]
            missing = {r["task"] for _, r in self.records if r["condition"] != "absent"} - self.embeddings.keys()
            if missing:
                raise ValueError(f"Missing {len(missing)} task embeddings")

    def __getitem__(self, index):
        sample = super().__getitem__(index)
        row = self.records[index][1]
        active = row["condition"] != "absent"
        sample["task_mask"] = torch.tensor(float(active))
        tokens = torch.zeros(self.max_text_length, dtype=torch.long)
        if active:
            vocabulary = self.manifest["text_vocabulary"]
            ids = [vocabulary.get(word, 1) for word in tokenize(row["task"])][:self.max_text_length] or [1]
            tokens[:len(ids)] = torch.tensor(ids)
        sample["task_tokens"] = tokens
        if self.embeddings is not None:
            sample["task_embeddings"] = torch.from_numpy(self.embeddings[row["task"]].copy()) if active else torch.zeros(self.text_dim)
        return sample
