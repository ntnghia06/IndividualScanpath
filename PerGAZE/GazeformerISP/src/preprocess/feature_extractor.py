"""Optional pretrained sentence embeddings compatible with the unified loader."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from dataset.schema import read_records
from opts import DATASET


def main():
    parser = argparse.ArgumentParser(description="Generate PerGAZE semantic task embeddings")
    parser.add_argument("--data_file", type=Path, default=DATASET / "PerGAZE.json")
    parser.add_argument("--output", type=Path, default=DATASET / "gazeformer_task_embeddings.npz")
    parser.add_argument("--lm_model", default="sentence-transformers/stsb-roberta-base-v2")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer
    tasks = sorted({r["task"] for r in read_records(args.data_file) if r["condition"] != "absent"})
    encoder = SentenceTransformer(args.lm_model, device=args.device)
    vectors = encoder.encode(tasks, batch_size=32, show_progress_bar=True).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, tasks=np.asarray(tasks), vectors=vectors)
    print(f"Saved {len(tasks)} task embeddings, shape={vectors.shape}, to {args.output}")


if __name__ == "__main__":
    main()
