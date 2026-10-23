"""Prepare and validate a safe NPZ with one fixed free-viewing sentence vector."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

PROMPT = "You are allowed to view visual stimuli or scenes without any specific task or instruction."
ENCODER = "sentence-transformers/stsb-roberta-base-v2"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_vector(path, prompt=PROMPT):
    with np.load(path, allow_pickle=False) as archive:
        if not {"tasks", "vectors", "encoder"} <= set(archive.files):
            raise ValueError("Use prepare_text.py to create the free-viewing NPZ with encoder metadata")
        if archive["encoder"].item() != ENCODER:
            raise ValueError("Free-viewing text must use the base SentenceTransformer encoder")
        tasks = archive["tasks"].tolist()
        vectors = archive["vectors"].astype(np.float32)
    if (not isinstance(tasks, list) or len(tasks) != len(set(tasks))
            or vectors.shape != (len(tasks), 768) or not np.isfinite(vectors).all()):
        raise ValueError("Invalid 768-dimensional free-viewing sentence archive")
    if prompt not in tasks:
        raise ValueError("Free-viewing prompt is missing from the text archive")
    return vectors[tasks.index(prompt)].copy()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Encode the fixed OSIE free-viewing instruction once")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if not args.prompt.strip():
        parser.error("prompt cannot be empty")
    if args.output.exists() and not args.overwrite:
        load_vector(args.output, args.prompt)
        print("Reusing validated free-viewing embedding:", args.output)
        return
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(ENCODER, device=args.device).eval()
    vectors = np.asarray(encoder.encode([args.prompt], show_progress_bar=False), dtype=np.float32)
    if vectors.shape != (1, 768) or not np.isfinite(vectors).all():
        raise ValueError("Text encoder returned invalid sentence features")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as stream:
        np.savez(stream, tasks=np.asarray([args.prompt]), vectors=vectors, encoder=np.asarray(ENCODER))
    load_vector(args.output, args.prompt)
    print("Saved:", args.output, "shape:", vectors.shape, "sha256:", sha256(args.output))
