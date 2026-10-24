import argparse
import math
from pathlib import Path

from .text import PROMPT
from bootstrap import ROOT


def parse_args(mode, branch, evaluation=False, argv=None):
    parser = argparse.ArgumentParser(description=f"Gazeformer {mode} transfer from PerGAZE to COCO-FV")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data_root", type=Path, default=ROOT.parents[2] / "Experiments/coco_freeview")
    parser.add_argument("--test_file", type=Path)
    parser.add_argument("--img_dir", type=Path)
    parser.add_argument("--text_embeddings", type=Path, help="Separate free-viewing NPZ made by prepare_text.py")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--eval_seed", type=int)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3 if mode == "embedding" else 1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--clip", type=float, default=12.5)
    parser.add_argument("--init", choices=("mean", "random"), default="mean")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--test_batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu_ids", type=int, nargs="+")
    parser.add_argument("--eval_repeat_num", type=int, default=1)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--log_root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare_only", action="store_true", help="Save support/remainder without loading text or weights")
    args = parser.parse_args(argv)
    if min(args.k, args.epoch, args.batch, args.test_batch, args.eval_repeat_num) < 1:
        parser.error("K, epochs, batch sizes and repeats must be positive")
    if not all(math.isfinite(v) for v in (args.lr, args.weight_decay, args.clip)) or args.lr <= 0:
        parser.error("LR must be positive; LR, weight decay and clipping must be finite")
    if min(args.weight_decay, args.workers, args.max_batches, args.eval_max_batches) < 0:
        parser.error("weight decay, workers and batch limits must be nonnegative")
    if mode == "embedding" and args.weight_decay != 0:
        parser.error("Embedding-only ISP-S requires zero weight decay")
    args.eval_seed = args.seed if args.eval_seed is None else args.eval_seed
    if not all(0 <= value < 2**32 for value in (args.seed, args.eval_seed)):
        parser.error("Seeds must be in [0, 2**32)")
    if not args.prompt.strip():
        parser.error("Free-viewing prompt cannot be empty")
    if args.prepare_only and (args.resume or evaluation):
        parser.error("--prepare_only is only valid for a new training split")
    args.test_file = args.test_file or args.data_root / "COCOFreeView_fixations_test.json"
    args.img_dir = args.img_dir or args.data_root / "images"
    args.text_embeddings = args.text_embeddings or args.data_root / "gazeformer_freeview_embeddings.npz"
    args.log_root = args.log_root or branch / f"runs/cocofv_{mode}_k{args.k}_seed{args.seed}"
    if args.checkpoint is None:
        if evaluation:
            args.checkpoint = args.log_root / "checkpoints/best.pth"
        elif not args.prepare_only:
            parser.error("Supply the trained PerGAZE GazeformerISP --checkpoint")
    return args
