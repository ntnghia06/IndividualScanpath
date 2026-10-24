import argparse
import math
from pathlib import Path
from bootstrap import ROOT


def parse_args(evaluation=False):
    parser = argparse.ArgumentParser(description="PerGAZE ChenLSTM subject-embedding transfer to COCO-FV test")
    workspace = ROOT.parents[2]
    parser.add_argument("--checkpoint", type=Path, help="PerGAZE base checkpoint (adapted checkpoint for test.py)")
    parser.add_argument("--data_root", type=Path, default=workspace / "Experiments/coco_freeview")
    parser.add_argument("--test_file", type=Path)
    parser.add_argument("--img_dir", type=Path)
    parser.add_argument("--k", type=int, default=5, help="Support scanpaths per COCO-FV observer, sampled from test")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--eval_seed", type=int)
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--test_batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, choices=(0.,), default=0.,
                        help="ISP-S requires zero weight decay to preserve source observer rows")
    parser.add_argument("--clip", type=float, default=12.5)
    parser.add_argument("--init", choices=("mean", "random"), default="mean")
    parser.add_argument("--task_head", type=int, choices=range(-1, 18), default=-1,
                        help="Fixed transferred head: -1=VQA (default), 0..17=COCO object head")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu_ids", type=int, nargs="+")
    parser.add_argument("--eval_repeat_num", type=int, default=1)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--max_batches", type=int, default=0)
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--log_root", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare_only", action="store_true", help="Validate inputs and save support/remainder without loading weights")
    args = parser.parse_args()
    if min(args.k, args.epoch, args.batch, args.test_batch, args.eval_repeat_num) < 1 or args.lr <= 0:
        parser.error("k, epochs, batches, repeats and learning rate must be positive")
    if min(args.workers, args.max_batches, args.eval_max_batches, args.weight_decay) < 0:
        parser.error("workers, batch limits and weight decay must be nonnegative")
    if not all(math.isfinite(value) for value in (args.lr, args.weight_decay, args.clip)):
        parser.error("learning rate, weight decay and clip must be finite")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    if args.prepare_only and (args.resume or evaluation):
        parser.error("--prepare_only is only valid for a new split preparation with train.py")
    args.test_file = args.test_file or args.data_root / "COCOFreeView_fixations_test.json"
    args.img_dir = args.img_dir or args.data_root / "images"
    args.att_dir = None  # COCO-FV uses zero guidance, no external attention files.
    args.eval_seed = args.seed if args.eval_seed is None else args.eval_seed
    args.log_root = args.log_root or Path(__file__).resolve().parents[1] / f"runs/cocofv_subject_ft_k{args.k}_seed{args.seed}"
    if args.checkpoint is None:
        if evaluation:
            args.checkpoint = args.log_root / "checkpoints/best.pth"
        elif not args.prepare_only:
            parser.error("--checkpoint is required: supply the trained PerGAZE ChenLSTMISP checkpoint")
    return args
