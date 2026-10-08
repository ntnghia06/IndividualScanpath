import argparse
from pathlib import Path
from bootstrap import BASE_SRC
from opts import DATASET


def parse_args(evaluation=False):
    parser = argparse.ArgumentParser(description="K-shot subject-embedding adaptation of PerGAZE GazeformerISP")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--support_file", type=Path, default=DATASET / "support.json")
    parser.add_argument("--test_file", type=Path, default=DATASET / "test_unseen.json")
    parser.add_argument("--img_dir", type=Path, default=DATASET / "images")
    parser.add_argument("--att_dir", type=Path, default=DATASET / "attention_reasoning")
    parser.add_argument("--text_embeddings", type=Path)
    parser.add_argument("--feature_dir", type=Path, default=DATASET / "gazeformer_image_features")
    parser.add_argument("--k", type=int, default=5, help="Exactly k support scanpaths per observer")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--eval_seed", type=int, help="Defaults to seed; same evaluation draws after each epoch")
    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--test_batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--clip", type=float, default=12.5)
    parser.add_argument("--init", choices=("mean", "random"), default="mean")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu_ids", type=int, nargs="+")
    parser.add_argument("--eval_repeat_num", type=int, default=1)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--max_batches", type=int, default=0, help="Functional checks only; 0 trains all selected support")
    parser.add_argument("--eval_max_batches", type=int, default=0, help="Functional checks only; 0 evaluates the full unseen set")
    parser.add_argument("--log_root", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if min(args.k, args.epoch, args.batch, args.test_batch, args.eval_repeat_num) < 1 or args.lr <= 0:
        parser.error("k, epochs, batches, repeats and learning rate must be positive")
    if min(args.workers, args.max_batches, args.eval_max_batches) < 0:
        parser.error("workers and batch limits must be nonnegative")
    args.eval_seed = args.seed if args.eval_seed is None else args.eval_seed
    if args.log_root is None:
        args.log_root = Path(__file__).resolve().parents[1] / f"runs/k{args.k}_seed{args.seed}_by_condition"
    if args.checkpoint is None:
        args.checkpoint = (args.log_root / "checkpoints/best.pth" if evaluation else
                           BASE_SRC.parents[2] / "pergaze_gazeformer_run/checkpoints/best.pth")
    return args
