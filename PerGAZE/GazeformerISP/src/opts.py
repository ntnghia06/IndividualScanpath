import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[5]
DATASET = PROJECT / "dataset/PerGAZED/dataset"
RUN = Path(__file__).resolve().parents[1] / "runs"


def parse_opt(description="Train unified PerGAZE"):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--train_file", "--data_file", dest="train_file", type=Path, default=DATASET / "train.json")
    parser.add_argument("--val_file", type=Path, default=DATASET / "test_seen.json")
    parser.add_argument("--img_dir", type=Path, default=DATASET / "images")
    parser.add_argument("--att_dir", type=Path, default=DATASET / "attention_reasoning")
    parser.add_argument("--log_root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda:0")
    parser.add_argument("--gpu_ids", type=int, nargs="+", help="Visible CUDA IDs; default uses all available GPUs")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0, help="0 works on Windows; increase on Linux")
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--start_rl_epoch", type=int, default=5, help="Zero-based epoch; >= epoch disables RL")
    parser.add_argument("--rl_sample_number", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rl_lr_initial_decay", type=float, default=.1)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--clip", type=float, default=-1)
    parser.add_argument("--lambda_1", type=float, default=1.)
    parser.add_argument("--max_length", type=int, default=16)
    parser.add_argument("--min_length", type=int, default=1)
    parser.add_argument("--blur_sigma", type=float, default=None)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--action_map_num", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=.4)
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_batches", type=int, default=0, help="0 processes the entire split")
    parser.add_argument("--eval_max_batches", type=int, default=0)
    parser.add_argument("--split", choices=("train", "validation", "all"), default="validation")
    parser.add_argument("--eval_repeat_num", type=int, default=1)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--smoke_test", action="store_true", help="Forward/backward and inference on one real record per condition")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_encoder", type=int, default=6)
    parser.add_argument("--num_decoder", type=int, default=6)
    parser.add_argument("--encoder_dropout", type=float, default=.1)
    parser.add_argument("--decoder_dropout", type=float, default=.2)
    parser.add_argument("--text_dim", type=int, default=768)
    parser.add_argument("--max_text_length", type=int, default=64)
    parser.add_argument("--text_embeddings", type=Path, default=DATASET / "gazeformer_task_embeddings.npz")
    parser.add_argument("--feature_dir", type=Path, default=DATASET / "gazeformer_image_features")
    parser.add_argument("--train_backbone", action="store_true", help="Fine-tune the Gazeformer ResNet; default frozen")
    parser.add_argument("--backbone_weights", choices=("coco", "imagenet"), default="coco")
    parser.add_argument("--test_batch", type=int, default=1)
    parser.add_argument("--warmup_epoch", type=int, default=1)
    parser.add_argument("--no_eval_epoch", type=int, default=-1)
    parser.add_argument("--supervised_save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rl_baseline", choices=("mean", "leave_one_out"), default="mean")
    args = parser.parse_args()
    args._provided_hyperparams = [token.split("=")[0].lstrip("-").replace("-", "_")
                                  for token in sys.argv[1:] if token.startswith("--")]
    args.model = "gazeformer"
    if args.log_root is None:
        args.log_root = RUN
    if args.model == "gazeformer" and (min(args.nhead, args.num_encoder, args.num_decoder, args.text_dim, args.max_text_length) < 1
            or args.hidden_dim < 4 or args.hidden_dim % 4 or args.hidden_dim % args.nhead):
        parser.error("Invalid Gazeformer dimensions, heads or layer counts")
    if args.batch < 1 or args.test_batch < 1 or args.warmup_epoch < 0 or args.epoch < 1 or args.start_rl_epoch < 0 or args.max_length < 1 or not 0 <= args.min_length <= args.max_length:
        parser.error("Invalid batch or scanpath length")
    if args.start_rl_epoch < args.epoch and args.rl_sample_number < 2:
        parser.error("RL requires at least two samples for a baseline")
    if args.eval_repeat_num < 1 or args.workers < 0 or min(args.max_batches, args.eval_max_batches) < 0:
        parser.error("Invalid repeat, workers or batch limit")
    return args
