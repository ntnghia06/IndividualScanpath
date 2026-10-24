"""Re-evaluate the saved COCO-FV remainder, never the selected support scanpaths."""
import bootstrap
import copy
import gc
import json
import sys
from pathlib import Path

import torch

from adaptation import IDENTITY_SCHEME, require_full_checkpoint
from fewshot import sha256, score, evaluate_seeded, make_network, validate_base_checkpoint
from geometry import GEOMETRY
from options import parse_args
from cocofv_data import prepare_split, SubjectAdaptationDataset
from parallel import resolve_devices, wrap_model, load_model_state
from runtime import load_checkpoint, loader, write_json


def main():
    args = parse_args(evaluation=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(("cuda:0" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if args.device == "auto" and args.gpu_ids and torch.cuda.is_available():
        device = torch.device("cuda", args.gpu_ids[0])
    device, args.gpu_ids = resolve_devices(device, args.gpu_ids, torch.cuda.device_count())
    checkpoint = load_checkpoint(args.checkpoint, torch.device("cpu"))
    validate_base_checkpoint(checkpoint)
    manifest = copy.deepcopy(checkpoint["manifest"])
    require_full_checkpoint(manifest)
    if (manifest.get("geometry") != GEOMETRY
            or manifest.get("subject_identity_scheme") != IDENTITY_SCHEME
            or manifest.get("transfer_protocol") != "cocofv_test_support_record_split_v1"):
        raise ValueError("Expected an COCO-FV transfer full-FT checkpoint")
    if sha256(args.test_file) != checkpoint["provenance"]["cocofv_test"]:
        raise ValueError("COCO-FV source test JSON differs from the adapted run")
    saved_args = checkpoint["adaptation_config"]
    for key in ("k", "seed", "eval_seed", "task_head", "greedy", "eval_repeat_num"):
        setattr(args, key, saved_args[key])
    if not any(a == "--log_root" or a.startswith("--log_root=") for a in sys.argv[1:]):
        args.log_root = args.checkpoint.resolve().parents[1]
    _, tests, selection = prepare_split(args)
    if selection != checkpoint["selection"]:
        raise ValueError("Support selection differs from the adapted checkpoint")
    config = checkpoint["config"]
    args.min_length = config["min_length"]
    network = make_network(config, manifest)
    load_model_state(network, checkpoint["model"])
    network.requires_grad_(False)
    report_path = args.log_root / "report.json"
    report = (json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file()
              else checkpoint.get("report", {}))
    del checkpoint
    gc.collect()
    network = wrap_model(network, device, args.gpu_ids)
    manifest["record_splits"] = ["validation"] * len(tests)
    data = SubjectAdaptationDataset(tests, manifest, args.img_dir, split="validation",
        max_length=config["max_length"], blur_sigma=config["blur_sigma"], task_head=args.task_head)
    summary, predictions = evaluate_seeded(network, loader(args, data, evaluation=True), args, device,
                                           "COCO-FV remaining test reevaluation")
    report["reevaluation"] = {"checkpoint": str(args.checkpoint), "score": score(summary),
        "metrics": summary, "predictions": predictions, "test_samples": len(tests),
        "full_remaining_test_evaluation": args.eval_max_batches == 0}
    write_json(args.log_root / "predictions_reevaluation.json", predictions)
    write_json(report_path, report)
    print({"score": score(summary), "report": str(report_path)}, flush=True)


if __name__ == "__main__":
    main()
