"""Re-evaluate an adapted checkpoint and append to its single JSON report."""
import bootstrap
import copy
import gc
import json
from argparse import Namespace
from pathlib import Path

import torch

from dataset.schema import read_records
from adaptation_dataset import SubjectAdaptationDataset
from adaptation import IDENTITY_SCHEME
from fewshot import sha256, score, evaluate_seeded
from geometry import GEOMETRY
from models.gazeformer.model import GazeformerISP
from options import parse_args
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
    manifest = copy.deepcopy(checkpoint["manifest"])
    if manifest.get("geometry") != GEOMETRY or "adaptation_subjects" not in manifest:
        raise ValueError("Expected a GazeformerISP-S adapted checkpoint")
    if manifest.get("adaptation_sampling_protocol") != "original_sigma2_v1":
        raise ValueError("Old adaptation sampling protocol; start a new run")
    if manifest.get("subject_identity_scheme") != IDENTITY_SCHEME:
        raise ValueError("This adaptation checkpoint pooled TA/TP observers; run finetuning again with separate identities")
    if sha256(args.test_file) != checkpoint["provenance"]["test_unseen"]:
        raise ValueError("test_unseen differs from the adapted run")
    config = checkpoint["config"]
    saved_text = config.get("text_embeddings")
    if args.text_embeddings is None and saved_text:
        args.text_embeddings = Path(saved_text)
    if bool(saved_text) != bool(args.text_embeddings):
        raise ValueError("Use the checkpoint's original text backend")
    if args.text_embeddings and sha256(args.text_embeddings) != checkpoint["provenance"]["text_embeddings"]:
        raise ValueError("Text archive differs from the adapted run")
    args.min_length = config["min_length"]
    args.eval_seed = checkpoint["adaptation_config"]["eval_seed"]
    args.greedy = checkpoint["adaptation_config"]["greedy"]
    args.eval_repeat_num = checkpoint["adaptation_config"]["eval_repeat_num"]
    network = GazeformerISP(Namespace(**config), manifest, pretrained=not any(k.startswith("backbone.") for k in checkpoint["model"]))
    load_model_state(network, checkpoint["model"])
    network.requires_grad_(False)
    report_path = args.log_root / "report.json"
    report = (json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file() else
              checkpoint.get("report", {}))
    del checkpoint
    gc.collect()
    network = wrap_model(network, device, args.gpu_ids)
    tests = read_records(args.test_file)
    manifest["record_splits"] = ["validation"] * len(tests)
    data = SubjectAdaptationDataset(tests, manifest, args.img_dir, args.att_dir, split="validation",
        max_length=config["max_length"], blur_sigma=config["blur_sigma"],
        max_text_length=config["max_text_length"], text_embeddings=args.text_embeddings)
    summary, predictions = evaluate_seeded(network, loader(args, data, evaluation=True), args, device,
                                          "Adapted checkpoint evaluation on unseen")
    report["reevaluation"] = {"checkpoint": str(args.checkpoint), "score": score(summary),
                              "metrics": summary, "predictions": predictions,
                              "full_unseen_evaluation": args.eval_max_batches == 0}
    write_json(report_path, report)
    print({"score": score(summary), "report": str(report_path)}, flush=True)


if __name__ == "__main__":
    main()
