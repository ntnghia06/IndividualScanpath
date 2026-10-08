import bootstrap
import gc
import hashlib
import json
import random
import sys
import time
from argparse import Namespace
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from adaptation import (select_support, build_manifest, expand_subject_embeddings,
                        freeze_for_adaptation, IDENTITY_SCHEME)
from dataset.schema import read_records
from adaptation_dataset import SubjectAdaptationDataset
from geometry import GEOMETRY
from models.gazeformer.model import GazeformerISP
from models.loss import supervised_loss
from options import parse_args
from parallel import (resolve_devices, wrap_model, unwrap_model, model_state_dict,
                      load_model_state, restore_cuda_rng)
from runtime import write_json, loader, move, forward, evaluate, load_checkpoint


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def score(summary):
    values = summary["overall"]["metrics"]
    a = values["ScanMatch_without_duration"]["mean"]
    b = values["ScanMatch_with_duration"]["mean"]
    return 2 * a * b / (a + b) if a + b else 0.


def evaluate_seeded(network, batches, args, device, description):
    with torch.random.fork_rng(devices=args.gpu_ids):
        torch.manual_seed(args.eval_seed)
        torch.cuda.manual_seed_all(args.eval_seed)
        return evaluate(network, batches, args, device, args.eval_max_batches, description=description)


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested if args.device == "auto" else args.device)
    if args.device == "auto" and args.gpu_ids and torch.cuda.is_available():
        device = torch.device("cuda", args.gpu_ids[0])
    device, args.gpu_ids = resolve_devices(device, args.gpu_ids, torch.cuda.device_count())
    args.log_root.mkdir(parents=True, exist_ok=True)
    last_path = args.log_root / "checkpoints/checkpoint.pth"
    if not args.resume and last_path.exists():
        raise FileExistsError("Adaptation checkpoint exists; use --resume or a new --log_root")

    checkpoint_hash = sha256(args.checkpoint)
    base = load_checkpoint(args.checkpoint, torch.device("cpu"))
    if base["manifest"].get("geometry") != GEOMETRY or base["config"].get("model") != "gazeformer":
        raise ValueError("Expected a compatible PerGAZE GazeformerISP base checkpoint")
    config = dict(base["config"])
    args.min_length = config["min_length"]
    saved_base_manifest = base["manifest"]
    old_subjects = dict(saved_base_manifest["subjects"])
    support_pool = read_records(args.support_file)
    tests = read_records(args.test_file)
    chosen, selection = select_support(support_pool, args.k, args.seed)
    manifest = build_manifest(saved_base_manifest, chosen, tests)
    manifest["metric_protocol"] = "original_mm_retrieval_duration_v2"
    if set(old_subjects) & set(manifest["adaptation_subjects"]):
        raise ValueError("Expected unseen support subjects; a support observer already exists in the base model")
    provenance = {"base_checkpoint": checkpoint_hash, "support": sha256(args.support_file),
                  "test_unseen": sha256(args.test_file)}
    saved_text = config.get("text_embeddings")
    if args.text_embeddings is None and saved_text:
        args.text_embeddings = Path(saved_text)
    if bool(saved_text) != bool(args.text_embeddings):
        raise ValueError("Use the base checkpoint's original text backend")
    if args.text_embeddings:
        if sha256(args.text_embeddings) != saved_base_manifest["text_embeddings_sha256"]:
            raise ValueError("Text embedding archive differs from the base checkpoint")
        provenance["text_embeddings"] = sha256(args.text_embeddings)

    network = GazeformerISP(Namespace(**config), saved_base_manifest, pretrained=False)
    load_model_state(network, base["model"])
    expand_subject_embeddings(network, old_subjects, manifest["subjects"], args.init)
    del base, support_pool
    gc.collect()
    network = network.to(device)
    adaptive_indices, gradient_hook = freeze_for_adaptation(network, manifest)
    reference_embedding = network.subject_embed.weight.detach().clone()
    network = wrap_model(network, device, args.gpu_ids)
    # Zero decay and fresh optimizer state preserve old rows whose gradients are zero.
    optimizer = torch.optim.Adam([unwrap_model(network).subject_embed.weight], lr=args.lr, weight_decay=0.)

    records = chosen + tests
    def make_data(split):
        return SubjectAdaptationDataset(records, manifest, args.img_dir, args.att_dir, split=split,
            max_length=config["max_length"], blur_sigma=config["blur_sigma"],
            max_text_length=config["max_text_length"], text_embeddings=args.text_embeddings, feature_dir=args.feature_dir)
    train_loader = loader(args, make_data("train"), shuffle=True)
    test_loader = loader(args, make_data("validation"), evaluation=True)
    resolved = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report_path = args.log_root / "report.json"
    report = {"config": resolved, "base_model_config": config, "provenance": provenance,
        "selection": selection, "subjects": manifest["subjects"], "geometry": GEOMETRY,
        "subject_identity_scheme": IDENTITY_SCHEME,
        "initialization_sources": "TA/TP use their own base population mean if available, otherwise legacy coco mean; VQA uses air mean",
        "support_samples": len(chosen), "test_samples": len(tests),
        "full_selected_support_training": args.max_batches == 0,
        "full_unseen_evaluation": args.eval_max_batches == 0,
        "adaptive_subjects": manifest["adaptation_subjects"],
        "trainable_parameter": "subject_embed.weight (selected subject rows only)",
        "trainable_elements": len(adaptive_indices) * config["embedding_dim"],
        "selection_split": "test_unseen", "selection_metric": "harmonic mean of both ScanMatch metrics",
        "evaluation_protocol": "test_unseen selects the best adaptation epoch as requested; this is not an untouched test estimate",
        "history": [], "best_epoch": None, "best_score": None, "best_metrics": None}
    start, best = 0, -float("inf")
    if args.resume:
        state = load_checkpoint(last_path, device)
        if state["manifest"].get("metric_protocol") != "original_mm_retrieval_duration_v2":
            raise ValueError("Adaptation eval protocol changed; start a new run")
        if state["manifest"].get("adaptation_sampling_protocol") != "original_sigma2_v1":
            raise ValueError("Old adaptation sampling protocol; start a new run")
        if state["manifest"].get("subject_identity_scheme") != IDENTITY_SCHEME:
            raise ValueError("Old adaptation pooled TA/TP; start a new run with separate observer identities")
        if state["provenance"] != provenance or state["selection"] != selection:
            raise ValueError("Resume requires identical checkpoint, support/test files, k and seed")
        for key in ("k", "seed", "eval_seed", "lr", "batch", "init", "eval_repeat_num", "greedy", "max_batches", "eval_max_batches"):
            if state["adaptation_config"][key] != resolved[key]:
                raise ValueError(f"Cannot change {key} while resuming adaptation")
        load_model_state(network, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_cpu"].cpu())
        restore_cuda_rng(state["rng_cuda"])
        start, best = state["epoch"] + 1, state["best_score"]
        report = state.get("report") or json.loads(report_path.read_text(encoding="utf-8"))
        del state
    else:
        baseline, _ = evaluate_seeded(network, test_loader, args, device, "Unseen baseline before adaptation")
        report["baseline"] = {"score": score(baseline), "metrics": baseline}
        write_json(report_path, report)
    print({"k": args.k, "seed": args.seed, "subjects": manifest["adaptation_subjects"],
           "support_samples": len(chosen), "test_samples": len(tests),
           "gpu_ids": args.gpu_ids, "only_subject_embedding": True}, flush=True)
    for epoch in range(start, args.epoch):
        began, total_loss, seen = time.monotonic(), 0., 0
        network.eval()  # Keep frozen network buffers/dropout fixed while retaining autograd.
        steps = min(len(train_loader), args.max_batches) if args.max_batches else len(train_loader)
        with tqdm(total=steps, desc=f"Subject embedding FT {epoch + 1}/{args.epoch}",
                  unit="batch", file=sys.stdout, dynamic_ncols=True) as progress:
            for raw in islice(train_loader, steps):
                batch = move(raw, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = forward(network, batch)
                prediction["actions"] = prediction["all_actions_prob"].clamp_min(1e-12).log()
                loss, _, _ = supervised_loss(prediction, batch, config["lambda_1"])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite adaptation loss")
                loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_([unwrap_model(network).subject_embed.weight], args.clip, error_if_nonfinite=True)
                optimizer.step()
                total_loss += float(loss.detach()) * len(raw["metadata"])
                seen += len(raw["metadata"])
                progress.set_postfix(loss=f"{total_loss / seen:.4f}")
                progress.update(1)
        if not torch.equal(unwrap_model(network).subject_embed.weight[:len(old_subjects)].detach(),
                           reference_embedding[:len(old_subjects)]):
            raise RuntimeError("An original subject embedding changed")
        metrics, predictions = evaluate_seeded(network, test_loader, args, device,
                                               f"Unseen evaluation {epoch + 1}/{args.epoch}")
        current = score(metrics)
        improved = current > best
        best = max(best, current)
        report["history"].append({"epoch": epoch + 1, "support_loss": total_loss / seen,
            "support_samples_processed": seen, "score": current, "metrics": metrics,
            "seconds": time.monotonic() - began})
        if improved:
            report.update(best_epoch=epoch + 1, best_score=current, best_metrics=metrics,
                          best_predictions=predictions)
        state = {"model": model_state_dict(network), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "best_score": best, "manifest": manifest, "config": config,
            "adaptation_config": resolved, "provenance": provenance, "selection": selection,
            "report": report,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        last_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, last_path)
        if improved:
            torch.save(state, last_path.parent / "best.pth")
        write_json(report_path, report)
        print({"epoch": epoch + 1, "score": current, "best_epoch": report["best_epoch"], "best_score": best}, flush=True)
    gradient_hook.remove()
    print("Report:", report_path, flush=True)
