import bootstrap
import gc
import hashlib
import json
import random
import sys
import time
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from adaptation import (build_manifest, expand_subject_embeddings,
                        freeze_for_adaptation, require_embedding_checkpoint, IDENTITY_SCHEME)
from cocofv_data import prepare_split
from cocofv_data import SubjectAdaptationDataset
from geometry import GEOMETRY
from models.baseline_attention import baseline
from dataset.tasks import HEAD_SCHEMA, OBJECT_TO_INDEX
from models.loss import duration_log_prob
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


def save_split(args, chosen, tests, selection):
    write_json(args.log_root / "support.json", chosen)
    write_json(args.log_root / "test_remaining.json", tests)
    write_json(args.log_root / "selection.json", {
        "source": str(args.test_file.resolve()), "source_sha256": sha256(args.test_file),
        "k": args.k, "seed": args.seed, "split_unit": "image_observer_record",
        "support_samples": len(chosen), "remaining_test_samples": len(tests),
        "selection": selection})


def score(summary):
    values = summary["overall"]["metrics"]
    a = values["ScanMatch_without_duration"]["mean"]
    b = values["ScanMatch_with_duration"]["mean"]
    return 2 * a * b / (a + b) if a + b else 0.


def evaluate_seeded(network, batches, args, device, description):
    with torch.random.fork_rng(devices=args.gpu_ids):
        torch.manual_seed(args.eval_seed)
        torch.cuda.manual_seed_all(args.eval_seed)
        summary, predictions = evaluate(network, batches, args, device, args.eval_max_batches, description=description)
        # The shared evaluator names PerGAZE conditions explicitly; all rows here are freeview.
        summary["by_condition"] = {"freeview": summary["overall"]}
        summary["retrieval"]["by_condition"] = {"freeview": summary["retrieval"]["overall"]}
        for row in predictions:
            width, height = row["original_size"]
            row["X"] = [p[0] / 320 * width for p in row["scanpath"]]
            row["Y"] = [p[1] / 240 * height for p in row["scanpath"]]
            row["T"] = [p[2] * 1000 for p in row["scanpath"]]
            row["length"] = len(row["scanpath"])
        return summary, predictions



def validate_base_checkpoint(checkpoint):
    manifest = checkpoint["manifest"]
    if (checkpoint["config"].get("model") != "chenlstm"
            or manifest.get("task_head_schema") != HEAD_SCHEMA
            or manifest.get("object_to_head") != OBJECT_TO_INDEX):
        raise ValueError("Expected ChenLSTMISP checkpoint with 18 TP/TA object heads and VQA branch")
    if any(key.split(":", 1)[0] not in ("ta", "tp", "air", "cocofv") for key in manifest["subjects"]):
        raise ValueError("Base checkpoint pools TA/TP observers; retrain with separate subject identities")


def make_network(config, manifest):
    # All weights, including ResNet, are restored from the base/adapted checkpoint.
    return baseline(convLSTM_length=config["max_length"], min_length=config["min_length"],
        subject_num=len(manifest["subjects"]), embedding_dim=config["embedding_dim"],
        action_map_num=config["action_map_num"], dropout=config["dropout"], pretrained=False)


def adaptation_loss(prediction, batch, duration_weight):
    action_mask = batch["action_masks"]
    action = -(batch["target_scanpaths"] * torch.log(prediction["all_actions_prob"] + 1e-7)
               * action_mask.unsqueeze(-1)).sum() / action_mask.sum()
    duration_mask = batch["duration_masks"]
    duration = -duration_log_prob(batch["durations"], prediction)[duration_mask == 1].sum() / duration_mask.sum()
    return action + duration_weight * duration

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

    chosen, tests, selection = prepare_split(args)
    if args.prepare_only:
        save_split(args, chosen, tests, selection)
        print({"support_samples": len(chosen), "remaining_test_samples": len(tests),
               "subjects": len(selection), "log_root": str(args.log_root)}, flush=True)
        return
    checkpoint_hash = sha256(args.checkpoint)
    base = load_checkpoint(args.checkpoint, torch.device("cpu"))
    validate_base_checkpoint(base)
    base_config = dict(base["config"])
    config = dict(base_config)
    args.min_length = config["min_length"]
    saved_base_manifest = base["manifest"]
    old_subjects = dict(saved_base_manifest["subjects"])
    if any(key.startswith("cocofv:") for key in old_subjects):
        raise ValueError("Training needs a PerGAZE source checkpoint; use --resume for an COCO-FV run")
    manifest = build_manifest(saved_base_manifest, chosen, tests)
    manifest["geometry"] = GEOMETRY.copy()
    manifest["transfer_protocol"] = "cocofv_test_support_record_split_v1"
    manifest["transfer_task_head"] = args.task_head
    manifest["metric_protocol"] = "original_mm_retrieval_duration_v2"
    if set(old_subjects) & set(manifest["adaptation_subjects"]):
        raise ValueError("Expected unseen support subjects; a support observer already exists in the base model")
    provenance = {"base_checkpoint": checkpoint_hash, "cocofv_test": sha256(args.test_file)}
    network = make_network(config, saved_base_manifest)
    load_model_state(network, base["model"])
    expand_subject_embeddings(network, old_subjects, manifest["subjects"], args.init)
    del base
    gc.collect()
    network = network.to(device)
    adaptive_indices, gradient_hook = freeze_for_adaptation(network, manifest)
    reference_embedding = network.subject_embedding.weight.detach().clone()
    trainable_elements = len(adaptive_indices) * config["embedding_dim"]
    network = wrap_model(network, device, args.gpu_ids)
    optimizer = torch.optim.Adam([unwrap_model(network).subject_embedding.weight], lr=args.lr, weight_decay=0.)

    records = chosen + tests
    def make_data(split):
        return SubjectAdaptationDataset(records, manifest, args.img_dir, args.att_dir, split=split,
            max_length=config["max_length"], blur_sigma=config["blur_sigma"], task_head=args.task_head)
    train_loader = loader(args, make_data("train"), shuffle=True)
    test_loader = loader(args, make_data("validation"), evaluation=True)
    resolved = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report_path = args.log_root / "report.json"
    report = {"dataset": "COCO-FV", "config": resolved, "base_model_config": base_config, "provenance": provenance,
        "selection": selection, "subjects": manifest["subjects"], "geometry": GEOMETRY,
        "subject_identity_scheme": IDENTITY_SCHEME,
        "initialization_sources": "Mean of all PerGAZE subject embeddings; seeded random std=0.1 if requested",
        "support_samples": len(chosen), "test_samples": len(tests),
        "full_selected_support_training": args.max_batches == 0,
        "full_unseen_evaluation": args.eval_max_batches == 0,
        "task_head": args.task_head, "guidance": "zero", "split_unit": "image_observer_record",
        "adaptive_subjects": manifest["adaptation_subjects"],
        "model": "COCO-FV-Transfer-ChenLSTMISP-S",
        "trainable_parameter": "subject_embedding.weight (new COCO-FV rows only)",
        "trainable_elements": trainable_elements,
        "selection_split": "cocofv_test_remaining", "selection_metric": "harmonic mean of both ScanMatch metrics",
        "evaluation_protocol": "Support is sampled per observer from COCO-FV test; selected records are excluded from evaluation. Remaining test selects the best epoch, so this is model-selected performance. Other observers on the same image can occur in support and evaluation.",
        "adaptation_mode": "cocofv_subject_embedding_v1", "history": [], "best_epoch": None, "best_score": None, "best_metrics": None}
    start, best = 0, -float("inf")
    if args.resume:
        state = load_checkpoint(last_path, torch.device("cpu"))
        validate_base_checkpoint(state)
        if state["manifest"].get("geometry") != GEOMETRY:
            raise ValueError("Resume geometry differs from ChenLSTMISP-S")
        require_embedding_checkpoint(state["manifest"])
        if (state["manifest"].get("transfer_protocol") != "cocofv_test_support_record_split_v1"
                or state["manifest"].get("transfer_task_head") != args.task_head):
            raise ValueError("Resume COCO-FV transfer protocol/head differs")
        if state["manifest"].get("metric_protocol") != "original_mm_retrieval_duration_v2":
            raise ValueError("Adaptation eval protocol changed; start a new run")
        if state["manifest"].get("adaptation_sampling_protocol") != "original_sigma2_v1":
            raise ValueError("Old adaptation sampling protocol; start a new run")
        if state["manifest"].get("subject_identity_scheme") != IDENTITY_SCHEME:
            raise ValueError("Resume COCO-FV subject identity scheme differs")
        if state["provenance"] != provenance or state["selection"] != selection:
            raise ValueError("Resume requires the same base checkpoint, COCO-FV test JSON, k and seed")
        for key in ("task_head", "k", "seed", "eval_seed", "lr", "weight_decay", "batch", "test_batch", "clip", "init", "eval_repeat_num", "greedy", "max_batches", "eval_max_batches"):
            if state["adaptation_config"][key] != resolved[key]:
                raise ValueError(f"Cannot change {key} while resuming adaptation")
        load_model_state(network, state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng_cpu"].cpu())
        restore_cuda_rng(state["rng_cuda"])
        start, best = state["epoch"] + 1, state["best_score"]
        report = state.get("report") or json.loads(report_path.read_text(encoding="utf-8"))
        report["config"] = resolved
        del state
    else:
        save_split(args, chosen, tests, selection)
        baseline, _ = evaluate_seeded(network, test_loader, args, device, "COCO-FV remainder before finetuning")
        report["baseline"] = {"score": score(baseline), "metrics": baseline}
        write_json(report_path, report)
    print({"k": args.k, "seed": args.seed, "subjects": manifest["adaptation_subjects"],
           "support_samples": len(chosen), "test_samples": len(tests),
           "gpu_ids": args.gpu_ids, "only_subject_embedding": True}, flush=True)
    for epoch in range(start, args.epoch):
        began, total_loss, seen = time.monotonic(), 0., 0
        network.eval()  # Keep frozen parameters, dropout and BatchNorm buffers unchanged; retain autograd.
        steps = min(len(train_loader), args.max_batches) if args.max_batches else len(train_loader)
        with tqdm(total=steps, desc=f"COCO-FV subject embedding FT {epoch + 1}/{args.epoch}",
                  unit="batch", file=sys.stdout, dynamic_ncols=True) as progress:
            for raw in islice(train_loader, steps):
                batch = move(raw, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = forward(network, batch)
                loss = adaptation_loss(prediction, batch, config["lambda_1"])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite adaptation loss")
                loss.backward()
                gradient = unwrap_model(network).subject_embedding.weight.grad
                if gradient is None or not torch.isfinite(gradient).all():
                    raise FloatingPointError("Missing/nonfinite subject embedding gradient")
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_([unwrap_model(network).subject_embedding.weight],
                                                  args.clip, error_if_nonfinite=True)
                optimizer.step()
                total_loss += float(loss.detach()) * len(raw["metadata"])
                seen += len(raw["metadata"])
                progress.set_postfix(loss=f"{total_loss / seen:.4f}")
                progress.update(1)
        if not torch.equal(unwrap_model(network).subject_embedding.weight[:len(old_subjects)].detach(),
                           reference_embedding[:len(old_subjects)]):
            raise RuntimeError("An original PerGAZE subject embedding changed")
        metrics, predictions = evaluate_seeded(network, test_loader, args, device,
                                               f"COCO-FV remainder {epoch + 1}/{args.epoch}")
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
            write_json(args.log_root / "predictions_best.json", predictions)
        write_json(args.log_root / "predictions_last.json", predictions)
        write_json(report_path, report)
        print({"epoch": epoch + 1, "score": current, "best_epoch": report["best_epoch"], "best_score": best}, flush=True)
    gradient_hook.remove()
    print("Report:", report_path, flush=True)
