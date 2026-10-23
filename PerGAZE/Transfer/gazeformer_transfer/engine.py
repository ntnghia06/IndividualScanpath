import gc
import copy
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

from .adaptation import build_manifest, expand_embeddings, configure, require_mode, MODES
from .data import prepare_split, OSIEDataset
from .text import sha256, load_vector, ENCODER
from .options import parse_args
from geometry import GEOMETRY, SCANPATH_SIZE
from models.gazeformer.model import GazeformerISP
from models.loss import supervised_loss
from parallel import (resolve_devices, wrap_model, unwrap_model, model_state_dict,
                      load_model_state, restore_cuda_rng)
from runtime import write_json, loader, move, forward, evaluate, load_checkpoint


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
            row["X"] = [p[0] / SCANPATH_SIZE[1] * width for p in row["scanpath"]]
            row["Y"] = [p[1] / SCANPATH_SIZE[0] * height for p in row["scanpath"]]
            row["T"] = [p[2] * 1000 for p in row["scanpath"]]
            row["length"] = len(row["scanpath"])
        return summary, predictions



def validate_base_checkpoint(checkpoint):
    manifest, config = checkpoint["manifest"], checkpoint["config"]
    if config.get("model") != "gazeformer" or manifest.get("geometry") != GEOMETRY:
        raise ValueError("Expected a geometry-v3 PerGAZE Gazeformer checkpoint (24x32, 769 actions)")
    if config.get("text_dim") != 768 or not config.get("text_embeddings"):
        raise ValueError("Expected the pretrained 768D sentence-embedding text backend")
    subjects = manifest["subjects"]
    if not subjects or sorted(subjects.values()) != list(range(len(subjects))):
        raise ValueError("Checkpoint subjects must use consecutive indices")
    if any(key.split(":", 1)[0] not in ("ta", "tp", "air", "osie") for key in subjects):
        raise ValueError("Checkpoint uses incompatible pooled subject identities")


def make_network(config, manifest, state):
    # Current checkpoints carry the frozen backbone; compatible head-only checkpoints
    # initialize the same pretrained COCO backbone before strict predictor loading.
    has_backbone = any(key.removeprefix("module.").startswith("backbone.") for key in state)
    return GazeformerISP(Namespace(**config), manifest, pretrained=not has_backbone)


def device_for(args):
    requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(requested if args.device == "auto" else args.device)
    if args.device == "auto" and args.gpu_ids and torch.cuda.is_available():
        device = torch.device("cuda", args.gpu_ids[0])
    device, args.gpu_ids = resolve_devices(device, args.gpu_ids, torch.cuda.device_count())
    return device


def train_main(mode, branch):
    args = parse_args(mode, branch)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = device_for(args)
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
    vector = load_vector(args.text_embeddings, args.prompt)
    checkpoint_hash = sha256(args.checkpoint)
    base = load_checkpoint(args.checkpoint, torch.device("cpu"))
    validate_base_checkpoint(base)
    base_config = dict(base["config"])
    config = dict(base_config)
    args.min_length = config["min_length"]
    config["text_embeddings"] = str(args.text_embeddings)
    saved_base_manifest = base["manifest"]
    old_subjects = dict(saved_base_manifest["subjects"])
    if any(key.startswith("osie:") for key in old_subjects):
        raise ValueError("Training needs a PerGAZE source checkpoint; use --resume for an OSIE run")
    manifest = build_manifest(saved_base_manifest, chosen, tests, mode)
    manifest["geometry"] = GEOMETRY.copy()
    manifest["transfer_protocol"] = "osie_test_support_record_split_v1"
    manifest["source_text_embeddings_sha256"] = saved_base_manifest.get("text_embeddings_sha256")
    manifest["text_embeddings_sha256"] = sha256(args.text_embeddings)
    manifest["transfer_prompt"] = args.prompt
    manifest["transfer_text_encoder"] = ENCODER
    manifest["metric_protocol"] = "original_mm_retrieval_duration_v2"
    if set(old_subjects) & set(manifest["adaptation_subjects"]):
        raise ValueError("Expected unseen support subjects; a support observer already exists in the base model")
    provenance = {"base_checkpoint": checkpoint_hash, "osie_test": sha256(args.test_file),
                  "freeview_text": sha256(args.text_embeddings)}
    network = make_network(config, saved_base_manifest, base["model"])
    load_model_state(network, base["model"])
    expand_embeddings(network, old_subjects, manifest["subjects"], args.init)
    del base
    gc.collect()
    network = network.to(device)
    parameters, trainable_elements, gradient_hook = configure(network, manifest, mode)
    reference_embedding = network.subject_embed.weight.detach().clone()
    network = wrap_model(network, device, args.gpu_ids)
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)

    records = chosen + tests
    def make_data(split):
        return OSIEDataset(records, manifest, args.img_dir, vector, split=split,
            max_length=config["max_length"], blur_sigma=config["blur_sigma"],
            max_text_length=config["max_text_length"])
    train_loader = loader(args, make_data("train"), shuffle=True)
    test_loader = loader(args, make_data("validation"), evaluation=True)
    resolved = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    report_path = args.log_root / "report.json"
    report = {"config": resolved, "base_model_config": base_config, "provenance": provenance,
        "selection": selection, "subjects": manifest["subjects"], "geometry": GEOMETRY,
        "subject_identity_scheme": "osie_separate_v1",
        "initialization_sources": "Mean of all PerGAZE subject embeddings; seeded random std=1.0 if requested",
        "support_samples": len(chosen), "test_samples": len(tests),
        "full_selected_support_training": args.max_batches == 0,
        "full_unseen_evaluation": args.eval_max_batches == 0,
        "prompt": args.prompt, "text_encoder": ENCODER, "frozen_backbone": True, "split_unit": "image_observer_record",
        "adaptive_subjects": manifest["adaptation_subjects"],
        "model": "OSIE-Transfer-GazeformerISP", "mode": mode,
        "trainable_parameter": "new OSIE subject rows" if mode == "embedding" else "all predictor parameters; backbone frozen",
        "trainable_elements": trainable_elements,
        "selection_split": "osie_test_remaining", "selection_metric": "harmonic mean of both ScanMatch metrics",
        "evaluation_protocol": "Support is sampled per observer from OSIE test; selected records are excluded from evaluation. Remaining test selects the best epoch, so this is model-selected performance. Other observers on the same image can occur in support and evaluation.",
        "adaptation_mode": MODES[mode], "history": [], "best_epoch": None, "best_score": None, "best_metrics": None}
    start, best = 0, -float("inf")
    if args.resume:
        state = load_checkpoint(last_path, torch.device("cpu"))
        validate_base_checkpoint(state)
        if state["manifest"].get("geometry") != GEOMETRY:
            raise ValueError("Resume Gazeformer geometry differs")
        require_mode(state["manifest"], mode)
        if state["provenance"] != provenance or state["selection"] != selection:
            raise ValueError("Resume requires the same base checkpoint, OSIE test JSON, k and seed")
        for key in ("prompt", "k", "seed", "eval_seed", "lr", "weight_decay", "batch", "test_batch", "clip", "init", "eval_repeat_num", "greedy", "max_batches", "eval_max_batches"):
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
        baseline, _ = evaluate_seeded(network, test_loader, args, device, "OSIE remainder before finetuning")
        report["baseline"] = {"score": score(baseline), "metrics": baseline}
        write_json(report_path, report)
    print({"k": args.k, "seed": args.seed, "subjects": manifest["adaptation_subjects"],
           "support_samples": len(chosen), "test_samples": len(tests),
           "gpu_ids": args.gpu_ids, "mode": mode, "frozen_backbone": True}, flush=True)
    for epoch in range(start, args.epoch):
        began, total_loss, seen = time.monotonic(), 0., 0
        if mode == "embedding":
            network.eval()  # Keep dropout and frozen buffers fixed; retain autograd for embeddings.
        else:
            network.train()
            unwrap_model(network).backbone.eval()
        steps = min(len(train_loader), args.max_batches) if args.max_batches else len(train_loader)
        with tqdm(total=steps, desc=f"Gazeformer {mode} FT {epoch + 1}/{args.epoch}",
                  unit="batch", file=sys.stdout, dynamic_ncols=True) as progress:
            for raw in islice(train_loader, steps):
                batch = move(raw, device)
                optimizer.zero_grad(set_to_none=True)
                prediction = forward(network, batch)
                if mode == "embedding":
                    prediction["actions"] = prediction["all_actions_prob"].clamp_min(1e-12).log()
                loss, _, _ = supervised_loss(prediction, batch, config["lambda_1"])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite adaptation loss")
                loss.backward()
                if mode == "embedding" and unwrap_model(network).subject_embed.weight.grad is None:
                    raise FloatingPointError("Missing subject embedding gradient")
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(parameters, args.clip, error_if_nonfinite=True)
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in network.parameters()):
                    raise FloatingPointError("Nonfinite full-model gradient")
                optimizer.step()
                total_loss += float(loss.detach()) * len(raw["metadata"])
                seen += len(raw["metadata"])
                progress.set_postfix(loss=f"{total_loss / seen:.4f}")
                progress.update(1)
        if mode == "embedding" and not torch.equal(
                unwrap_model(network).subject_embed.weight[:len(old_subjects)].detach(),
                reference_embedding[:len(old_subjects)]):
            raise RuntimeError("An original PerGAZE subject embedding changed")
        metrics, predictions = evaluate_seeded(network, test_loader, args, device,
                                               f"OSIE remainder {epoch + 1}/{args.epoch}")
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
    if gradient_hook is not None:
        gradient_hook.remove()
    print("Report:", report_path, flush=True)


def test_main(mode, branch):
    args = parse_args(mode, branch, evaluation=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = device_for(args)
    checkpoint = load_checkpoint(args.checkpoint, torch.device("cpu"))
    validate_base_checkpoint(checkpoint)
    manifest = copy.deepcopy(checkpoint["manifest"])
    require_mode(manifest, mode)
    if manifest.get("geometry") != GEOMETRY:
        raise ValueError("OSIE Gazeformer geometry differs")
    if sha256(args.test_file) != checkpoint["provenance"]["osie_test"]:
        raise ValueError("OSIE source test JSON differs from the adapted run")
    saved_args = checkpoint["adaptation_config"]
    for key in ("k", "seed", "eval_seed", "prompt", "greedy", "eval_repeat_num"):
        setattr(args, key, saved_args[key])
    if not any(a == "--log_root" or a.startswith("--log_root=") for a in sys.argv[1:]):
        args.log_root = args.checkpoint.resolve().parents[1]
    _, tests, selection = prepare_split(args)
    if selection != checkpoint["selection"]:
        raise ValueError("Support selection differs from the adapted checkpoint")
    config = checkpoint["config"]
    args.min_length = config["min_length"]
    vector = load_vector(args.text_embeddings, args.prompt)
    if sha256(args.text_embeddings) != checkpoint["provenance"]["freeview_text"]:
        raise ValueError("Free-viewing text archive differs from the adapted run")
    network = make_network(config, manifest, checkpoint["model"])
    load_model_state(network, checkpoint["model"])
    network.requires_grad_(False)
    report_path = args.log_root / "report.json"
    report = (json.loads(report_path.read_text(encoding="utf-8")) if report_path.is_file()
              else checkpoint.get("report", {}))
    del checkpoint
    gc.collect()
    network = wrap_model(network, device, args.gpu_ids)
    manifest["record_splits"] = ["validation"] * len(tests)
    data = OSIEDataset(tests, manifest, args.img_dir, vector, split="validation",
        max_length=config["max_length"], blur_sigma=config["blur_sigma"],
        max_text_length=config["max_text_length"])
    summary, predictions = evaluate_seeded(network, loader(args, data, evaluation=True), args, device,
                                           "OSIE remaining test reevaluation")
    report["reevaluation"] = {"checkpoint": str(args.checkpoint), "score": score(summary),
        "metrics": summary, "predictions": predictions, "test_samples": len(tests),
        "full_remaining_test_evaluation": args.eval_max_batches == 0}
    write_json(args.log_root / "predictions_reevaluation.json", predictions)
    write_json(report_path, report)
    print({"score": score(summary), "report": str(report_path)}, flush=True)
