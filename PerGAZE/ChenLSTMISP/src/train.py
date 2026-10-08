"""Supervised and ScanMatch policy-gradient training across all conditions."""
import time
import sys
from itertools import islice
from tqdm import tqdm
from pathlib import Path

import torch

from dataset.dataset import collate_func
from dataset.schema import split_counts
from models.loss import supervised_loss, duration_log_prob
from models.sampling import sample_scanpaths
from opts import parse_opt
from runtime import (setup, dataset, loader, model, move, forward, evaluate, write_json,
                     load_checkpoint, restore_config)
from utils.evaluation import Metrics
from schedule import learning_rate_factor
from training_settings import restore_training_settings
from parallel import model_state_dict, load_model_state, restore_cuda_rng


def rl_loss(network, batch, args, metrics):
    # Eval mode returns probabilities and disables dropout, but retains gradients.
    network.eval()
    prediction = forward(network, batch)
    rewards, log_probabilities = [], []
    for _ in range(args.rl_sample_number):
        paths, action_logp, times, active, duration_mask = sample_scanpaths(prediction, args.min_length)
        rewards.append(torch.tensor([metrics.reward(gt, path) for gt, path in zip(batch["fix_vectors"], paths)],
                                    device=batch["images"].device))
        duration_logp = duration_log_prob(times.detach(), prediction)
        log_probabilities.append((action_logp * active).sum(-1) / active.sum()
                                 + (duration_logp * duration_mask).sum(-1) / duration_mask.sum())
    rewards = torch.stack(rewards)
    baseline = (rewards.mean(0, keepdim=True) if args.rl_baseline == "mean" else
                (rewards.sum(0, keepdim=True) - rewards) / (args.rl_sample_number - 1))
    advantage = rewards - baseline
    return -(torch.stack(log_probabilities) * advantage.detach()).sum(), rewards.mean()


def smoke_test(args, records, manifest, device):
    args.pretrained = False  # Smoke execution never needs a weight download.
    data = dataset(args, records, manifest, "train")
    indices = [next(i for i, (_, row) in enumerate(data.records) if row["condition"] == c)
               for c in ("present", "absent", "vqa")]
    network = model(args, manifest, device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    checks = []
    for index in indices:
        smoke_batch = max(args.batch, len(args.gpu_ids), 1)
        batch = move(collate_func([data[index] for _ in range(smoke_batch)]), device)
        network.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = forward(network, batch)
        loss, _, _ = supervised_loss(prediction, batch)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite smoke loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in network.parameters()):
            raise ValueError("Nonfinite gradient")
        if args.clip > 0:
            torch.nn.utils.clip_grad_norm_(network.parameters(), args.clip)
        optimizer.step()
        network.eval()
        with torch.no_grad():
            inference = forward(network, batch)
            paths, *_ = sample_scanpaths(inference, args.min_length, greedy=True)
        checks.append({"condition": batch["metadata"][0]["condition"], "loss": float(loss.detach()),
                       "actions_shape": list(prediction["actions"].shape), "predicted_length": len(paths[0]),
                       "guidance_max": float(batch["attention_maps"].max())})
        print(checks[-1], flush=True)
    optimizer.zero_grad(set_to_none=True)
    rl, reward = rl_loss(network, batch, args, Metrics())
    if not torch.isfinite(rl):
        raise ValueError("Nonfinite RL loss")
    rl.backward()
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in network.parameters()):
        raise ValueError("Nonfinite RL gradient")
    result = {"device": str(device), "gpu_ids": args.gpu_ids, "conditions": checks, "rl_loss": float(rl.detach()), "rl_reward": float(reward),
              "note": "Smoke uses random weights and a bounded sequence length; it is not a trained checkpoint."}
    write_json(args.log_root / "smoke_test.json", result)
    print("Smoke forward/backward/inference and policy-gradient checks passed", flush=True)


def main():
    args = parse_opt()
    records, manifest, device = setup(args)
    args.log_root.mkdir(parents=True, exist_ok=True)
    if args.smoke_test:
        print({"records": len(records), "subjects": len(manifest["subjects"]), "splits": split_counts(records, manifest),
               "device": str(device), "gpu_ids": args.gpu_ids,
               "data_parallel": len(args.gpu_ids) > 1, "global_batch": args.batch}, flush=True)
        smoke_test(args, records, manifest, device)
        return
    checkpoint_path = args.checkpoint or args.log_root / "checkpoints/checkpoint.pth"
    checkpoint = load_checkpoint(checkpoint_path, device) if args.resume else None
    if checkpoint and "scheduler" not in checkpoint:
        raise ValueError("Checkpoint predates the original warmup/decay schedule; start a new run")
    if checkpoint:
        restore_training_settings(args, checkpoint["config"])
        manifest = restore_config(args, checkpoint, manifest)
    elif checkpoint_path.exists():
        raise FileExistsError(f"Existing checkpoint: use --resume or a new --log_root: {checkpoint_path}")
    print({"records": len(records), "subjects": len(manifest["subjects"]), "splits": split_counts(records, manifest),
           "device": str(device), "gpu_ids": args.gpu_ids,
           "data_parallel": len(args.gpu_ids) > 1, "global_batch": args.batch}, flush=True)
    print({"max_length": args.max_length,
           "train_batch": args.batch, "validation_batch": args.test_batch}, flush=True)
    network = model(args, manifest, device, pretrained=False if checkpoint else None)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start, best = 0, -float("inf")
    if checkpoint:
        load_model_state(network, checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best = checkpoint["epoch"] + 1, checkpoint["best_metric"]
        torch.set_rng_state(checkpoint["rng_cpu"].cpu())
        if device.type == "cuda" and checkpoint.get("rng_cuda"):
            restore_cuda_rng(checkpoint["rng_cuda"])
    write_json(args.log_root / "manifest.json", manifest)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.log_root / "hparams.json", config)
    train = loader(args, dataset(args, records, manifest, "train"), shuffle=True)
    validation = loader(args, dataset(args, records, manifest, "validation"), evaluation=True)
    steps = min(len(train), args.max_batches) if args.max_batches else len(train)
    if checkpoint and checkpoint["batches_per_epoch"] != steps:
        raise ValueError("Batch size or batch limit changed the LR schedule; start a new run")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda iteration: learning_rate_factor(iteration, args, steps))
    if checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
        for group, rate in zip(optimizer.param_groups, scheduler.get_last_lr()):
            group["lr"] = rate
    metrics = Metrics()
    for epoch in range(start, args.epoch):
        started, total, seen = time.monotonic(), 0., 0
        reinforcement = epoch >= args.start_rl_epoch
        conditions = {"present": 0, "absent": 0, "vqa": 0}
        phase = "RL" if reinforcement else "Supervised"
        with tqdm(total=steps, desc=f"Epoch {epoch + 1}/{args.epoch} | {phase}",
                  unit="batch", dynamic_ncols=True, mininterval=1, file=sys.stdout) as progress:
            for raw_batch in islice(train, steps):
                batch = move(raw_batch, device)
                optimizer.zero_grad(set_to_none=True)
                if reinforcement:
                    loss, _ = rl_loss(network, batch, args, metrics)
                else:
                    network.train()
                    loss, _, _ = supervised_loss(forward(network, batch), batch, args.lambda_1)
                if not torch.isfinite(loss):
                    raise ValueError(f"Nonfinite loss at epoch {epoch}, batch {progress.n}")
                loss.backward()
                if args.clip > 0:
                    torch.nn.utils.clip_grad_norm_(network.parameters(), args.clip, error_if_nonfinite=True)
                elif any(p.grad is not None and not torch.isfinite(p.grad).all() for p in network.parameters()):
                    raise ValueError("Nonfinite gradient")
                optimizer.step()
                scheduler.step()
                total += float(loss.detach()) * len(raw_batch["metadata"])
                seen += len(raw_batch["metadata"])
                for info in raw_batch["metadata"]:
                    conditions[info["condition"]] += 1
                progress.set_postfix(loss=f"{total / seen:.4f}",
                                     lr=f"{optimizer.param_groups[0]['lr']:.2e}", refresh=False)
                progress.update(1)
        summary, score, improved = None, None, False
        if epoch > args.no_eval_epoch:
            summary, _ = evaluate(network, validation, args, device, args.eval_max_batches,
                                  description=f"Epoch {epoch + 1}/{args.epoch} | Validation")
            values = summary["overall"]["metrics"]
            a, b = (values[key]["mean"] for key in ("ScanMatch_without_duration", "ScanMatch_with_duration"))
            score = 2 * a * b / (a + b) if a + b else 0.
            improved = score > best
            best = max(best, score)
        state = {"model": model_state_dict(network), "optimizer": optimizer.state_dict(), "epoch": epoch,
                 "best_metric": best, "manifest": manifest, "config": config,
                 "scheduler": scheduler.state_dict(), "batches_per_epoch": steps,
                 "rng_cpu": torch.get_rng_state(),
                 "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, checkpoint_path)
        if improved:
            torch.save(state, checkpoint_path.parent / "best.pth")
        if args.supervised_save and epoch == args.start_rl_epoch - 1:
            torch.save(state, checkpoint_path.parent / "supervised.pth")
        report = {"epoch": epoch + 1, "stage": "RL" if reinforcement else "supervised",
                  "loss": total / seen, "samples": seen, "conditions": conditions,
                  "seconds": time.monotonic() - started, "validation": summary,
                  "validation_score": score, "learning_rate": optimizer.param_groups[0]["lr"]}
        write_json(args.log_root / f"epoch_{epoch + 1:03d}.json", report)
        print(report, flush=True)


if __name__ == "__main__":
    main()
