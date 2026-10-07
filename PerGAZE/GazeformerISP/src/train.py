"""Supervised and ScanMatch policy-gradient training across all conditions."""
import time
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
        log_probabilities.append((action_logp * active).sum(-1) / active.sum(-1).clamp_min(1)
                                 + (duration_logp * duration_mask).sum(-1) / duration_mask.sum(-1).clamp_min(1))
    rewards = torch.stack(rewards)
    advantage = rewards - (rewards.sum(0, keepdim=True) - rewards) / (args.rl_sample_number - 1)
    return -(torch.stack(log_probabilities) * advantage.detach()).mean(), rewards.mean()


def smoke_test(args, records, manifest, device):
    args.pretrained = False  # Smoke execution never needs a weight download.
    data = dataset(args, records, manifest, "all")
    indices = [next(i for i, (_, row) in enumerate(data.records) if row["condition"] == c)
               for c in ("present", "absent", "vqa")]
    network = model(args, manifest, device)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr)
    checks = []
    for index in indices:
        batch = move(collate_func([data[index]]), device)
        network.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = forward(network, batch)
        loss, _, _ = supervised_loss(prediction, batch)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite smoke loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in network.parameters()):
            raise ValueError("Nonfinite gradient")
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
    result = {"device": str(device), "conditions": checks, "rl_loss": float(rl.detach()), "rl_reward": float(reward),
              "note": "Smoke uses random weights and a bounded sequence length; it is not a trained checkpoint."}
    write_json(args.log_root / "smoke_test.json", result)
    print("Smoke forward/backward/inference and policy-gradient checks passed", flush=True)


def main():
    args = parse_opt()
    records, manifest, device = setup(args)
    args.log_root.mkdir(parents=True, exist_ok=True)
    print({"records": len(records), "subjects": len(manifest["subjects"]), "splits": split_counts(records, manifest),
           "device": str(device)}, flush=True)
    if args.smoke_test:
        smoke_test(args, records, manifest, device)
        return
    checkpoint_path = args.checkpoint or args.log_root / "checkpoints/checkpoint.pth"
    checkpoint = load_checkpoint(checkpoint_path, device) if args.resume else None
    if checkpoint:
        manifest = restore_config(args, checkpoint, manifest)
    elif checkpoint_path.exists():
        raise FileExistsError(f"Existing checkpoint: use --resume or a new --log_root: {checkpoint_path}")
    network = model(args, manifest, device, pretrained=False if checkpoint else None)
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start, best = 0, -float("inf")
    if checkpoint:
        network.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best = checkpoint["epoch"] + 1, checkpoint["best_metric"]
        torch.set_rng_state(checkpoint["rng_cpu"].cpu())
        if device.type == "cuda" and checkpoint.get("rng_cuda"):
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["rng_cuda"]])
    write_json(args.log_root / "manifest.json", manifest)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    write_json(args.log_root / "hparams.json", config)
    train = loader(args, dataset(args, records, manifest, "train"), shuffle=True)
    validation = loader(args, dataset(args, records, manifest, "validation"))
    metrics = Metrics()
    for epoch in range(start, args.epoch):
        started, total, seen = time.monotonic(), 0., 0
        reinforcement = epoch >= args.start_rl_epoch
        for group in optimizer.param_groups:
            group["lr"] = args.lr * (args.rl_lr_initial_decay if reinforcement else 1.)
        conditions = {"present": 0, "absent": 0, "vqa": 0}
        for i, raw_batch in enumerate(train):
            if args.max_batches and i >= args.max_batches:
                break
            batch = move(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            if reinforcement:
                loss, _ = rl_loss(network, batch, args, metrics)
            else:
                network.train()
                loss, _, _ = supervised_loss(forward(network, batch), batch, args.lambda_1)
            if not torch.isfinite(loss):
                raise ValueError(f"Nonfinite loss at epoch {epoch}, batch {i}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(network.parameters(), args.clip, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(raw_batch["metadata"])
            seen += len(raw_batch["metadata"])
            for info in raw_batch["metadata"]:
                conditions[info["condition"]] += 1
            if (i + 1) % 50 == 0:
                print(f"epoch {epoch + 1}, batch {i + 1}/{len(train)}, loss {total / seen:.4f}", flush=True)
        summary, _ = evaluate(network, validation, args, device, args.eval_max_batches)
        values = summary["overall"]["metrics"]
        a, b = (values[key]["mean"] for key in ("ScanMatch_without_duration", "ScanMatch_with_duration"))
        score = 2 * a * b / (a + b) if a + b else 0.
        improved = score > best
        best = max(best, score)
        state = {"model": network.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                 "best_metric": best, "manifest": manifest, "config": config,
                 "rng_cpu": torch.get_rng_state(),
                 "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else []}
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(state, checkpoint_path)
        if improved:
            torch.save(state, checkpoint_path.parent / "best.pth")
        report = {"epoch": epoch + 1, "stage": "RL" if reinforcement else "supervised",
                  "loss": total / seen, "samples": seen, "conditions": conditions,
                  "seconds": time.monotonic() - started, "validation": summary}
        write_json(args.log_root / f"epoch_{epoch + 1:03d}.json", report)
        print(report, flush=True)


if __name__ == "__main__":
    main()
