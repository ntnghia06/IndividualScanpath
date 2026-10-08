"""Evaluate the saved checkpoint with its original split and subject mapping."""
from opts import parse_opt
from parallel import load_model_state
from runtime import (setup, dataset, loader, model, evaluate, write_json, load_checkpoint,
                     restore_config)


def main():
    args = parse_opt("Evaluate PerGAZE")
    records, manifest, device = setup(args)
    path = args.checkpoint or args.log_root / "checkpoints/best.pth"
    checkpoint = load_checkpoint(path, device)
    manifest = restore_config(args, checkpoint, manifest)
    network = model(args, manifest, device, pretrained=False)
    load_model_state(network, checkpoint["model"])
    summary, rows = evaluate(network, loader(args, dataset(args, records, manifest, args.split), evaluation=True),
                             args, device, args.max_batches, description=f"Evaluate {args.split}")
    write_json(args.log_root / f"evaluation_{args.split}.json", summary)
    write_json(args.log_root / f"predictions_{args.split}.json", rows)
    print({"overall": summary["overall"], "retrieval_by_condition": summary["retrieval"]["by_condition"]}, flush=True)


if __name__ == "__main__":
    main()
