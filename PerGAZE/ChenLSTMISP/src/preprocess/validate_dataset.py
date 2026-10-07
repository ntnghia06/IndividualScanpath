"""Read every record, decode every image and load every required attention map."""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from dataset.dataset import load_guidance
from dataset.schema import image_path, split_counts
from opts import parse_opt
from runtime import setup, write_json


def main():
    args = parse_opt("Validate all PerGAZE records and assets")
    records, manifest, _ = setup(args)
    images, attention, errors, clipped, zero_attention = {}, set(), [], Counter(), set()
    checked_boxes = set()
    for i, row in enumerate(records):
        try:
            path = image_path(args.img_dir, row)
            if path not in images:
                with Image.open(path) as image:
                    image.load()
                    images[path] = image.size
            width, height = images[path]
            coords = np.asarray([row["X"], row["Y"], row["T"]], dtype=float).T
            if not np.isfinite(coords).all() or (coords[:, 2] <= 0).any():
                raise ValueError("Invalid fixation/duration")
            clipped[row["condition"]] += int(((coords[:, 0] < 0) | (coords[:, 0] >= width)
                                            | (coords[:, 1] < 0) | (coords[:, 1] >= height)).sum())
            box_key = None
            if row["condition"] == "present":
                box_key = (str(path), tuple(np.asarray(row["bbox"]).ravel()))
            if ((box_key is not None and box_key not in checked_boxes)
                    or (row["condition"] == "vqa" and row["qid"] not in attention)):
                guidance = load_guidance(row, args.att_dir, (width, height))
                if box_key is not None:
                    checked_boxes.add(box_key)
                if row["condition"] == "vqa":
                    attention.add(row["qid"])
                    if not guidance.max():
                        zero_attention.add(row["qid"])
        except (ValueError, OSError, KeyError) as error:
            errors.append({"index": i, "name": row["name"], "error": str(error)})
        if (i + 1) % 5000 == 0:
            print(f"Validated {i + 1}/{len(records)} records", flush=True)
    report = {"records": len(records), "conditions": dict(Counter(r["condition"] for r in records)),
              "subjects": len(manifest["subjects"]), "decoded_images": len(images),
              "loaded_attention_maps": len(attention), "clipped_fixations": dict(clipped),
              "zero_attention_qids": sorted(zero_attention), "split_counts": split_counts(records, manifest),
              "error_count": len(errors), "errors": errors}
    write_json(args.log_root / "validation_report.json", report)
    write_json(args.log_root / "manifest.json", manifest)
    print({k: v for k, v in report.items() if k != "errors"}, flush=True)
    if errors:
        raise SystemExit(f"Validation failed: {len(errors)} errors; see validation_report.json")


if __name__ == "__main__":
    main()
