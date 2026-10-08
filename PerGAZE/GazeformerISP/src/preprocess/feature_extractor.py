"""Precompute frozen COCO ResNet50 image features and sentence embeddings."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataset.schema import read_records
from dataset.features import cache_path, FEATURE_FORMAT, load_features
from dataset.text import task_text
from geometry import IMAGE_SIZE, FEATURE_GRID
from opts import DATASET


class ImageInputs(Dataset):
    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, output, relative = self.entries[index]
        with Image.open(path) as image:
            width, height = image.size
            tensor = T.functional.to_tensor(image.convert("RGB"))
        tensor = T.Resize(IMAGE_SIZE)(tensor)
        tensor = T.Normalize([.485, .456, .406], [.229, .224, .225])(tensor)
        return tensor, width, height, str(output), relative


def extract_images(rows, args):
    from dataset.schema import image_path
    entries, seen = {}, set()
    for row in rows:
        output, relative = cache_path(args.feature_dir, args.img_dir, row)
        if relative in seen:
            continue
        seen.add(relative)
        if output.is_file() and not args.overwrite:
            load_features(output, relative)
            continue
        entries[relative] = (image_path(args.img_dir, row), output, relative)
    print(f"Images to extract: {len(entries)}", flush=True)
    if not entries:
        return
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
    body = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1).backbone.body
    backbone = torch.nn.Sequential(*[getattr(body, key) for key in
        ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4")])
    del body
    device = torch.device(args.device)
    backbone.requires_grad_(False).eval().to(device)
    ids = args.gpu_ids or (list(range(torch.cuda.device_count())) if device.type == "cuda" else [])
    if len(ids) > 1:
        backbone = torch.nn.DataParallel(backbone, device_ids=ids)
    batches = DataLoader(ImageInputs(list(entries.values())), batch_size=args.batch,
                         num_workers=args.workers, pin_memory=device.type == "cuda")
    args.feature_dir.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for images, widths, heights, outputs, relatives in tqdm(batches, desc="COCO image features", unit="batch"):
            features = backbone(images.to(device))
            if features.shape[1:] != (2048, *FEATURE_GRID):
                raise ValueError(f"Unexpected ResNet feature grid: {features.shape}")
            if not torch.isfinite(features).all():
                raise FloatingPointError("Nonfinite extracted image features")
            features = features.cpu().float()
            for feature, width, height, output, relative in zip(features, widths, heights, outputs, relatives):
                target = Path(output)
                temporary = target.with_suffix(".tmp")
                torch.save({"format": FEATURE_FORMAT, "source": relative, "features": feature,
                            "width": int(width), "height": int(height)}, temporary)
                temporary.replace(target)


def extract_text(rows, args):
    from sentence_transformers import SentenceTransformer
    tasks = sorted({task_text(row) for row in rows})
    if args.output.is_file() and not args.overwrite:
        with np.load(args.output, allow_pickle=False) as archive:
            if ("encoder" in archive and archive["vectors"].shape == (len(tasks), 768)
                    and archive["tasks"].tolist() == tasks
                    and str(archive["encoder"].item()) == args.lm_model
                    and np.isfinite(archive["vectors"]).all()):
                print("Reusing complete sentence embedding archive", flush=True)
                return
        raise ValueError("Existing text archive differs; use --overwrite or another --output")
    encoder = SentenceTransformer(args.lm_model, device=args.device).eval()
    vectors = encoder.encode(tasks, batch_size=32, show_progress_bar=True).astype(np.float32)
    if vectors.shape != (len(tasks), 768) or not np.isfinite(vectors).all():
        raise ValueError("Expected finite 768-dimensional sentence embeddings")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, tasks=np.asarray(tasks), vectors=vectors, encoder=np.asarray(args.lm_model))
    print(f"Saved sentence embeddings: {vectors.shape}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files", type=Path, nargs="+")
    parser.add_argument("--train_file", "--data_file", dest="train_file", type=Path, default=DATASET / "train.json")
    parser.add_argument("--val_file", type=Path, default=DATASET / "test_seen.json")
    parser.add_argument("--img_dir", type=Path, default=DATASET / "images")
    parser.add_argument("--feature_dir", type=Path, default=DATASET / "gazeformer_image_features")
    parser.add_argument("--output", type=Path, default=DATASET / "gazeformer_task_embeddings.npz")
    parser.add_argument("--mode", choices=("all", "images", "text"), default="all")
    parser.add_argument("--lm_model", default="sentence-transformers/stsb-roberta-base-v2")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gpu_ids", type=int, nargs="+")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--storage_dtype", choices=("float32",), default="float32")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch < 1 or args.workers < 0:
        parser.error("Invalid batch/workers")
    # Include every split so unseen questions can be encoded by the frozen LM.
    files = args.files or [p for p in (args.train_file, args.val_file,
        args.train_file.parent / "support.json", args.train_file.parent / "test_unseen.json") if p.is_file()]
    rows = [row for path in files for row in read_records(path)]
    if args.mode in ("all", "images"):
        extract_images(rows, args)
    if args.mode in ("all", "text"):
        extract_text(rows, args)


if __name__ == "__main__":
    main()
