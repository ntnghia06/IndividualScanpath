# GazeformerISP-FT: PerGAZE to OSIE

Optimize all predictor parameters (transformer, heads, query/observer embeddings), while the COCO image backbone stays frozen in eval/no_grad mode. This follows the current PerGAZE GazeformerISP-FT protocol.

This branch transfers the **PerGAZE** architecture, not the separate original
OSIE architecture. Keep the full PerGAZE source and sibling shared package
`../gazeformer_transfer/`; both FT and ISP-S use that package. Base model files
and the ChenLSTM transfer branches are unchanged.

## Input and fixed free-viewing task

Defaults read `D:/PerScan/Experiments/osie/osie_fixations_test.json` and `images/`.
The JSON needs `name`, positive integer `subject`, `X`, `Y`, `T` (milliseconds),
`length`, and `split="test"`. No DINOv2 cache, boxes or external attention maps.

Supply the actual trained PerGAZE Gazeformer checkpoint via `--checkpoint`.
It must use geometry-v3 (24x32 grid, 769 actions), 768D SentenceTransformer
features and separate ta/tp/air observer identities. Current online checkpoints
include the frozen COCO backbone. Compatible head-only checkpoints initialize
the original pretrained COCO backbone before strict loading (internet needed
once for weights). Architecture dimensions/max_length come from the checkpoint.

Unlike ChenLSTM, there is **no task-head index**. All OSIE records use the same
768D vector encoding this instruction:

> You are allowed to view visual stimuli or scenes without any specific task or instruction.

`prepare_text.py` uses `sentence-transformers/stsb-roberta-base-v2` once and saves
`tasks`, `vectors`, and `encoder` metadata in a safe NPZ. Existing valid files are
reused without importing/downloading the text encoder. `--overwrite` explicitly
regenerates the file. The text encoder is not finetuned. This is a **new transfer
text archive**, so the original PerGAZE task NPZ is neither needed nor modified.
The adapted run records both the source text hash (when present) and the new
free-viewing archive hash. Resume/reevaluation require the same new archive.
A `.npy` pickle dictionary or a PerGAZE NPZ without encoder metadata is not accepted
by this preparation/validation path; use the provided text utility.

## Support and adaptation

K=5, seed=10 selects exactly five scanpaths per OSIE observer **from test**:
75 support records and 975 remaining evaluation records for the provided data.
Selection is identical to ChenLSTM transfer at the same K/seed/JSON and stable
under input reordering. Selected (image, observer) records are excluded from
evaluation, but different observers on the same image can occur on both sides.
This is not an image-disjoint benchmark. Train/validation OSIE files are unused.

Append identities `osie:1` ... `osie:15`; default `--init mean` initializes new
rows from the mean of all source observer embeddings. `--init random` uses
seeded normal noise with std=1.0, matching GazeformerISP-S. Existing row values
are copied exactly at initialization. In ISP-S their gradients are masked and
an invariant is checked after every epoch; zero weight decay is mandatory.

Default settings: **3 supervised epochs, LR=1e-5, batch=1**, gradient clip=12.5,
weight decay=0. No RL. Image preprocessing matches PerGAZE: ToTensor, resize to
768x1024, ImageNet normalize; frozen online COCO ResNet produces a 24x32 feature
grid. Targets and metrics retain the checkpoint's **512x352** scanpath frame,
not the original OSIE model's 512x384 frame. Target X/Y are scaled using the
actual image size, and T is converted to seconds internally.

The run evaluates a baseline before adaptation and the remaining test after each
epoch, with fixed evaluation RNG. Best uses the harmonic mean of ScanMatch
with/without duration on that remainder, matching ChenLSTM transfer. It is
model-selected performance on remaining OSIE test, not an untouched test estimate.

## Run from D:/PerScan

```powershell
conda activate cs117

# Optional: validate inputs/save support and remainder without loading weights/text.
python src/IndividualScanpath/PerGAZE/Transfer/GazeformerISP-FT/src/train.py --prepare_only

# Prepare free-viewing text once (can run this in Colab instead).
python src/IndividualScanpath/PerGAZE/Transfer/GazeformerISP-FT/src/prepare_text.py --output Experiments/osie/gazeformer_freeview_embeddings.npz --device cuda:0

# Supply your actual trained source checkpoint.
python src/IndividualScanpath/PerGAZE/Transfer/GazeformerISP-FT/src/train.py --checkpoint "D:/path/to/pergaze_gazeformer/checkpoints/best.pth" --device cuda:0 --gpu_ids 0 --k 5 --seed 10

# Resume by repeating that train command with --resume; --epoch may extend a run.
python src/IndividualScanpath/PerGAZE/Transfer/GazeformerISP-FT/src/test.py --device cuda:0 --gpu_ids 0

python -m unittest discover -s src/IndividualScanpath/PerGAZE/Transfer/GazeformerISP-FT/tests -v
```

Override `--data_root`, `--test_file`, `--img_dir`, `--text_embeddings`, `--prompt`
and `--log_root` as needed. `--workers 0` is Windows-safe. Batch limits
`--max_batches 1 --eval_max_batches 1` are only for functional checks; reports
flag incomplete runs. Separate output directories are required for predictor FT
and embedding-only ISP-S; their adapted checkpoints cannot resume/evaluate as
the other mode. Resume verifies source/text hashes, selection and hyperparameters.

## Colab

Open `kaggle_gazeformer_ft_osie_colab.ipynb` and select a GPU runtime.
Configure DATA (OSIE on Drive), CHECKPOINT/BASE_RUN (trained PerGAZE run), TEXT
(new free-viewing NPZ) and RUN (separate Drive output directory). The default
source run is `MyDrive/PerGAZE/pergaze_gazeformer_run_online`.

If the new code has not been pushed to GitHub, upload both `GazeformerISP-FT/src/`
and `gazeformer_transfer/` under `MyDrive/PerGAZE/transfer_code/Transfer/`, or edit
TRANSFER_SOURCE. The notebook clones the full repo and overlays those source
folders. It prepares support and text, trains, supports resume/reevaluation,
streams/copies logs, and optionally downloads a ZIP. Inputs and run outputs
stay on Drive. FT and ISP-S can share the same free-viewing NPZ.

## Outputs

Default directory: `runs/osie_predictor_k5_seed10/` inside this branch.

- `support.json`, `test_remaining.json`, `selection.json`: source hash, K/seed,
  selected IDs and raw records.
- `checkpoints/checkpoint.pth`, `checkpoints/best.pth`: model, optimizer, expanded
  subject map, transfer/text provenance, configuration, report and RNG.
- `report.json`: baseline, history, best metrics/predictions, retrieval and protocol.
- `predictions_best.json`, `predictions_last.json`: X/Y in original image pixels,
  T in milliseconds; `scanpath` retains 512x352 model coordinates and seconds.
- Reevaluation appends to report and saves `predictions_reevaluation.json`.

Tests cover actual OSIE split counts/consistency with ChenLSTM, text preparation,
geometry, original transformer/head gradient flow, and both training modes with
resume/reevaluation using a small network. They verify frozen backbone/buffers,
ISP-S preservation of all other weights, new embedding updates and support exclusion.
