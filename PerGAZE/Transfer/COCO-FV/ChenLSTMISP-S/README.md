# PerGAZE ChenLSTMISP-S transfer to COCO-FV

Embedding-only transfer based on `../../../ChenLSTMISP-S` and the COCO-FV data/selection
pipeline in `../ChenLSTMISP-FT`. Load a trained PerGAZE ChenLSTMISP model, append
new `cocofv:101` ... `cocofv:116` (10 identities) subject embeddings and adapt only those new rows.
The full repository is required for the sibling `../../../ChenLSTMISP/src` model,
sampling and metrics. This branch does not depend on transfer FT source files
at runtime and does not modify the base or FT implementations.

## Inputs and support selection

Defaults read `D:/PerScan/Experiments/coco_freeview/COCOFreeView_fixations_test.json` and the flat
`images/` directory next to it. The JSON uses `name`, positive integer `subject`,
`X`, `Y`, `T` (milliseconds), `length`, and `split="test"`. Coordinates use the
original image dimensions. No feature cache, text embeddings, boxes or external
attention maps are needed. Supply the actual trained PerGAZE checkpoint using
`--checkpoint`; compatible models have separate ta/tp/air subjects, 18 COCO
object branches and the VQA branch.

Exactly K records per subject are sampled from COCO-FV **test**, using a fixed seed.
Default K=5, seed=10 yields 50 support and 6,412 evaluation records. Selection is
identical to transfer FT for the same input/K/seed and stable under record
reordering. Selected (image, subject) records are removed from evaluation;
images can occur on both sides for different observers. This is a record split,
not an image-disjoint evaluation. Duplicate records and K values leaving no
evaluation records for a subject are rejected. COCO-FV train/validation are unused.

## Adaptation

- Only new COCO-FV embedding rows learn. Original subject rows use masked gradients
  and are checked after each epoch. Backbone, prediction heads and all other
  parameters are frozen. The model remains in eval mode even during optimization,
  preserving BatchNorm buffers and disabling dropout while retaining autograd.
- Adam optimizes only the embedding table, with zero weight decay and fresh
  optimizer state. Default LR=1e-3, 3 supervised epochs, batch=1, gradient clip=12.5.
  Nonzero `--weight_decay` is rejected. No RL stage.
- Default initialization is the mean of all source observer embeddings;
  `--init random` uses seeded normal noise with std=0.1. Source rows remain intact.
- Guidance is all zero. `--task_head -1` selects the frozen VQA branch as the
  default transfer configuration; 0..17 chooses a COCO object branch instead.
  There is no free-viewing head or textual task input. Choosing VQA does not
  establish that it is the best head for COCO-FV.
- Architecture/max_length come from the checkpoint. Images/scanpaths use the
  original ChenLSTM 320x240 frame and 40x30 action grid plus STOP. Action and
  duration losses follow ISP-S, using eval-mode probabilities with autograd.

Baseline and per-epoch evaluation use a fixed seed on remaining COCO-FV test. Best
epoch is chosen by harmonic mean of ScanMatch with/without duration on this
remainder, matching transfer FT. This is model-selected performance, not an
untouched held-out test estimate.

## Run from D:/PerScan

```powershell
conda activate cs117

# Validate images/JSON and save support/remainder without loading model weights.
python src/IndividualScanpath/PerGAZE/Transfer/COCO-FV/ChenLSTMISP-S/src/train.py --prepare_only

# Replace the checkpoint path with the trained PerGAZE model.
python src/IndividualScanpath/PerGAZE/Transfer/COCO-FV/ChenLSTMISP-S/src/train.py --checkpoint "D:/path/to/pergaze/checkpoints/best.pth" --device cuda:0 --gpu_ids 0 --k 5 --seed 10

# Repeat the training command with --resume to restore the last completed epoch.
# Keep the same original base checkpoint and settings; --epoch may extend a run.

python src/IndividualScanpath/PerGAZE/Transfer/COCO-FV/ChenLSTMISP-S/src/test.py --device cuda:0 --gpu_ids 0

python -m unittest discover -s src/IndividualScanpath/PerGAZE/Transfer/COCO-FV/ChenLSTMISP-S/tests -v
```

Override `--data_root`, or `--test_file` and `--img_dir`, for other input paths.
`--workers 0` is Windows-safe. `--max_batches 1 --eval_max_batches 1` enables a
limited functional check; reports mark incomplete training/evaluation. Different
settings/head choices should use separate `--log_root` directories. A full-FT
adapted checkpoint cannot resume or evaluate as an ISP-S transfer checkpoint.

## Colab

Open `kaggle_chenlstm_s_cocofv_colab.ipynb` and select a GPU runtime. Configure:

- `DATA`: default `MyDrive/PerGAZE/coco_freeview`, with test JSON and `images/`.
- `CHECKPOINT`/`BASE_RUN`: actual trained PerGAZE ChenLSTMISP checkpoint.
- `RUN`: separate COCO-FV ISP-S output directory on Drive.
- `TRANSFER_SOURCE`: optional `MyDrive/PerGAZE/transfer_code/COCO-FV/ChenLSTMISP-S`
  containing this branch's `src/`, if the new source is not yet on GitHub.

The notebook clones the full base repository, optionally overlays transfer code
from Drive, prepares support, streams subprocess logs and supports resume,
best-checkpoint reevaluation and optional ZIP download. Inputs stay on Drive;
checkpoints/reports persist there after each epoch. Default LR is 1e-3 and
weight decay must remain zero.

## Outputs

Default directory: `runs/cocofv_subject_ft_k5_seed10/` in this branch.

- `support.json`, `test_remaining.json`, `selection.json`: raw split records,
  selected IDs, K, seed and source JSON SHA-256.
- `checkpoints/checkpoint.pth`, `checkpoints/best.pth`: last/best model,
  embedding optimizer, configuration, subject map, provenance and RNG state.
- `report.json`: baseline, epoch losses/metrics, retrieval matrices, best
  predictions and explicit adaptation/head/evaluation protocol.
- `predictions_best.json`, `predictions_last.json`: X/Y in original image pixels
  and T in milliseconds. `scanpath` keeps model-space coordinates and seconds.
- Reevaluation appends to the report and writes `predictions_reevaluation.json`.

Resume checks source checkpoint/test hashes, selected records, adaptation mode,
head, seed, initialization, optimization and evaluation settings. Reevaluation
reconstructs the saved support selection and excludes it again.
