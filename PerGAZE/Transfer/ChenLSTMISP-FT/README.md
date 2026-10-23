# PerGAZE ChenLSTMISP-FT transfer to OSIE

Based on `../../ChenLSTMISP-FT`. Load a trained PerGAZE ChenLSTMISP checkpoint,
add OSIE observer embeddings, then finetune the full model with supervised
action/duration loss. This branch imports the existing PerGAZE architecture,
loss, sampling and metric code; keep the full repository. No base source files
are modified, no RL is run, and no OSIE train/validation files are used.

## Inputs and protocol

- `Experiments/osie/osie_fixations_test.json`: OSIE test records, including
  `name`, `subject` (positive integer), `X`, `Y`, `T` (milliseconds), `length`,
  and `split="test"`. Coordinates refer to the original image dimensions.
- `Experiments/osie/images/`: flat image directory. No DINOv2 features, text
  embeddings, boxes or external attention maps are needed.
- `--checkpoint`: compatible PerGAZE ChenLSTMISP checkpoint containing `model`,
  `config`, `manifest`, separate ta/tp/air observer identities and the current
  18 COCO object heads plus VQA head. Supply your actual trained checkpoint;
  the script never automatically selects a functional-test checkpoint.

Default K=5 selects exactly five scanpaths **per OSIE observer** from the test
JSON using seed=10. Sampling is stable under JSON reordering. The original
1,050 samples yield 75 support and 975 remaining evaluation samples. Support
and evaluation have no overlapping (image, observer) records. Images can still
occur on both sides for different observers: this is a record split, not an
image-disjoint benchmark. Duplicate image/observer records and K values leaving
no evaluation sample for an observer are rejected.

New identities `osie:1` ... `osie:15` are appended to the base subject mapping;
source rows are never overwritten during initialization. Default initialization
is the mean of **all** source observer embeddings; `--init random` uses seeded
normal noise with std=0.1. Every parameter is unfrozen and passed to Adam.
Normal train mode enables dropout and BatchNorm updates. Parameters belonging
to inactive task branches or unused observer rows can have no gradient; this
does not freeze them. The fixed OSIE task head defaults to `--task_head -1`
(the transferred VQA branch), with zero spatial guidance. `--task_head 0..17`
selects an existing COCO object branch instead. There is no pretrained
free-viewing head, so the head choice is an explicit transfer configuration.

Defaults match full FT: 3 supervised epochs, LR=1e-5, weight decay=0,
train/eval batch=1, gradient clipping=12.5. Geometry/max_length and loss settings
come from the checkpoint: images/scanpaths 320x240 and action grid 40x30 + STOP.
All epochs evaluate the remaining test using the same evaluation seed. Best is
selected by harmonic mean of ScanMatch with/without duration on that remainder,
matching the source FT branch. This is model-selected performance on remaining
OSIE test, **not an untouched held-out test estimate**.

## Run from D:/PerScan

```powershell
conda activate cs117

# Validate inputs and save the reproducible support/remainder split; no checkpoint needed.
python src/IndividualScanpath/PerGAZE/Transfer/ChenLSTMISP-FT/src/train.py --prepare_only

# Supply the actual trained PerGAZE checkpoint path.
python src/IndividualScanpath/PerGAZE/Transfer/ChenLSTMISP-FT/src/train.py --checkpoint "D:/path/to/pergaze/checkpoints/best.pth" --device cuda:0 --gpu_ids 0 --k 5 --seed 10

# Resume: retain the same base checkpoint, data, K, seed and optimization settings.
# Repeat the training command above with --resume; --epoch can extend the run.

# Reevaluate best.pth on the saved selection's remainder.
python src/IndividualScanpath/PerGAZE/Transfer/ChenLSTMISP-FT/src/test.py --device cuda:0 --gpu_ids 0

python -m unittest discover -s src/IndividualScanpath/PerGAZE/Transfer/ChenLSTMISP-FT/tests -v
```

Use `--data_root`, or `--test_file` and `--img_dir`, to override data locations.
Use a different `--log_root` for each experiment or head choice. `--workers 0`
is Windows-safe. `--max_batches 1 --eval_max_batches 1` limits functional checks;
reports flag incomplete runs. No pretrained backbone download is needed.

## Google Colab (one GPU)

Open `kaggle_chenlstm_ft_osie_colab.ipynb` and select a GPU runtime. The notebook
follows the existing PerGAZE `ChenLSTMISP-FT/kaggle_chenlstm_ft_colab.ipynb`, with
OSIE support selection and transfer-specific arguments.

Upload OSIE's `images/` and `osie_fixations_test.json` to `MyDrive/PerGAZE/osie/`,
or change `DATA`. Set `BASE_RUN`/`CHECKPOINT` to the trained PerGAZE model and
`RUN` to a separate OSIE output directory. Defaults are K=5, seed=10, 3 epochs,
LR=1e-5, batch=1 and `TASK_HEAD=-1`. Data and checkpoints are read from Drive;
outputs are written directly to Drive after each epoch.

The notebook clones the full repository for the base architecture. If this
new transfer branch has not been pushed to GitHub, upload this folder to
`MyDrive/PerGAZE/transfer_code/ChenLSTMISP-FT/` (including `src/`), or edit
`TRANSFER_SOURCE`. Its source overrides the cloned transfer files without copying
datasets or run checkpoints. If GitHub already contains the new code, that
override is optional.

The notebook prepares the split before a new run, streams subprocess logs,
supports `RESUME=True`, and includes optional best-checkpoint reevaluation and
ZIP download. The selected head is a transferred branch, not a text input.

## Outputs

Default: `runs/osie_full_ft_k5_seed10/` inside this branch.

- `support.json`, `test_remaining.json`, `selection.json`: selected raw records,
  source JSON SHA-256, seed, K and per-observer selection IDs.
- `checkpoints/checkpoint.pth`, `checkpoints/best.pth`: last/best model,
  optimizer, configuration, provenance, expanded subject map and RNG state.
- `report.json`: baseline, epoch losses/metrics, retrieval, best predictions,
  selected head and protocol. Reevaluation appends its results.
- `predictions_best.json`, `predictions_last.json`: predictions with `X`, `Y`
  in original image pixels and `T` in milliseconds. The `scanpath` field retains
  model-space 320x240 coordinates and durations in seconds, used for metrics.

Resume verifies checkpoint/data hashes, selected records and adaptation settings.
PerGAZE-only FT/ISP-S checkpoints cannot be resumed as an OSIE transfer run.
Metrics operate in the same 320x240 coordinate frame as the source ChenLSTM.
