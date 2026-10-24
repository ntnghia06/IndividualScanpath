# PerGAZE transfer to COCO-Freeview

Dataset-specific counterparts of `../OSIE`, with four model branches and separate
COCO-FV subject identities, checkpoint modes, source hashes and output directories.
No OSIE or base model source is changed. These scripts load a trained **PerGAZE**
checkpoint and adapt to COCO-Freeview; an OSIE-adapted checkpoint is not a source
checkpoint for this protocol.

| Branch | Trainable parameters | Default LR |
|---|---|---|
| ChenLSTMISP-FT | Full model; active branch receives gradients | 1e-5 |
| ChenLSTMISP-S | New COCO-FV subject embedding rows only | 1e-3 |
| GazeformerISP-FT | Predictor; COCO image backbone stays frozen | 1e-5 |
| GazeformerISP-S | New COCO-FV subject embedding rows only | 1e-3 |

## Data and support

Default input: `D:/PerScan/Experiments/coco_freeview/COCOFreeView_fixations_test.json`
and the neighboring flat `images/` directory. Records require `condition="freeview"`,
`split="test"`, `name`, positive integer `subject`, X/Y/T and consistent `length`.
X/Y use original image pixels (the supplied stimuli are 1680x1050), and T is in
milliseconds. Loaders scale coordinates by each image's actual width and height.
No DINOv2 files, boxes or external attention maps are used.

The 6,462 test records cover 648 images and 10 observers with original IDs
101, 102, 103, 104, 105, 107, 108, 109, 110 and 116. IDs are retained in metadata
and mapped to appended model rows under `cocofv:<id>`, distinct from ta/tp/air
and OSIE observers. Default initialization uses the mean source observer embedding.

Default K=5, seed=10 samples exactly five records per observer: **50 support and
6,412 remaining evaluation records**. All four branches use the same selection.
Sampling is stable under JSON reordering. Duplicate (image, subject) records,
non-freeview conditions and K values leaving no evaluation records are rejected.
Selected records are excluded from evaluation, but other observers viewing the
same image can occur on both sides. Train/validation JSONs are not read.

Three supervised adaptation epochs run by default. Best epoch uses the harmonic
mean of both ScanMatch metrics on the remaining test; this is model-selected
performance on that remainder. Support/remainder, config, hashes, predictions,
metrics and last/best checkpoint are saved in each branch's own `runs/` folder.
Resume restores model/optimizer/RNG and verifies selection, data, mode and settings.

## Local use from D:/PerScan

```powershell
conda activate cs117

# Choose one branch; each train.py supports split preparation without a checkpoint.
python src/IndividualScanpath/PerGAZE/Transfer/COCO-FV/ChenLSTMISP-FT/src/train.py --prepare_only

# ChenLSTM: zero guidance, default --task_head -1 reuses the PerGAZE VQA branch.
python src/IndividualScanpath/PerGAZE/Transfer/COCO-FV/ChenLSTMISP-FT/src/train.py --checkpoint "D:/path/to/chenlstm/checkpoints/best.pth" --device cuda:0 --gpu_ids 0

# Gazeformer: create the free-viewing sentence NPZ once, shared by FT and ISP-S.
python src/IndividualScanpath/PerGAZE/Transfer/COCO-FV/GazeformerISP-FT/src/prepare_text.py --output Experiments/coco_freeview/gazeformer_freeview_embeddings.npz --device cuda:0
python src/IndividualScanpath/PerGAZE/Transfer/COCO-FV/GazeformerISP-FT/src/train.py --checkpoint "D:/path/to/gazeformer/checkpoints/best.pth" --device cuda:0 --gpu_ids 0
```

Replace `-FT` with `-S` for embedding-only adaptation. Use `--log_root` for a new
experiment. Repeat the original train command with `--resume` to continue; keep
the original PerGAZE source checkpoint. `test.py` reevaluates adapted best.pth
on the same support-excluded remainder. Each branch has detailed instructions.

Gazeformer receives a fixed 768D SentenceTransformer free-viewing vector, not a
task-head index. The original PerGAZE text archive is not needed. Text is encoded
once, and its encoder/prompt/hash are checked. ChenLSTM keeps 320x240/40x30;
Gazeformer keeps its source checkpoint's 512x352 scanpath frame and 24x32 grid.
Prediction X/Y are exported in original image pixels, T in milliseconds.

## Colab

Each model folder includes `kaggle_*_cocofv_colab.ipynb`. Select a GPU, mount Drive,
and set DATA to `MyDrive/PerGAZE/coco_freeview`, CHECKPOINT to the trained PerGAZE
model and RUN to a separate COCO-FV output directory. Notebooks prepare the split,
create/reuse Gazeformer NPZ when needed, train, resume and optionally reevaluate.

Code is expected at `PerGAZE/Transfer/COCO-FV/` inside the cloned repository.
If it is not yet on GitHub, use the notebook's Drive source override:

- ChenLSTM: `MyDrive/PerGAZE/transfer_code/COCO-FV/<branch>/src/`.
- Gazeformer: upload both `<branch>/src/` and `gazeformer_transfer/` under
  `MyDrive/PerGAZE/transfer_code/COCO-FV/`.

Inputs/checkpoints remain on Drive. The full base repository is required for the
PerGAZE architecture. Do not mix OSIE and COCO-FV run folders or adapted checkpoints.
