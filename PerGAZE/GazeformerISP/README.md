# PerGAZE GazeformerISP

Independent PerGAZE implementation with the same folder layout as the AiR and
COCO_Search18 branches. Run commands from `D:\PerScan`.

```text
GazeformerISP/
  bash/               train.sh, test.sh
  src/
    dataset/          JSON loader, subject mapping and explicit file membership
    models/           network, losses and scanpath sampling
    preprocess/       full dataset validation and optional text embedding extraction
    utils/            evaluation and original metric implementations
    opts.py
    runtime.py
    train.py
    test.py
  tests/
  requirements.txt
  train.ps1
  test.ps1
```

Training reads all records in `dataset/PerGAZED/dataset/train.json` (24,194
samples); validation reads all records in `test_seen.json` (2,659 samples).
Override with `--train_file` and `--val_file`; `--data_file` is an alias for
`--train_file`. There is no random split. File membership determines the split
for each record even when images or the original JSON split fields overlap.

`present` uses `images/TP/task/name` and JSON bbox `[x,y,width,height]`; `vqa`
uses `images/VQA/name` and `attention_reasoning/qid.npy`; `absent` uses
`images/TA/task/name` with zero auxiliary guidance. Flat TP/TA image directories
are also supported. Fixations use actual image dimensions and are clipped at
boundaries; `T` is milliseconds.
The explanation field `prediction` is not a training target.

COCO TP/TA share observer IDs, and AiR observers use a separate namespace.
All observer embeddings are defined from training subjects; validation rejects
observers absent from training. The manifest stores per-record file membership
and SHA-256 hashes of both input files. The full dataset currently has 23
training observers and the same 23 validation observers.

The network adapts the original GazeformerISP Transformer, observer-centric
integration and adaptive fixation heads. Images are resized to 1024x768 (width x height); online
ResNet50 features form a 32x24 token grid. Fixation logits are interpolated to
32x22 plus the stop action (705 actions in total). Nonzero bbox/attention guidance is normalized and
averaged with learned task attention. TA text is masked. Default text encoding
is learned mean word embeddings with a training-only vocabulary; pretrained
SentenceTransformer vectors are optional. This is an adaptation of the original
offline-feature architecture, not an exact benchmark replica.

Defaults: 6 encoder/6 decoder layers, hidden dim 512, 8 attention heads and
4 fixation heads. COCO Mask R-CNN backbone weights are downloaded on first use;
the backbone and its batch normalization are frozen unless `--train_backbone`
is set. `--backbone_weights imagenet` selects ImageNet weights instead.
Use compatible torch/torchvision versions.

```powershell
conda activate cs117
python src/IndividualScanpath/PerGAZE/GazeformerISP/src/preprocess/validate_dataset.py
python src/IndividualScanpath/PerGAZE/GazeformerISP/src/train.py
python src/IndividualScanpath/PerGAZE/GazeformerISP/src/test.py

# Short functional check on one real sample per condition, plus RL
python src/IndividualScanpath/PerGAZE/GazeformerISP/src/train.py --smoke_test --max_length 3 --rl_sample_number 2 --batch 1 --log_root src/IndividualScanpath/PerGAZE/GazeformerISP/runs/smoke

# Resume the default run or evaluate all records
python src/IndividualScanpath/PerGAZE/GazeformerISP/src/train.py --resume --epoch 30
python src/IndividualScanpath/PerGAZE/GazeformerISP/src/test.py --split validation

python -m unittest discover -s src/IndividualScanpath/PerGAZE/GazeformerISP/tests
```

Training hyperparameters default to `--hyperparam_preset air`, matching the
original AiR `opts.py`. `coco` matches COCO `opts.py`; `air_run` and `coco_run`
include overrides from their original `bash/train.sh`. CLI arguments override
the selected preset. Data files, subject mapping and the chosen image geometry
remain PerGAZE-specific. Batch counts individual scanpaths, whereas the original
loaders count groups of observers.

Both models use original one-epoch warmup, linear supervised LR decay and
linear RL LR decay with multiplier 0.1. Gaussian target blur is disabled.
Validation uses separate `--test_batch 1`, with the preset's `no_eval_epoch`.
RL defaults to the original all-sample mean baseline. `clip=-1` in Gazeformer
presets disables gradient clipping. Schedulers are saved in checkpoints.
`--device auto` selects CUDA if available. `--no-pretrained` uses random
backbone weights. `--workers 0` is Windows-safe. Batch limits remain available
for functional checks; zero processes the full split.

Default output is this model's own `runs/` directory. It contains manifests,
hyperparameters, validation/epoch reports, `checkpoints/checkpoint.pth`,
`checkpoints/best.pth`, evaluation metrics and predicted scanpaths. Test/resume
reuse checkpoint architecture, observer mapping and file membership and verify
both input JSON SHA-256 hashes. Checkpoints from the other architecture are rejected.

ScanMatch, SED and STDE are included; install `multimatch-gaze` for MultiMatch.
Paths shorter than three fixations are excluded from MultiMatch only. Predicted
fixations over 10 seconds or paths over 8192 temporal symbols receive temporal
ScanMatch 0 and `duration_outlier=1`, avoiding excessive temporal expansion.
Real PerGAZE fixation durations are below three seconds. `--greedy` chooses
most likely cells and median durations; `--eval_repeat_num N` repeats sampling.

Optional semantic text, as in the original Gazeformer pipeline:

```powershell
# Install sentence-transformers first; model weights download on first use.
python src/IndividualScanpath/PerGAZE/GazeformerISP/src/preprocess/feature_extractor.py --device cuda:0
python src/IndividualScanpath/PerGAZE/GazeformerISP/src/train.py --text_embeddings dataset/PerGAZED/dataset/gazeformer_task_embeddings.npz
```

The NPZ archive stores Unicode task strings and float vectors without pickle.
`--lm_model` can point to a local SentenceTransformer directory. The vector
dimension overrides `text_dim`; test/resume restore the archive path and verify
its SHA-256. Provide the same contents with `--text_embeddings` if relocating it.

The original metric copyright/GPL notices and architecture attribution are
retained in source. Cite IndividualScanpath and the corresponding source
models/datasets. Functional tests do not constitute full model training.

Image geometry is centralized in `src/geometry.py`: image 1024x768, scanpath
512x352, feature grid 32x24 and action grid 32x22. Action cells cover 16x16
scanpath pixels. ScanMatch uses 16x11 bins; SED, STDE and MultiMatch use the
512x352 frame. All bbox/map guidance is resized to the action grid before
integration. Checkpoints from the older 640x480 / 320x240 configuration are
rejected and require retraining because the feature/action grids have changed.

`test.py` now evaluates the validation file (`test_seen.json`) by default and
writes `evaluation_validation.json` and `predictions_validation.json`. This file
is also used to select the best checkpoint during training. Use `--split train`
or `--split all` only for diagnostic evaluation. No third held-out test set is
created. Checkpoints from the former 80/10/10 split require a new training run.

## Hyperparameter presets

Choose the original source's defaults or its published run script:

```powershell
python src/train.py --hyperparam_preset air
python src/train.py --hyperparam_preset coco
python src/train.py --hyperparam_preset air_run
python src/train.py --hyperparam_preset coco_run
```

| Preset | Epochs | Start RL (zero-based) | Train batch | Max/min fixations | Seed | Val after epoch index |
| --- | --- | --- | --- | --- | --- | --- |
| air | 40 | 25 | 1 | 16 / 1 | 0 | 5 |
| coco | 40 | 25 | 3 | 7 / 1 | 0 | 5 |
| air_run | 40 | 25 | 1 | 16 / 1 | 10 | 5 |
| coco_run | 40 | 20 | 3 | 7 / 1 | 10 | 5 |

Gazeformer head dropout is 0.4, encoder dropout 0.1 and decoder dropout 0.2.
The requested 1024x768 image / 512x352 scanpath geometry is preserved.

`epoch > no_eval_epoch` enables validation. `checkpoint.pth` is still saved
at every epoch even if validation is deferred; `best.pth` is saved only after
validation improves. `supervised.pth` is saved immediately before RL by default,
instead of copying the entire run directory like the original scripts.
Reports now include learning rate and validation score.

Old fixed-LR checkpoints require a fresh run. Resume restores saved training
hyperparameters and scheduler state; explicit conflicting hyperparameters or
a changed batch limit are rejected. `--epoch` can extend the target duration.
Evaluation can use an old checkpoint because it does not resume the scheduler.
For a two-GPU ChenLSTM run, `--batch 4` is a hardware override, not the original
preset batch. Validation batch 1 uses only one GPU per batch; override
`--test_batch` if desired.
