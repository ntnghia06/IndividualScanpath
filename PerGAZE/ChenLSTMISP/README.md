# PerGAZE ChenLSTMISP

Independent PerGAZE implementation with the same folder layout as the AiR and
COCO_Search18 branches. Run commands from `D:\PerScan`.

```text
ChenLSTMISP/
  bash/               train.sh, test.sh
  src/
    dataset/          JSON loader, subject mapping and explicit file membership
    models/           network, losses and scanpath sampling
    preprocess/       full dataset validation
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

The ChenLSTMISP network adapts the original AiR implementation; zero guidance
matches the OSIE initialization. Images are resized to 240x320 and the action
map is 30x40. ImageNet ResNet50 weights are downloaded on first use.

```powershell
conda activate cs117
python src/IndividualScanpath/PerGAZE/ChenLSTMISP/src/preprocess/validate_dataset.py
python src/IndividualScanpath/PerGAZE/ChenLSTMISP/src/train.py
python src/IndividualScanpath/PerGAZE/ChenLSTMISP/src/test.py

# Short functional check on one real sample per condition, plus RL
python src/IndividualScanpath/PerGAZE/ChenLSTMISP/src/train.py --smoke_test --max_length 3 --rl_sample_number 2 --batch 1 --log_root src/IndividualScanpath/PerGAZE/ChenLSTMISP/runs/smoke

# Resume the default run or evaluate all records
python src/IndividualScanpath/PerGAZE/ChenLSTMISP/src/train.py --resume --epoch 30
python src/IndividualScanpath/PerGAZE/ChenLSTMISP/src/test.py --split validation

python -m unittest discover -s src/IndividualScanpath/PerGAZE/ChenLSTMISP/tests
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

The original metric copyright/GPL notices and architecture attribution are
retained in source. Cite IndividualScanpath and the corresponding source
models/datasets. Functional tests do not constitute full model training.

`test.py` now evaluates the validation file (`test_seen.json`) by default and
writes `evaluation_validation.json` and `predictions_validation.json`. This file
is also used to select the best checkpoint during training. Use `--split train`
or `--split all` only for diagnostic evaluation. No third held-out test set is
created. Checkpoints from the former 80/10/10 split require a new training run.

## Kaggle: two GPUs with DataParallel

Open `kaggle_chenlstm_2gpu.ipynb` in Kaggle, select GPU T4 x2, enable Internet
and attach the PerGAZE dataset. The public GitHub repository needs no token.
Use `--device cuda:0 --gpu_ids 0 1 --batch 4`; batch is the global batch,
approximately two samples per GPU. With no `--gpu_ids`, all visible CUDA GPUs
are selected automatically. Train, RL, validation and test use the same wrapper.
A small final batch may use fewer devices. Checkpoints save the unwrapped model
state and work with one or two GPUs. CUDA RNG restoration supports fewer GPUs.
Console startup reports selected GPUs and whether DataParallel is active.

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
| air | 30 | 20 | 2 | 16 / 0 | 1 | -1 |
| coco | 30 | 20 | 3 | 7 / 0 | 1 | 1 |
| air_run | 40 | 20 | 2 | 16 / 0 | 1 | -1 |
| coco_run | 20 | 10 | 3 | 7 / 0 | 10 | 1 |

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
