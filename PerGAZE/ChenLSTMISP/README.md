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
python src/IndividualScanpath/PerGAZE/ChenLSTMISP/src/train.py --resume --epoch 10
python src/IndividualScanpath/PerGAZE/ChenLSTMISP/src/test.py --split validation

python -m unittest discover -s src/IndividualScanpath/PerGAZE/ChenLSTMISP/tests
```

Training uses AiR run hyperparameters with the requested 5 supervised + 5 RL
epoch schedule. No configuration selector is needed:
10 epochs, start RL at index 5, train batch 2, max/min
fixations 16/0, seed 1, clipping 12.5, dropout 0.2 and validation after every
epoch.
Both models use one-epoch warmup and linear supervised/RL learning-rate decay,
LR 1e-4, RL multiplier 0.1, weight decay 5e-5, validation batch 1 and five RL
samples with a mean reward baseline. Target blur is disabled. Data files and
image geometry remain PerGAZE-specific. Individual CLI flags can override
these defaults. Batch counts individual scanpaths, whereas original loaders
count groups of observers. `--no-pretrained` uses random backbone weights.
`--workers 0` is Windows-safe. Batch limits are available for functional checks;
zero processes the full split.

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
Use `--device cuda:0 --gpu_ids 0 1`; the default global batch 2 gives
approximately one sample per GPU. With no `--gpu_ids`, all visible CUDA GPUs
are selected automatically. Train, RL, validation and test use the same wrapper.
A small final batch may use fewer devices. Checkpoints save the unwrapped model
state and work with one or two GPUs. CUDA RNG restoration supports fewer GPUs.
Console startup reports selected GPUs and whether DataParallel is active.

## Training and resume settings

Run `python src/train.py` with your data paths; the default schedule is 5 supervised epochs
followed by 5 RL epochs, with validation after every epoch. `epoch > no_eval_epoch` enables validation.
`checkpoint.pth` is saved at every epoch even if validation is deferred;
`best.pth` is saved when validation improves. `supervised.pth` is saved
immediately before RL. Reports include learning rate and validation score.

Resume restores saved training settings and scheduler state; explicit
conflicting hyperparameters or a changed batch limit are rejected. `--epoch`
can extend the target duration. Older fixed-LR checkpoints need a new run.
The selected PerGAZE dataset files and image/scanpath geometry are unchanged.

The Kaggle notebook uses default global train batch 2 on two GPUs (one sample
per GPU); no configuration-selection argument is passed. Validation batch 1
uses one GPU per batch, matching the AiR run defaults. To increase the global
train batch, pass `--batch 4` explicitly.

Train now displays a tqdm batch progress bar with phase, epoch, mean loss,
learning rate, elapsed time and ETA. Validation/evaluation have their own bars
with batch counts and processed prediction counts. Bars respect batch limits
and are emitted as text to stdout for Kaggle subprocess cells.

Training schedule: displayed epochs 1-5 are supervised (SFT), epochs 6-10
are RL. One-epoch warmup is part of SFT. Both models validate after each epoch
and save supervised.pth after epoch 5. Existing checkpoints keep their saved
schedule on resume; start a new run directory for the new 5+5 schedule.


## Current backbone and guidance

ChenLSTM keeps the original trainable ImageNet dilated ResNet50 (no COCO cache).
TP builds guidance from PerGAZE xywh bbox; VQA reads qid.npy, resized to 30x40
and normalized; TA has zero guidance. TP/TA subjects now use separate tp:/ta:
identities. Checkpoints with pooled coco: observers require a new training run.
Gazeformer cached-v3 geometry changes do not alter ChenLSTM 320x240/30x40.


## Original RL and FP32 revision

RL now sums trial losses with batch-wide mask denominators. Sampling uses
exp(mu + noise * sigma2) as in the original implementation. STOP-masked
probabilities select actions; unmasked probabilities supply log-probabilities.
Duration parameter and sample clamps are removed; invalid values stop with an
explicit error. Checkpoints from the earlier RL protocol cannot resume training.
Gazeformer uses FP32 feature caches and ToTensor -> Resize -> Normalize.
Its old feature caches must be re-extracted using --mode images --overwrite;
casting old FP16 values to FP32 is not supported as a migration.
