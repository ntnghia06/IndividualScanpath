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

Default training: batch 2, 30 epochs, supervised for epochs 0-19 and ScanMatch
policy gradients from epoch 20. `--device auto` chooses CUDA if available;
`--device cpu` is supported. `--no-pretrained` runs with random backbone weights
without downloading weights. `--workers 0` is Windows-safe. `--max_batches N`
and `--eval_max_batches N` bound functional runs; zero processes the full split.

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
