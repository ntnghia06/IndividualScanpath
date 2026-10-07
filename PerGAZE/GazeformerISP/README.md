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
python src/IndividualScanpath/PerGAZE/GazeformerISP/src/train.py --resume --epoch 40
python src/IndividualScanpath/PerGAZE/GazeformerISP/src/test.py --split validation

python -m unittest discover -s src/IndividualScanpath/PerGAZE/GazeformerISP/tests
```

Training defaults directly match the original AiR `opts.py` plus its
`bash/train.sh` overrides. No configuration selector is needed:
40 epochs, start RL at index 25, train batch 1, max/min
fixations 16/1, seed 10, clipping disabled, head dropout 0.4 and validation
starting at epoch index 6.
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

## Training and resume settings

Run `python src/train.py` with your data paths; all defaults already match the
original AiR run script. `epoch > no_eval_epoch` enables validation.
`checkpoint.pth` is saved at every epoch even if validation is deferred;
`best.pth` is saved when validation improves. `supervised.pth` is saved
immediately before RL. Reports include learning rate and validation score.

Resume restores saved training settings and scheduler state; explicit
conflicting hyperparameters or a changed batch limit are rejected. `--epoch`
can extend the target duration. Older fixed-LR checkpoints need a new run.
The selected PerGAZE dataset files and image/scanpath geometry are unchanged.

Train now displays a tqdm batch progress bar with phase, epoch, mean loss,
learning rate, elapsed time and ETA. Validation/evaluation have their own bars
with batch counts and processed prediction counts. Bars respect batch limits
and are emitted as text to stdout for Kaggle subprocess cells.

## Ready-to-run Kaggle notebook

Import `kaggle_gazeformer_2gpu.ipynb` into Kaggle, enable Internet, select GPU
T4 x2 and attach the PerGAZE dataset. The notebook uses the public GitHub repo,
validates data, runs a bounded smoke check, trains, evaluates and packages results.
All AiR run defaults are retained except notebook global train batch 2 on two
GPUs (one sample per GPU); a one-GPU session uses batch 1. `--gpu_ids 0 1` enables
DataParallel; omitting the IDs uses all visible GPUs automatically. Validation
batch 1 uses one GPU per batch; `--test_batch 2` is an optional override.
Portable checkpoints can be loaded on one or two GPUs. Resume keeps the saved
batch size and training settings. Optional semantic text embeddings are written
to `/kaggle/working/gazeformer_task_embeddings.npz` and must be retained when used.
The default remains learned word embeddings and requires no extra text model.
