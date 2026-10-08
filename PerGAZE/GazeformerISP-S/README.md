# PerGAZE GazeformerISP-S: k-shot subject embedding finetuning

Requires a newly trained cached-v3 GazeformerISP base checkpoint: sentence
embeddings, 769 actions and separate TA/TP identities. Earlier online-backbone,
705-action or pooled-observer checkpoints are incompatible.

Select exactly k support scanpaths per (condition, observer) using seed 10 by
default, then finetune only new subject embedding rows for 3 supervised epochs.
The nine unseen identities are ta:7/8/9, tp:7/8/9, air:JY/SC/YN. k=5 selects
45 samples. TA user 7 and TP user 7 are different people. New rows initialize
from the mean of the corresponding seen population, or seeded random with
--init random. All other weights and old subject rows remain frozen.

## Inputs

- --checkpoint: best.pth or checkpoint.pth from the new v3 base training.
- --support_file / --test_file: support.json / test_unseen.json.
- --feature_dir: frozen COCO image feature .pth files, same cache format as base.
- --text_embeddings: exact gazeformer_task_embeddings.npz used by base training,
  including support/unseen questions and TA instructions. SHA-256 must match.
- --img_dir: original image tree for resolving portable feature cache keys.

The base notebook precomputes both encoders for all four splits. The few-shot
notebook can reuse an imported image cache or extract support/unseen features
before finetuning. It discovers the imported original NPZ, or accepts an explicit
TEXT_EMBEDDINGS path. Do not regenerate a different archive for adaptation.

## Run from D:/PerScan

```powershell
python src/IndividualScanpath/PerGAZE/GazeformerISP-S/src/train.py --checkpoint PATH_TO_V3_BEST --k 5 --feature_dir PATH_TO_CACHE --text_embeddings PATH_TO_ORIGINAL_NPZ
```

Default k=5, seed=10, epoch=3, lr=1e-3, batch=2, test_batch=4. Supports
DataParallel with --gpu_ids 0 1. Sampling is reproducible under record reordering.
Support and test overlap is rejected. Evaluation uses the same RNG seed per epoch.
The model stays in eval mode during finetuning; only masked new embedding rows
receive gradients. Adam uses no weight decay and a fresh optimizer state.

## Outputs and resume

Default runs/k5_seed10_by_condition/ contains checkpoints/checkpoint.pth,
checkpoints/best.pth and report.json. The report records input/config provenance,
selected support IDs, baseline, epoch losses/metrics and the best adapted epoch.
Best score is the harmonic mean of the two ScanMatch means on test_unseen.
This requested selection uses test_unseen itself, not an untouched test estimate.

Import the whole run directory into Kaggle, restore it to RUN, retain original
NPZ and image cache (or re-extract images), then add --resume with identical k,
seed and input files. src/test.py re-evaluates adapted best.pth and appends to
report.json. Old pooled-TA/TP adaptation runs must not be resumed.


The original-sigma2 sampling revision requires FP32 caches produced using
ToTensor -> Resize -> Normalize. Re-extract earlier caches; old adaptation runs
must restart because their evaluation sampling protocol differs.
