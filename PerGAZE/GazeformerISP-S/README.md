# PerGAZE GazeformerISP-S: k-shot subject embedding finetuning

Load a trained PerGAZE GazeformerISP checkpoint, sample exactly **k scanpaths
per support observer**, finetune only their subject embedding rows for **3
epochs**, and evaluate `test_unseen.json` after each epoch. Save the adapted
checkpoint with the highest harmonic mean of the two ScanMatch metrics.

The branch reuses the sibling `GazeformerISP/src` loader/model/metrics and needs
the full repository. Model dimensions, max_length, vocabulary, geometry and
text backend are restored from the base checkpoint. No pretrained weight
download is necessary: all model weights are loaded from that checkpoint.

Current support data has six unseen observers: `air:JY`, `air:SC`, `air:YN`,
`coco:7`, `coco:8`, `coco:9`. The base checkpoint has 23 seen observers.
New embedding rows default to the mean of the original observers from the same
dataset. `--init random` enables seeded random initialization instead.

All existing model parameters, old embedding rows, BatchNorm buffers and
dropout behavior are frozen. The model stays in eval mode while autograd
computes support loss through the frozen network to the new embedding rows.
Loss is the original supervised action loss plus duration loss. Adam updates
only the subject-embedding parameter with masked gradients, zero weight decay
and a fresh optimizer state. There is no RL adaptation stage.

## Run locally from D:\PerScan

```powershell
conda activate cs117
python src/IndividualScanpath/PerGAZE/GazeformerISP-S/src/train.py --k 5
```

Default base checkpoint:
`src/IndividualScanpath/pergaze_gazeformer_run/checkpoints/best.pth`.
Pass `--checkpoint` to use another base checkpoint (best.pth or checkpoint.pth).
Default data files are `dataset/PerGAZED/dataset/support.json` and
`test_unseen.json`; image/map directories match the base training pipeline.

`--k` is a hyperparameter, default 5. Default seed is 10, epochs 3, embedding
learning rate 1e-3, train batch 2 and evaluation batch 4. Record IDs are sorted
and each observer has a deterministic seeded sampler. Exactly k samples must
be available for every support observer; duplicates and overlap between selected
support and test are rejected. Sampling is pooled over that observer's available
conditions/tasks, without imposing category quotas. The test set never provides
gradient updates. Its observers must have selected support examples.

Supports CPU, one GPU, or DataParallel via `--gpu_ids 0 1`. Default CUDA mode
uses all visible GPUs. Evaluation sampling uses the same eval seed after every
epoch, isolated from the training RNG. `--greedy` makes decoding deterministic.

## Outputs

Default: `GazeformerISP-S/runs/k5_seed10/`.

- `checkpoints/checkpoint.pth`: last adapted model, optimizer and RNG for resume.
- `checkpoints/best.pth`: highest-scoring adapted epoch (among epochs 1-3).
- **`report.json`**: one report containing config, base config, input hashes,
  selected support IDs per user, initialization baseline, all epoch losses and
  metrics, best epoch, best metrics and best predicted scanpaths. Metrics include
  overall, per-condition and per-subject statistics from the base evaluator.

The requested best-epoch selection uses **test_unseen** itself. Reported best
performance therefore includes test-based model selection; it is not an
untouched held-out test estimate. The initialization baseline is reported for
comparison, but best.pth is selected only among adapted epochs.

## Kaggle

Import `kaggle_gazeformer_s.ipynb`, enable Internet and GPU T4 x2, attach:

1. The PerGAZE dataset with support.json/test_unseen.json/images/attention_reasoning.
2. Output of the base Gazeformer run with checkpoints/best.pth.

Set K and SEED in the notebook. Choose the correct checkpoint if multiple runs
are attached. Optional pretrained text mode requires the **same NPZ contents**
used by the base run; pass `--text_embeddings` with its relocated path. It must
also cover all support/test tasks. Learned-text base checkpoints need no NPZ.

## Resume and functional checks

```powershell
python src/IndividualScanpath/PerGAZE/GazeformerISP-S/src/train.py --k 5 --resume
python -m unittest discover -s src/IndividualScanpath/PerGAZE/GazeformerISP-S/tests
```

Resume requires the same base checkpoint bytes, selected support records, k,
seed, input files and training settings. Restore the entire adapted run directory,
including report.json and both checkpoint files. GPU count may change. Like the
base trainer, interrupted epochs resume from the last completed epoch.

`--max_batches` / `--eval_max_batches` are optional functional-test limits;
leave both zero to train all k samples per observer and evaluate all unseen
samples. Full model training settings and geometry are inherited from the base
checkpoint; changing them is not part of subject-only adaptation.

To re-evaluate the saved adapted best checkpoint, run `src/test.py --k 5`
with the same image/map/test paths. It appends a reevaluation section to the
same report.json. Use --checkpoint to select another adapted checkpoint.
