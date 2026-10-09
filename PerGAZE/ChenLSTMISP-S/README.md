# PerGAZE ChenLSTMISP-S: subject embedding adaptation

Load the best checkpoint from PerGAZE/ChenLSTMISP Colab training, select seeded
K support records per unseen (condition, observer), finetune only new subject
embedding rows for 3 supervised epochs, and evaluate all test_unseen after each.

## Colab

Open kaggle_chenlstm_s_colab.ipynb and enable one GPU. Default paths match the
training notebook: MyDrive/PerGAZE/dataset and
MyDrive/PerGAZE/runs/pergaze_chenlstm_colab_1gpu/checkpoints/best.pth.
The dataset also needs support.json and test_unseen.json. Images and attention
maps are read directly from Drive, with no dataset copying or feature cache.
No SentenceTransformer/text NPZ is needed. ResNet and all other weights are
restored from the base checkpoint; no pretrained model download is necessary.

## Protocol

Defaults: K=5, seed=10, 3 epochs, LR=1e-3, train/eval batch=1. Adaptation seed is
independent of base training seed. TA 7/8/9, TP 7/8/9 and VQA JY/SC/YN are nine
separate identities: K=5 selects 45 support records. Sampling is stable under
record reordering, checks duplicates and selected support/test overlap, and
requires support for every test identity. IDs of selected records are reported.

New embeddings use the mean seen embedding in their own condition population.
--init random uses seeded normal initialization with std=0.1. Existing subject
indices are preserved. ResNet, ConvLSTM, all 18 COCO object branches, the shared
VQA branch, old embedding rows and all BatchNorm buffers stay frozen. The whole
model stays in eval mode with autograd enabled only for new subject embedding
rows. Adam has zero weight decay and fresh optimizer state; masked gradients
and a per-epoch invariant check preserve seen rows. There is no RL stage.
Supervised action and duration losses match the original ChenLSTM formulas.

Every epoch evaluates the full unseen set with a fixed eval seed. Best adapted
epoch is selected by harmonic mean of the two ScanMatch means. test_unseen is
used for model selection as requested, not as an untouched test estimate.
Report metrics reuse ChenLSTM ScanMatch 16x12 bins, padded MultiMatch, SED/STDE
and condition-specific subject retrieval. Model geometry/max_length comes from
the base checkpoint: images/scanpaths 320x240, grid 30x40 plus STOP.

## Outputs and resume

RUN/checkpoints/best.pth, checkpoint.pth and one report.json retain config, file
hashes, selection, initialization baseline, each epoch and best metrics/predictions.
Colab saves them directly to Drive after every epoch and copies subprocess logs
to RUN/logs. Set RESUME=True with identical inputs and adaptation settings after
session interruption. Keep the original base checkpoint and full adapted run.
src/test.py appends reevaluation to the same report.json.

Compatible base checkpoints must have separate TA/TP identities, 18 object
branches and the VQA branch. Old shared-head or pooled-observer checkpoints fail
explicitly. The branch reuses sibling ChenLSTMISP source, so clone the full repo.

From the ChenLSTMISP-S directory:

```powershell
python src/train.py --checkpoint PATH_TO_BASE_BEST --support_file PATH_TO_SUPPORT --test_file PATH_TO_UNSEEN --img_dir PATH_TO_IMAGES --att_dir PATH_TO_MAPS --k 5 --seed 10
python -m unittest discover -s tests -v
```
