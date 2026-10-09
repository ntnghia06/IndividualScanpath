# PerGAZE GazeformerISP-S: k-shot subject embedding finetuning

Requires a compatible geometry-v3 GazeformerISP base checkpoint: sentence
embeddings, 769 actions and separate TA/TP identities. Earlier 705-action
or pooled-observer checkpoints are incompatible.

Select exactly k support scanpaths per (condition, observer) using seed 10 by
default, then finetune only new subject embedding rows for 3 supervised epochs.
The nine unseen identities are ta:7/8/9, tp:7/8/9, air:JY/SC/YN. k=5 selects
45 samples. TA user 7 and TP user 7 are different people. New rows initialize
from the mean of the corresponding seen population, or seeded random with
--init random. All other weights and old subject rows remain frozen.

## Inputs

Use a compatible GazeformerISP checkpoint with geometry-v3 (769 actions),
separate TP/TA subjects and the original sentence archive. Images are read from
--img_dir and processed online by the frozen COCO ResNet50 in each batch.
No --feature_dir or pre-extracted image files are needed. The notebook prepares
no image cache; TEXT_EMBEDDINGS points to the exact NPZ used by the base run.
The NPZ must include support/unseen text and its SHA-256 must match.

## Run from D:/PerScan

```powershell
python src/IndividualScanpath/PerGAZE/GazeformerISP-S/src/train.py --checkpoint PATH_TO_BEST --k 5 --text_embeddings PATH_TO_ORIGINAL_NPZ
```

Defaults: k=5, seed=10, epoch=3, lr=1e-3, batch=2, test_batch=4. DataParallel
supports --gpu_ids 0 1. Compatible head-only cached-feature checkpoints initialize
the fixed pretrained COCO backbone before loading heads; new online checkpoints
load their saved backbone without a download. Only new subject rows are adapted.

## Outputs and resume

Default runs/k5_seed10_by_condition/ contains checkpoints/checkpoint.pth,
checkpoints/best.pth and report.json. The report records input/config provenance,
selected support IDs, baseline, epoch losses/metrics and the best adapted epoch.
Best score is the harmonic mean of the two ScanMatch means on test_unseen.
This requested selection uses test_unseen itself, not an untouched test estimate.

Import the whole run directory into Kaggle, restore it to RUN, retain original
NPZ and original image input, then add --resume with identical k,
seed and input files. src/test.py re-evaluates adapted best.pth and appends to
report.json. Old pooled-TA/TP adaptation runs must not be resumed.


Few-shot eval now uses original-MM/retrieval/duration metrics, including padding
counts and subject retrieval matrices in report.json. Start a new adaptation run
rather than resuming best-epoch selection from the previous evaluation protocol.
Compatible adapted weights may be re-evaluated; the resulting summary includes
its metric_protocol identifier.


## Google Colab (one GPU)

Open kaggle_gazeformer_s_colab.ipynb, select a GPU runtime and mount Drive.
Edit DATA_SOURCE, CHECKPOINT_SOURCE and TEXT_SOURCE. The notebook stages original
images and model/text inputs to /content by default, always uses --gpu_ids 0,
and saves checkpoints/report directly to the configured RUN on Drive after every
epoch. Set RESUME=True after a session interruption with identical settings.
Set COPY_DATA_TO_LOCAL=False to read images directly from Drive. No image features
are precomputed or cached. Only the original sentence embedding NPZ is required.
