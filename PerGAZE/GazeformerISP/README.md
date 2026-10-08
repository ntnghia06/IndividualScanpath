# PerGAZE GazeformerISP (cached features, geometry v3)

Train on train.json, validate on test_seen.json. Defaults: 5 supervised epochs,
5 RL epochs, one warmup epoch. TP/TA/VQA subject identities are independent.

## Inputs and architecture

- Frozen MaskRCNN COCO ResNet50: resize images to 1024x768; cache features
  [2048,24,32] in .pth files. Each file retains original image dimensions.
- SentenceTransformer sentence-transformers/stsb-roberta-base-v2 produces
  768-dimensional vectors saved in gazeformer_task_embeddings.npz.
- TP text: object name. VQA text: question in task. TA text:
  "Search for the <object> in the image." All conditions use text.
- Gazeformer learns visual attention from image features and sentence embeddings.
  It does not consume bbox/attention_reasoning maps and has no 50/50 prior blend.
- Prediction grid 24x32: 769 actions including STOP, without logit interpolation.
  Scanpath coordinates remain width=512, height=352. Cell indices are obtained
  by proportional scaling; decoding returns cell centers in this frame.
- No backbone or learned word embedding is included in the training model.
  --train_backbone and non-COCO backbone selection are rejected.

ChenLSTMISP keeps its original trainable ImageNet dilated ResNet50 and guidance
maps, as requested. Its image/scanpath geometry remains 320x240 and 30x40.

## Precompute, then train (from D:/PerScan)

```powershell
conda activate cs117
pip install -r src/IndividualScanpath/PerGAZE/GazeformerISP/requirements.txt
python src/IndividualScanpath/PerGAZE/GazeformerISP/src/preprocess/feature_extractor.py --batch 4 --device cuda:0
python src/IndividualScanpath/PerGAZE/GazeformerISP/src/train.py --gpu_ids 0 1 --workers 2
```

The extractor defaults to the local dataset folder and includes train, test_seen,
support and test_unseen when present. --files explicitly selects JSON files.
Frozen encoders use only image pixels and task text, never scanpath labels.
--mode images/text/all selects the preprocessing stage. Existing valid image
caches are reused; --overwrite replaces them. Default .pth storage is float16,
with float32 model inputs; --storage_dtype float32 preserves full precision.
A feature file takes about 3 MiB at float16. Cache filenames hash relative image
paths, preventing collisions between conditions/tasks with identical filenames.

Use --feature_dir and --text_embeddings to relocate caches in train/test.
Missing or incompatible caches fail with an error; there is no online fallback.
The sentence archive must cover every task text being evaluated. Resume checks
its SHA-256 and JSON split hashes. Keep the same archive contents when relocating.

## Kaggle

Open kaggle_gazeformer_2gpu.ipynb and enable GPU/internet. The notebook installs
sentence-transformers, runs batched COCO ResNet extraction on available GPUs,
encodes all four JSON splits, and then trains from /kaggle/working caches.
Train remains DataParallel on two visible GPUs. Cache extraction batch and train
batch are separate settings. Input datasets are read-only; outputs go to working.

Retain the full run, gazeformer_task_embeddings.npz and optionally the image
cache as Kaggle output. Re-extract images if the cache is not imported next
session; point FEATURE_DIR/TEXT_FILE to imported artifacts to reuse them.

Outputs: hparams.json, manifest.json, epoch_*.json, checkpoints/checkpoint.pth,
best.pth and supervised.pth. Best score is the harmonic mean of the two ScanMatch
validation means. Test uses the same cached inputs and writes evaluation JSON.

## Checkpoint migration

This revision changes text backend, action grid and subject identities.
Earlier online-backbone/705-action or pooled-TA/TP checkpoints cannot resume or
be passed to GazeformerISP-S. Train a new v3 base model. GazeformerISP-S uses
its exact original sentence archive and cached support/unseen image features.
