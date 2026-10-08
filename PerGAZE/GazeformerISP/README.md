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
caches are reused; --overwrite replaces them. Default .pth storage and model inputs are FP32.
A feature file takes about 6 MiB. Cache filenames hash relative image
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


## Original RL and FP32 revision

RL now sums trial losses with batch-wide mask denominators. Sampling uses
exp(mu + noise * sigma2) as in the original implementation. STOP-masked
probabilities select actions; unmasked probabilities supply log-probabilities.
Duration parameter clamps remain removed; RL clips duration at the original
condition-specific positions described below. Invalid model parameters stop explicitly. Checkpoints from the earlier RL protocol cannot resume training.
Gazeformer uses FP32 feature caches and ToTensor -> Resize -> Normalize.
Its old feature caches must be re-extracted using --mode images --overwrite;
casting old FP16 values to FP32 is not supported as a migration.


## Original metric and duration protocol

ScanMatch uses 16x12 bins. MultiMatch pads each scanpath to at least 3 fixations
using (1,1,0.001) as in the original evaluator. Reports count padded target,
prediction and pair comparisons, and retain null values with valid_count=0 when
MultiMatch is undefined. The former temporal-score outlier cutoff is removed.

Evaluation reports retrieval under retrieval: temporal ScanMatch score matrices,
R@1/3/5/10 in percent, MRR and rank statistics, overall and by condition. Groups
use condition + image + task + question ID + repeat. Candidate observers remain
condition-specific. Reports include candidate counts and single-candidate queries.
Duplicate ground truths for one observer use their maximum similarity. All-invalid
rows are excluded, invalid cells rank as -1, and ties follow original reverse
numpy argsort. Metrics describe only evaluated records if a batch limit is used.

RL resamples nonfinite reward trials; a retry limit produces an explicit error
instead of an endless loop. ChenLSTM TP/TA clips sampled duration to 3 seconds
before creating reward paths and duration loss. ChenLSTM VQA and Gazeformer
use raw sampled duration for reward, then clip only duration-loss input to
0..100 seconds. Duration log-density and supervised/RL masks use the original
formula with epsilon=1e-7. Eval/test sampling is not clipped, matching the original
evaluation path. Structural invalid model parameters still cause explicit errors.

Earlier checkpoint weights can be evaluated with this metric protocol when
architecture-compatible. Start a new run instead of resuming old best-score,
optimizer or RL state across the protocol change. Cached image features and
sentence vectors do not need regeneration for this revision.
