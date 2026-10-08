# PerGAZE GazeformerISP: online image features

Train on train.json, validate on test_seen.json. Defaults remain 5 supervised
and 5 RL epochs with one warmup epoch. TP/TA/VQA observers are independent.

## Image and text inputs

Images are decoded on demand during train, validation and test. Preprocessing is
ToTensor -> Resize((768,1024)) -> Normalize. The frozen COCO MaskRCNN ResNet50
backbone runs inside each model batch, producing 2048x24x32 features. Backbone
parameters and BatchNorm buffers are frozen; train() keeps the backbone in eval
mode. No .pth feature cache is read or written by training/evaluation.

SentenceTransformer stsb-roberta-base-v2 text vectors are still prepared once,
as gazeformer_task_embeddings.npz. TP encodes object names, VQA encodes questions,
and TA encodes "Search for the <object> in the image." All vectors are 768D.
Gazeformer learns attention from image features and text without external bbox/
attention maps. Its action grid is 24x32 (769 actions including STOP), with
scanpath coordinates width=512, height=352. ChenLSTM remains independent.

## Prepare text, then train (from D:/PerScan)

```powershell
pip install -r src/IndividualScanpath/PerGAZE/GazeformerISP/requirements.txt
python src/IndividualScanpath/PerGAZE/GazeformerISP/src/preprocess/feature_extractor.py --mode text --device cuda:0
python src/IndividualScanpath/PerGAZE/GazeformerISP/src/train.py --gpu_ids 0 1 --workers 2
```

The text extractor includes train/test_seen/support/test_unseen when present.
--files explicitly chooses JSON files. The frozen text encoder never uses gaze
labels. Train/test require --img_dir and --text_embeddings, not --feature_dir.
Legacy image-extraction utilities remain available only for explicit use and
are not part of the default pipeline.

## Kaggle

Use kaggle_gazeformer_2gpu.ipynb. Enable GPU and internet for downloading weights
once. Setup prepares text embeddings only, then train and test directly read
DATA/images. Online ResNet is replicated by DataParallel on both visible GPUs.
Keep the whole run directory and text NPZ for resume or GazeformerISP-S; image
cache files are unnecessary. Delete obsolete caches manually if they occupy disk.

## Checkpoints

New checkpoints include frozen backbone weights and original text archive hash.
Resume restores model, optimizer, scheduler and RNG. The optimizer contains
trainable transformer/head/subject parameters only, matching older head-only
cached-feature checkpoints. Compatible cached-v3 checkpoints initialize the
same pretrained COCO backbone and load their heads strictly. Older geometry,
subject, text or loss/metric protocols retain the existing compatibility checks.
GazeformerISP-S also supports compatible head-only checkpoints by initializing
the frozen pretrained backbone, then adapting only new subject embedding rows.

## Metrics and outputs

Supervised action loss uses log(softmax+1e-7); duration loss and RL duration
handling follow the original implementation. ScanMatch uses 16x12 bins;
MultiMatch pads short paths to 3 fixations and drops the full 5D MM vector when
one component is invalid. Retrieval temporal ScanMatch is independent of MM.
Reports contain overall/per-condition/per-subject metrics, padding counts,
retrieval matrices and R@1/3/5/10/MRR. SED retains the corrected region handling.

Outputs include hparams.json, manifest.json, epoch_*.json and checkpoints/
checkpoint.pth, best.pth and supervised.pth. Best is selected by the harmonic
mean of the two ScanMatch validation means. Test writes report and prediction
JSON files. TensorBoard and grouped observer batching remain outside this change.
