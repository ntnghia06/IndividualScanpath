# PerGAZE GazeformerISP

Independent PerGAZE implementation with the same folder layout as the AiR and
COCO_Search18 branches. Run commands from `D:\PerScan`.

```text
GazeformerISP/
  bash/               train.sh, test.sh
  src/
    dataset/          JSON loader, subject mapping and image-disjoint splits
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

Reads `dataset/PerGAZED/dataset/PerGAZE.json` directly. `present` uses
`images/TP/task/name` and JSON bbox `[x,y,width,height]`; `vqa` uses
`images/VQA/name` and `attention_reasoning/qid.npy`; `absent` uses
`images/TA/task/name` with zero auxiliary guidance. Flat TP/TA image directories
are also supported. Fixation coordinates use actual image dimensions, are
clipped at boundaries and scaled to 320x240; `T` is milliseconds.
The explanation field `prediction` is not a training target.

COCO TP/TA share observer IDs, and AiR observers use a separate namespace.
Splits are deterministic 80/10/10 by image, stratified by condition membership.
All questions/tasks/observers on one image stay in the same split. This is a
new PerGAZE split rather than the original dataset benchmark split.

The network adapts the original GazeformerISP Transformer, observer-centric
integration and adaptive fixation heads. Images are resized to 480x640; online
ResNet50 features form a 15x20 token grid. Fixation logits are interpolated to
30x40 plus the stop action. Nonzero bbox/attention guidance is normalized and
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
python src/IndividualScanpath/PerGAZE/GazeformerISP/src/train.py --resume --epoch 30
python src/IndividualScanpath/PerGAZE/GazeformerISP/src/test.py --split all

python -m unittest discover -s src/IndividualScanpath/PerGAZE/GazeformerISP/tests
```

Default training: batch 2, 30 epochs, supervised for epochs 0-19 and ScanMatch
policy gradients from epoch 20. `--device auto` chooses CUDA if available;
`--device cpu` is supported. `--no-pretrained` runs with random backbone weights
without downloading weights. `--workers 0` is Windows-safe. `--max_batches N`
and `--eval_max_batches N` bound functional runs; zero processes the full split.

Default output is this model's own `runs/` directory. It contains manifests,
hyperparameters, validation/epoch reports, `checkpoints/checkpoint.pth`,
`checkpoints/best.pth`, evaluation metrics and predicted scanpaths. Test/resume
reuse checkpoint architecture, observer mapping and splits and verify input
JSON SHA-256. Checkpoints from the other architecture are rejected.

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
