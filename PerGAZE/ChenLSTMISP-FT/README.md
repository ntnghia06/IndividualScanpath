# PerGAZE ChenLSTMISP-FT

Full-model alternative to ChenLSTMISP-S. Same base checkpoint, deterministic K support
samples per condition-specific unseen observer and test_unseen evaluation.
Defaults K=5, seed=10, 3 supervised epochs, LR=1e-5, weight decay 0, batch 1.
9 identities select 45 support samples at K=5. New subject rows initialize from
seen population means (or seeded random). All parameters are unfrozen; there
is no row mask or old-row preservation constraint. Normal train mode enables
dropout and updates BatchNorm. Parameters not used by a support batch may have
no gradient. No RL or pretrained SentenceTransformer finetuning is performed.

Use kaggle_chenlstm_ft_colab.ipynb on one GPU. It reads dataset/checkpoint from
Drive, uses a separate FT output directory and saves best/last checkpoint plus
report.json. Best is chosen on test_unseen by harmonic mean of both ScanMatch
means, so it is model-selected performance on that set. Resume restores full
optimizer/model/RNG with identical inputs and settings. ISP-S runs are rejected
for resume/eval by adaptation_mode. Image geometry/max_length come from the
base model. Gazeformer uses its original frozen sentence vector archive; image
backbone gradients are enabled in FT. No original ISP/ISP-S code is changed.

The full repository is required because this branch imports the sibling model
source. src/train.py and src/test.py mirror ISP-S inputs; --lr and --weight_decay
control full-model optimization. Tests verify non-embedding/backbone updates.
