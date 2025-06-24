# Contrastive Learning for Anomaly Detection

Supporting repository for the paper *[Anomaly preserving contrastive neural embeddings for end-to-end model-independent searches at the LHC](https://arxiv.org/abs/2502.15926)* and my Master's thesis *Representation learning for New Physics Discovery in the CMS experiment*.
## Construct low-dimensional expressive contrastive embeddings for improved anomaly detection in high-energy particle physics

Based on a [synthetic particle physics dataset](https://www.nature.com/articles/s41597-022-01187-8) simulating the data stream at the Level 1 Trigger at CMS (CERN), run the following contrastive learning methods with a 5-folded dataset containing a mixture of four Standard model background processes (`create_kfold_traintestfile.py`).

For training a supervised or self-supervised (with standard naive-masking augmentation) embedding using fully-connected models run:

`python3 train_with_signal.py PATH_TO_DATASET --epochs 50 --batch-size 1024 --type Delphes --loss-temp 0.9 --base-lr 0.2 --supervision selfsupervised (supervised) --latent-dim 4 --k-fold --train`

For training with a supervised Transformer architecture run:

`python3 train_with_signal_lxplus.py PATH_TO_DATASET --epochs 50 --batch-size 1024 --type Delphes --loss-temp 0.1 --base-lr 0.00025 --supervision supervised --k-fold --train`

More detail on the constructed contrastive embeddings can be found in the paper. However, feel free to try different training parameters and modify model and augmentation choices in the code directly.
## Linear evaluation and Anomaly Detection on the contrastive embeddings

For evaluating the background seperation of the low-dimensional embedding, train a linear layer on-top of the frozen embedding:

`python3 linear_evaluation.py PATH_TO_DATASET --pretrained PATH_TO_MODELWEIGHTS_vaeX.pth --arch SimpleDense_ADC --latent-dim 4 --type freeze --k-fold X`

For performing model-agnostic anomaly detection with NPLM run:

`python3 run-NPLM-toy.py --n_toys 250 --size_ref 1000000 --size_back 100000 --model_name PATH_TO_MODEL --size_sig SIZE_SIGNAL --signal SIGNAL(4-7) --inference`



### Content
- `cl` construct low-dimensional contrastive embeddings from high-dimensional collider data
- `dino` construct embeddings with DINO
- `nplm` perform anomaly detection based on log-likelihood ratio testing
- `orca` open-world semi-supervised anomaly detection with ORCA
