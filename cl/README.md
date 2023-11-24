# Contrastive Learning + Normalizing Flows = Anomaly Detection ᓬ(•ᴗ•)ᕒ 


## Introduction

**This project can be broken into two primary components**
- _Contrastive Learning:_ Trains an auto-encoder to compress raw CMS event data into a meaningful, compact latent space. Uses CL to cluster events of identical labels and distance dissimilar events. 
- _Normalizing Flows:_ Uses normalizing flows to detect anomalous data from standard model physics events within the compressed representation. 

## Creating Testing Data

`python make_datasets.py --new_filename 'zscore.npz'`

## Training Encoder with Contrastive Learning

`python test.py --train True`

## Plotting results

`python test.py --plot True`

`python test.py --anomaly True`

## Files
- `configs.npy` Contains relevant constants associated with normalization and project hyper-parameters in a dict. 
- `data_preprocessing.py` All functions associated with preprocessing Delphes and raw CMS data before use. Experimentally concluded _zscore_ and _max pT_ normalization are optimal for Delphes. 
- `dnn_classifier.py` First iteration of classifiers that implements a simple DNN. See normalizing flow for current use. 
- `graphing_module.py` Repository of all graphing modules: PCA, Corner, ROC, tSNE, ect
- `hyperparam_search.py` Uses 💗KerasTuner💗 for hyperparameter search. 
- `losses.py` Defines custom loss functions: reconstruction, KL, and SimCLR
- `make_datasets.py` Creates biased-training datasets st event representations are more distributed across tt and QCD compared to Delphes. Also provides interface for converting raw cms data into interpretable representation. 
- `models.py` Defines models as the name suggests 
- `nf_classifier.py` Creates initial normalizing flow class and trains model using Delphes
- `predict.py` Given pre-trained AE, visualizes where raw CMS data lies in the latent space. 
- `test.py` Overlord file. Used for all initial training and plotting matters. 

