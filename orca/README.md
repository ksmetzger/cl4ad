# Open-world semi-supervised anomaly detection with ORCA using DELPHES dataset
## Modified files from ORCA in order to work with particle physics simulation dataset DELPHES
Repository supporting the semester project: "Open-World Semi-Supervised Learning for New Physics Discovery With CMS", Kyle Metzger, ETH Zürich.
The files are copied and modified from https://github.com/snap-stanford/orca.

License: 
	
 	MIT License	
  	Copyright (c) 2022 Kaidi Cao

		Permission is hereby granted, free of charge, to any person obtaining a copy
		of this software and associated documentation files (the "Software"), to deal
		in the Software without restriction, including without limitation the rights
		to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
		copies of the Software, and to permit persons to whom the Software is
		furnished to do so, subject to the following conditions:

		The above copyright notice and this permission notice shall be included in all
		copies or substantial portions of the Software.

		THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
		IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
		FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
		AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
		LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
		OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
		SOFTWARE.
		
## Run ORCA with DELPHES
The project uses a [synthetic particle physics dataset](https://doi.org/10.1038/s41597-022-01187-8) simulating the data stream at the Level 1 Trigger at CMS (CERN). An embedding trained using *contrastive learning* with a variational autoencoder (https://github.com/katyagovorkova/cl-orca/tree/main/cl/cl) is compared to the direct input. Two different sized fully-connected models are considered.
### In order to run from direct input
+ `python orca_delphes.py --dataset background_with_signal_cvae --size large --labeled-num 4 --milestones 10 20 --epochs 30 --batch-size 1024`
+ `python orca_delphes.py --dataset background_with_signal_cvae --size simple --labeled-num 4 --milestones 10 20 --epochs 30 --batch-size 1024`
### In order to run from embedding input
+ `python orca_delphes.py --dataset background_with_signal_cvae_latent --size large --labeled-num 4 --milestones 10 20 --epochs 30 --batch-size 1024`
+ `python orca_delphes.py --dataset background_with_signal_cvae_latent --size simple --labeled-num 4 --milestones 10 20 --epochs 30 --batch-size 1024`
### Files
+ `orca_delphes.py` trains ORCA classifier
+ `open_world_delphes.py` contains dataset classes + preparation
+ `utils.py` supporting functions
+ `plotting.py` plotting file for 2D t-SNE plots and softmax distribution plots
+ `models.py` fully-connected models used for training
