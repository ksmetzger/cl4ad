# Open-world semi-supervised anomaly detection with ORCA using DELPHES dataset
## Modified files from ORCA in order to work with particle physics simulation dataset DELPHES
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
Use weights `model_manual.pth` from https://github.com/katyagovorkova/cl4ad/blob/main/src/cl/models.py as pretraining by freezing the first two layers of the model `autoencoder.py`.
The delphes dataset is normalized by pT scaling [normalization](https://github.com/katyagovorkova/cl4ad/blob/d0a9095a8c4f86a8b55fa66638d11de153ee489d/src/data_preprocessing.py#L46).
### In order to run
+ `python orca_delphes.py --dataset background_with_signal_cvae --labeled-num 4`
## These files are a work in progress and not yet polished/well commented. I'll soon update them to make it a bit more readable/usable. :)