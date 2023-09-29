# STPT

we are inspired by the success of the pre-trained language models to train a spatial-temporal pre-trained model using self-supervised learning. 
Such a model extracts knowledge from diverse HSTD (\eg, GPS traces, transportation data, mobile phone data), and can be fine-tuned and applied to various downstream applications (\eg, human verification and identification, traffic forecast, and human mobility analysis) given a small number of training data (for fine-tuning) as is shown in Fig.1.
% 
We thus introduce the Spatial-Temporal Pre-Training model, STPT, to generate robust and generic representations from HSTD using our novel self-supervised learning approach.

<p align="center">
<img src="/resources/framework.png" alt=STPT framework" height="350">
</p>


## Prerequisites
- [Python 3.9.12](https://www.continuum.io/downloads)
- [Keras 2.10.0](https://keras.io/)
- [Tensorflow 2.10.1](https://www.tensorflow.org/)
- GPU for fast training


## Usage
- Execute ```pretrain/main.py``` to train STPT.
- Execute ```finetune/classification/train.py``` to fine-tune the classification task.
- Execute ```finetune/seek_serve/train_ss.py``` to fine-tune the seeking and serving identification task.

``` -->

