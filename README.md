# Noise2Score-Reproduce

## Introduction 

Originally submitted to NeurIPS 2021 conference by Kwanyoung Kim and Jong Chul Ye, "Noise2Score: Tweedie’s Approach to Self-Supervised Image
Denoising without Clean Images" was published with the goal of unifying previouse image-denoising neural network architectures that don't rely on clean training data. 

The method relies on a U-Net Architecture that learns an encoding map of the noisy images to a score function $l'(y)$ that is trained with respect to an AR-DAE Loss(amortized residual denoising autoencoder). The DAE and AR-DAE loss had both previously been shown to be used for score-function learning but around the time the paper came out, AR-DAE had shown to be a more stable loss function to train with. After learning the score function, the model retrieves the final denoised image by deterministically calculating the expected original values of the unoised image pixels using Tweedie's Formula for exponential probability distributions(which each correspond to their own exponential noise)

This repository reproduces the Gaussian noise $\sigma=25$  result on CBSD68, Kodak, and Set 12 datasets. 

Here is a visualization of our result: 
<img src='assets/img11.png'>

| Dataset | Mean PSNR Value | 
| ------- | --------- |
| CBSD68  | 23.037    |
| Kodak   | 23.013    |
| Set12   | 22.962    |



## To Install this repo
1. Clone the Repository
2. Create the virtual environment: `python -m venv venv`
3. Install dependencies by running: `pip install -r requirements.txt` 
