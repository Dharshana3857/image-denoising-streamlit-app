# MNIST Image Denoising using Convolutional Autoencoders

A college mini-project that trains a Convolutional Autoencoder to remove artificially added Gaussian noise from MNIST handwritten digit images, evaluates it using MSE, PSNR, and SSIM, and demonstrates it through an interactive Streamlit web application.

Note: This model is trained only on MNIST (28x28 grayscale handwritten digits). It is a focused academic mini-project, not a general-purpose photo denoiser.

## Overview

Digital images are often affected by noise during acquisition, transmission, or storage, which can reduce their visual clarity and hurt downstream tasks such as digit recognition and OCR. This project applies a Convolutional Autoencoder to learn how to reconstruct clean handwritten digit images from their noisy counterparts, using the MNIST dataset as a lightweight, well-understood benchmark suitable for an academic mini-project.

The pipeline covers the full cycle: synthetic noise generation, model training with validation and early stopping, quantitative evaluation (MSE / PSNR / SSIM), and an interactive web demo.

## Features

- Convolutional Autoencoder built with TensorFlow/Keras (Conv2D + MaxPooling encoder, Conv2D + UpSampling decoder)
- Synthetic Gaussian noise generation with configurable noise factor
- Training with validation split, early stopping, and best-model checkpointing
- Quantitative evaluation using MSE, PSNR, and SSIM (via scikit-image)
- Interactive Streamlit web app with two demo modes (sample MNIST image / upload your own digit)
- Self-contained Google Colab notebook for training end-to-end in the browser

## Tech Stack

- Language: Python
- Deep Learning: TensorFlow / Keras
- Numerical Computing: NumPy
- Evaluation Metrics: scikit-image (PSNR, SSIM)
- Visualization: Matplotlib
- Web App: Streamlit
- Dataset: MNIST (60,000 train / 10,000 test images)

## Project Structure

## Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

## Usage

### Train Locally

```bash
cd src
python train.py --epochs 20 --batch_size 128 --noise_factor 0.5
python evaluate.py
```

### Train on Google Colab

1. Upload `notebooks/Image_Denoising_Colab.ipynb` to Google Colab.
2. Run all cells.
3. Download the trained model and outputs.

### Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

## Model Architecture

Encoder:
- Conv2D (32 filters, 3x3, ReLU) -> MaxPooling2D (2x2)
- Conv2D (32 filters, 3x3, ReLU) -> MaxPooling2D (2x2) - bottleneck

Decoder:
- Conv2D (32 filters, 3x3, ReLU) -> UpSampling2D (2x2)
- Conv2D (32 filters, 3x3, ReLU) -> UpSampling2D (2x2)
- Conv2D (1 filter, 3x3, Sigmoid) - final denoised output

Trained with the Adam optimizer and Binary Crossentropy loss, using early stopping based on validation loss.

## Evaluation Metrics

- MSE (Mean Squared Error) - average squared pixel-wise difference; lower is better.
- PSNR (Peak Signal-to-Noise Ratio, dB) - higher is better.
- SSIM (Structural Similarity Index) - perceptual similarity; closer to 1 is better.

## Results

| Evaluation Scope | MSE | PSNR (dB) | SSIM |
|---|---|---|---|
| Noisy vs Original (Baseline) | | | |
| Denoised vs Original (Model) | | | |

## Limitations

- Trained and evaluated only on MNIST (28x28 grayscale, single handwritten digit per image).
- Not expected to perform well on natural photos, color images, or multi-object scenes.
- The image upload feature in the Streamlit app is for qualitative demonstration only, since no clean reference is available for uploaded images.

## Future Work

- Extend to CIFAR-10 or real-world noisy datasets.
- Experiment with other noise types (salt-and-pepper, speckle).
- Explore deeper architectures (U-Net-based autoencoders, GAN-based denoising).

## License

This project is licensed under the MIT License.
