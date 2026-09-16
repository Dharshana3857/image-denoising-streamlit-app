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
