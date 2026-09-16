"""
Trains the convolutional autoencoder to denoise MNIST digits.

Usage (from the src/ directory):
    python train.py --epochs 20 --batch_size 128 --noise_factor 0.5

Produces:
    ../saved_model/denoising_autoencoder_final.h5   (final trained model)
    ../saved_model/denoising_autoencoder.h5          (best checkpoint by val_loss)
    ../outputs/loss_curve.png                        (training/validation loss plot)
"""

import os
import argparse

import matplotlib
matplotlib.use("Agg")  # safe for headless environments
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from model import build_autoencoder
from data_utils import load_mnist_noisy


def main(args):
    print("Loading MNIST and generating noisy versions...")
    (x_train, x_train_noisy), (x_test, x_test_noisy) = load_mnist_noisy(
        noise_factor=args.noise_factor
    )

    # Carve out a validation split from the training set
    n_val = int(len(x_train) * args.val_split)
    x_val, x_val_noisy = x_train[:n_val], x_train_noisy[:n_val]
    x_train_fit, x_train_noisy_fit = x_train[n_val:], x_train_noisy[n_val:]

    print(f"Train: {x_train_fit.shape[0]} | Val: {x_val.shape[0]} | Test: {x_test.shape[0]}")

    autoencoder = build_autoencoder(input_shape=x_train.shape[1:])
    autoencoder.summary()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.model_dir, "denoising_autoencoder.h5")

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True, verbose=1),
    ]

    history = autoencoder.fit(
        x_train_noisy_fit,
        x_train_fit,
        epochs=args.epochs,
        batch_size=args.batch_size,
        shuffle=True,
        validation_data=(x_val_noisy, x_val),
        callbacks=callbacks,
        verbose=2,
    )

    final_path = os.path.join(args.model_dir, "denoising_autoencoder_final.h5")
    autoencoder.save(final_path)
    print("Saved final model to", final_path)

    # Plot and save the loss curve (real values from this training run only)
    plt.figure()
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Binary Crossentropy)")
    plt.title("Training vs Validation Loss")
    plt.legend()
    loss_curve_path = os.path.join(args.output_dir, "loss_curve.png")
    plt.savefig(loss_curve_path)
    print("Saved loss curve to", loss_curve_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a convolutional denoising autoencoder on MNIST")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--noise_factor", type=float, default=0.5, help="Std-dev multiplier for Gaussian noise")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--model_dir", type=str, default="../saved_model")
    parser.add_argument("--output_dir", type=str, default="../outputs")
    args = parser.parse_args()
    main(args)
