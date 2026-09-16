"""
Evaluates the trained denoising autoencoder on the MNIST test set:
- Computes MSE / PSNR / SSIM for (noisy vs original) as a baseline
  and (denoised vs original) for the model.
- Saves a side-by-side visualization: original / noisy / denoised.

Usage (from the src/ directory):
    python evaluate.py --model_path ../saved_model/denoising_autoencoder_final.h5
"""

import os
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from data_utils import load_mnist_noisy
from metrics import compute_metrics


def visualize(originals, noisy, denoised, n=10, save_path=None):
    plt.figure(figsize=(20, 6))
    for i in range(n):
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(originals[i].squeeze(), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(noisy[i].squeeze(), cmap="gray")
        plt.title("Noisy")
        plt.axis("off")

        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(denoised[i].squeeze(), cmap="gray")
        plt.title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print("Saved visualization to", save_path)


def main(args):
    (_, _), (x_test, x_test_noisy) = load_mnist_noisy(noise_factor=args.noise_factor)

    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"No trained model found at {args.model_path}. Run train.py first."
        )

    autoencoder = load_model(args.model_path)

    denoised = autoencoder.predict(x_test_noisy)

    metrics_noisy = compute_metrics(x_test, x_test_noisy)      # baseline: no denoising
    metrics_denoised = compute_metrics(x_test, denoised)        # model output

    print("=== Baseline: Noisy vs Original (no denoising) ===")
    print(metrics_noisy)
    print("=== Model: Denoised vs Original ===")
    print(metrics_denoised)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Evaluated on MNIST test set (10,000 images)\n\n")
        f.write("Baseline (noisy vs original, i.e. no denoising applied):\n")
        f.write(str(metrics_noisy) + "\n\n")
        f.write("Model (denoised vs original):\n")
        f.write(str(metrics_denoised) + "\n")
    print("Saved metrics to", metrics_path)

    visualize(
        x_test, x_test_noisy, denoised,
        n=10, save_path=os.path.join(args.output_dir, "sample_results.png")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the trained denoising autoencoder")
    parser.add_argument("--model_path", type=str, default="../saved_model/denoising_autoencoder_final.h5")
    parser.add_argument("--noise_factor", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default="../outputs")
    args = parser.parse_args()
    main(args)
