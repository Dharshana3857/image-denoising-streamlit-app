"""
Data loading utilities: loads MNIST and generates Gaussian-noise-corrupted
versions of the images for the denoising autoencoder task.
"""

import numpy as np
from tensorflow.keras.datasets import mnist


def load_mnist_noisy(noise_factor: float = 0.5, seed: int = 42):
    """Loads MNIST and creates artificially noisy versions of train/test sets.

    Args:
        noise_factor: standard deviation multiplier for the Gaussian noise added.
        seed: RNG seed for reproducibility.

    Returns:
        (x_train, x_train_noisy), (x_test, x_test_noisy)
        All arrays are float32 in [0, 1] with shape (N, 28, 28, 1).
    """
    (x_train, _), (x_test, _) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    rng = np.random.default_rng(seed)

    x_train_noisy = x_train + noise_factor * rng.normal(size=x_train.shape)
    x_test_noisy = x_test + noise_factor * rng.normal(size=x_test.shape)

    x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0).astype("float32")
    x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0).astype("float32")

    return (x_train, x_train_noisy), (x_test, x_test_noisy)


if __name__ == "__main__":
    (x_train, x_train_noisy), (x_test, x_test_noisy) = load_mnist_noisy()
    print("x_train:", x_train.shape, "x_train_noisy:", x_train_noisy.shape)
    print("x_test:", x_test.shape, "x_test_noisy:", x_test_noisy.shape)
