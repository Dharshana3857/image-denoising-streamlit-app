"""
Evaluation metrics for the denoising autoencoder: MSE, PSNR, and SSIM.
Computed per-image and averaged over a batch/dataset.
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def compute_metrics(originals: np.ndarray, reconstructions: np.ndarray) -> dict:
    """Computes average MSE, PSNR, and SSIM between two batches of images.

    Args:
        originals: array of shape (N, H, W, C), values in [0, 1].
        reconstructions: array of shape (N, H, W, C), values in [0, 1].

    Returns:
        dict with keys 'mse', 'psnr', 'ssim' (floats, averaged over N).
    """
    assert originals.shape == reconstructions.shape, "Shape mismatch between originals and reconstructions"

    mse_list, psnr_list, ssim_list = [], [], []

    for orig, recon in zip(originals, reconstructions):
        o = orig.squeeze()
        r = recon.squeeze()

        mse_val = float(np.mean((o - r) ** 2))
        psnr_val = float(psnr(o, r, data_range=1.0))
        ssim_val = float(ssim(o, r, data_range=1.0))

        mse_list.append(mse_val)
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)

    return {
        "mse": float(np.mean(mse_list)),
        "psnr": float(np.mean(psnr_list)),
        "ssim": float(np.mean(ssim_list)),
    }
