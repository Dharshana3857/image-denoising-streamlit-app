# Image Denoising Using Convolutional Autoencoders

A college mini-project (Neural Networks and Deep Learning) that trains a
convolutional autoencoder to remove artificial Gaussian noise from MNIST
handwritten-digit images, evaluates it with MSE / PSNR / SSIM, and provides
a Streamlit app to try the trained model on sample digits (and, optionally,
your own digit-style images).

**Scope note:** the model is trained only on MNIST (28x28 grayscale
handwritten digits). This is a focused mini-project, not a general-purpose
photo denoiser — see the "Streamlit Demo" section below for what that means
in practice.

## Project Structure

```
Image_Denoising_Autoencoder_MiniProject/
├── README.md
├── requirements.txt
├── src/
│   ├── model.py         # Autoencoder architecture (build_autoencoder)
│   ├── data_utils.py    # Loads MNIST, generates noisy versions
│   ├── metrics.py        # MSE / PSNR / SSIM computation
│   ├── train.py           # Training script (saves model + loss curve)
│   └── evaluate.py        # Evaluation script (saves metrics + visualization)
├── app/
│   └── streamlit_app.py   # Streamlit demo (see "Streamlit Demo" section)
├── notebooks/
│   └── Image_Denoising_Colab.ipynb  # Self-contained Colab training notebook
├── saved_model/            # Trained model (.h5) goes here — created by train.py
├── outputs/                 # loss_curve.png, sample_results.png, metrics.txt
└── report/
    └── Project_Report_Outline.md   # Skeleton for your written report
```

## How It Works

1. **Data**: MNIST digits (28x28 grayscale) are normalized to `[0, 1]`.
2. **Noise**: Artificial Gaussian noise (configurable `noise_factor`,
   default `0.5`) is added to each image and clipped back to `[0, 1]`.
3. **Model**: A small convolutional autoencoder (Conv2D + MaxPooling
   encoder, Conv2D + UpSampling decoder) learns to map noisy images back
   to their clean originals.
4. **Training**: Adam optimizer, binary crossentropy loss, a validation
   split, early stopping, and checkpointing on best validation loss.
5. **Evaluation**: MSE, PSNR, and SSIM are computed on the full test set,
   comparing (a) noisy vs. original (baseline, no denoising) and
   (b) denoised vs. original (the model's actual performance). See
   "Evaluation Pipeline" below for exactly how each metric is computed.
6. **Demo**: A Streamlit app lets you run the trained model on real MNIST
   test images (with live MSE/PSNR/SSIM, since a genuine clean reference
   exists) or, optionally, on your own uploaded digit-style image
   (qualitative only — see below).

No results are pre-filled anywhere in this project — every number and
image in `outputs/` is generated when you actually run training/evaluation
locally or in Colab.

## Evaluation Pipeline (MSE / PSNR / SSIM)

`src/metrics.py` defines `compute_metrics(originals, reconstructions)`,
used by both `src/evaluate.py` and the Streamlit app, so there is a single
source of truth for how metrics are computed:

- Inputs are batches of images, shape `(N, H, W, C)`, pixel values in `[0, 1]`.
- For each image pair, `.squeeze()` drops the channel dimension (grayscale),
  then:
  - **MSE** = `mean((original - reconstruction) ** 2)`
  - **PSNR** = `skimage.metrics.peak_signal_noise_ratio(original, reconstruction, data_range=1.0)`
  - **SSIM** = `skimage.metrics.structural_similarity(original, reconstruction, data_range=1.0)`
- The final reported values are the mean of each metric across all `N` images.

`src/evaluate.py` runs this twice on the full 10,000-image MNIST test set:
- **Baseline**: noisy images vs. originals (i.e., what the metrics look
  like with *no* denoising at all).
- **Model**: the autoencoder's denoised output vs. originals.

Comparing these two tells you how much the model actually improved things,
rather than just reporting one number in isolation. `data_range=1.0` is
used consistently because all images are normalized floats in `[0, 1]`.

## Setup

### macOS / Linux

```bash
# 1. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

### Windows

```powershell
# 1. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

Notes for Windows:
- Use `python` (not `python3`) unless your installation specifically
  registered `python3` — check with `python --version` first.
- If `venv\Scripts\activate` is blocked by PowerShell's execution policy,
  either run it from Command Prompt (`venv\Scripts\activate.bat`) instead,
  or run PowerShell as Administrator once and execute:
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- All commands below (`python train.py`, `streamlit run ...`, etc.) are
  identical on Windows once the virtual environment is activated — just
  use `python` instead of `python3` if you normally would.

## Option A: Train Locally (macOS / Linux / Windows)

```bash
cd src
python train.py --epochs 20 --batch_size 128 --noise_factor 0.5
```

(On Windows this is the same command — `python` instead of `python3`.)

This will:
- Download MNIST automatically (via `tensorflow.keras.datasets.mnist`;
  needs internet on first run).
- Train the autoencoder, using early stopping (patience=5).
- Save the best checkpoint to `saved_model/denoising_autoencoder.h5`.
- Save the final model to `saved_model/denoising_autoencoder_final.h5`.
- Save a training/validation loss plot to `outputs/loss_curve.png`.

Then evaluate:

```bash
python evaluate.py --model_path ../saved_model/denoising_autoencoder_final.h5
```

This will:
- Print and save MSE / PSNR / SSIM (baseline vs. model) to
  `outputs/metrics.txt`.
- Save a visualization grid (original / noisy / denoised) to
  `outputs/sample_results.png`.

## Option B: Train on Google Colab (Step by Step)

1. Go to [colab.research.google.com](https://colab.research.google.com/)
   and sign in with a Google account.
2. `File > Upload notebook...` and select
   `notebooks/Image_Denoising_Colab.ipynb` from this project (or drag-and-drop it).
3. *(Optional, for a small speedup)* `Runtime > Change runtime type >
   Hardware accelerator > T4 GPU > Save`. CPU also works fine for this
   small model/dataset.
4. `Runtime > Run all` (or run each cell individually with Shift+Enter,
   top to bottom). The notebook is fully self-contained — it re-defines
   the model, loads MNIST, trains, evaluates, and visualizes, mirroring
   `src/` exactly, and needs nothing else uploaded.
5. Training takes roughly a few minutes on CPU (faster on GPU). Watch the
   per-epoch `loss` / `val_loss` printed by cell 5 ("Train the Model").
6. The evaluation cell prints real MSE / PSNR / SSIM for both the noisy
   baseline and the trained model, computed on the full MNIST test set.
7. The final cell (cell 9, "Download Everything") bundles the trained
   model plus `loss_curve.png`, `sample_results.png`, and `metrics.txt`
   into one zip and downloads it through your browser.
8. Unzip that download locally and:
   - copy `denoising_autoencoder_final.h5` into this project's
     `saved_model/` folder,
   - copy `loss_curve.png`, `sample_results.png`, and `metrics.txt` into
     this project's `outputs/` folder.
9. You can now run the Streamlit demo locally (see below) using the
   Colab-trained model, and use the real numbers/images in your report.

This means the entire pipeline — training, validation, evaluation, and
visualization — can be run start to finish inside Google Colab alone,
with no local Python environment required until you want to run the
Streamlit demo.

## Streamlit Demo

Make sure a trained model exists at
`saved_model/denoising_autoencoder_final.h5` (from either Option A or B
above), then from the project root:

```bash
streamlit run app/streamlit_app.py
```

(Same command on Windows, inside the activated virtual environment.)

This opens a local web page with two modes:

- **Sample MNIST test image (recommended)** — pick any image from the
  real MNIST test set, apply adjustable Gaussian noise, and see the
  original / noisy / denoised images side by side, along with **live
  MSE / PSNR / SSIM** computed against the real clean original (using the
  exact same `compute_metrics()` function as `src/evaluate.py`).
- **Upload your own digit image (experimental)** — upload a single
  handwritten digit, ideally cropped and on a plain background. Because
  MNIST digits are light strokes on a dark background, an "invert colors"
  option is provided for photos of pen-on-paper digits. **No MSE/PSNR/SSIM
  is shown in this mode**, because there is no known clean reference image
  to compare against for an arbitrary upload — only a qualitative preview
  is shown.

**This app is intentionally scoped to MNIST-style digit images.** The
model was never trained on natural photos, faces, or general objects, so
it is not presented as (and should not be used as) a general-purpose photo
denoiser — results on such images will look poor, which is expected and
worth discussing as a limitation in your report.

## Full Run-Through (Start to Finish, Local)

```bash
# 1. Set up environment (macOS/Linux shown; see Windows section above for venv activation)
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Train
cd src
python train.py

# 3. Evaluate
python evaluate.py

# 4. Check results
#    - outputs/loss_curve.png
#    - outputs/sample_results.png
#    - outputs/metrics.txt

# 5. Launch the demo app
cd ..
streamlit run app/streamlit_app.py
```

## Full Run-Through (Start to Finish, Google Colab Only)

1. Upload `notebooks/Image_Denoising_Colab.ipynb` to Google Colab.
2. `Runtime > Run all`.
3. Review the printed metrics and the inline loss-curve / sample-results
   plots.
4. Run the final cell to download a zip with the trained model and all
   output files.
5. *(Optional)* Bring the downloaded model into `saved_model/` locally to
   run the Streamlit demo, as described above.

## Customization

- Change noise strength: `--noise_factor` in `train.py` / `evaluate.py`
  (both must match, or retrain, to keep results consistent) — or
  `NOISE_FACTOR` at the top of the Colab notebook.
- Change training length: `--epochs`, `--batch_size` (locally), or
  `EPOCHS` / `BATCH_SIZE` in the notebook.
- Swap dataset: the architecture in `src/model.py` also works for
  CIFAR-10 with minor changes (input shape `(32, 32, 3)`, and switching
  the final activation/loss if desired) — left as a possible extension
  (see `report/Project_Report_Outline.md`, Future Work section).

## Requirements

See `requirements.txt`. Core dependencies: TensorFlow/Keras, NumPy,
Matplotlib, scikit-image (for PSNR/SSIM), Streamlit, Pillow.
