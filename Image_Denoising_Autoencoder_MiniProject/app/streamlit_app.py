"""
Streamlit demo for the MNIST Handwritten-Digit Denoising Autoencoder.

IMPORTANT SCOPE NOTE: the underlying model is trained ONLY on the MNIST
dataset (28x28 grayscale images of a single handwritten digit, rendered as
white strokes on a black background). This app is a demo of that specific
model, not a general-purpose photo denoiser. It intentionally leads with a
"try a real MNIST test image" mode (where we have a known clean original,
so real MSE/PSNR/SSIM can be shown), and treats arbitrary image uploads as
a clearly-labeled, best-effort/experimental mode.

Run (from the project root, after training a model):
    streamlit run app/streamlit_app.py
"""

import os
import sys

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist

# Make src/ importable so the app reuses the exact same metrics code
# used by src/evaluate.py (single source of truth for MSE/PSNR/SSIM).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from metrics import compute_metrics  # noqa: E402

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "saved_model", "denoising_autoencoder_final.h5"
)


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data
def get_mnist_test_set():
    """Loads the real MNIST test set (needs internet on first run, same as training)."""
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = x_test.astype("float32") / 255.0
    x_test = np.expand_dims(x_test, -1)
    return x_test, y_test


def add_noise(arr: np.ndarray, noise_factor: float) -> np.ndarray:
    rng = np.random.default_rng()
    noisy = arr + noise_factor * rng.normal(size=arr.shape)
    return np.clip(noisy, 0.0, 1.0).astype("float32")


def preprocess_uploaded_image(image: Image.Image, invert: bool) -> np.ndarray:
    """Converts an uploaded image to a (1, 28, 28, 1) float32 array in [0, 1]."""
    image = image.convert("L")
    image = image.resize((28, 28))
    arr = np.array(image).astype("float32") / 255.0
    if invert:
        arr = 1.0 - arr  # MNIST is white digit on black background
    arr = np.expand_dims(arr, axis=(0, -1))
    return arr


st.set_page_config(page_title="MNIST Digit Denoising Autoencoder", layout="centered")
st.title("MNIST Handwritten-Digit Denoising Autoencoder")
st.caption(
    "This demo runs a convolutional autoencoder trained **only** on MNIST "
    "(28x28 grayscale images, white digit strokes on a black background). "
    "It is **not** a general-purpose photo denoiser — it doesn't know about "
    "faces, objects, or color photos, and results on anything outside the "
    "MNIST domain should be expected to be poor. Use the sample-image mode "
    "below to see the model perform on data it was actually trained for."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Trained model not found at `{MODEL_PATH}`.\n\n"
        "Train it first: run `src/train.py` locally, or use the Colab "
        "notebook in `notebooks/`, then place "
        "`denoising_autoencoder_final.h5` in the `saved_model/` folder."
    )
    st.stop()

model = get_model()

mode = st.radio(
    "Choose a demo mode",
    [
        "Sample MNIST test image (recommended — shows real MSE/PSNR/SSIM)",
        "Upload your own digit image (experimental, qualitative only)",
    ],
)

noise_factor = st.slider("Gaussian noise factor to apply", 0.0, 1.0, 0.5, 0.05)

st.divider()

if mode.startswith("Sample MNIST"):
    x_test, y_test = get_mnist_test_set()
    idx = st.slider("Pick a test-set image index", 0, len(x_test) - 1, 0)
    st.write(f"True digit label: **{int(y_test[idx])}**")

    original = x_test[idx : idx + 1]
    noisy = add_noise(original, noise_factor)
    denoised = model.predict(noisy, verbose=0)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(original.squeeze(), clamp=True, width=150)
    with col2:
        st.subheader("Noisy")
        st.image(noisy.squeeze(), clamp=True, width=150)
    with col3:
        st.subheader("Denoised")
        st.image(denoised.squeeze(), clamp=True, width=150)

    # Real metrics: we have a genuine clean original here, so these are
    # computed live, using the same compute_metrics() as src/evaluate.py.
    m = compute_metrics(original, denoised)
    st.write("**Denoising quality for this image (denoised vs. original):**")
    st.write(f"MSE: `{m['mse']:.5f}`  |  PSNR: `{m['psnr']:.2f} dB`  |  SSIM: `{m['ssim']:.4f}`")
    st.caption(
        "These numbers are computed live for this single image and will vary "
        "by image and by the random noise draw — they are not fixed/pre-recorded values."
    )

else:
    st.write(
        "For the best (still imperfect) results, upload a tightly-cropped image "
        "of a single handwritten digit on a plain background — as close to the "
        "MNIST style as possible."
    )
    invert = st.checkbox(
        "Invert colors (use this if your digit is dark strokes on a light background, "
        "e.g. pen on paper — MNIST digits are light strokes on a dark background)",
        value=True,
    )
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        original_arr = preprocess_uploaded_image(image, invert=invert)
        noisy_arr = add_noise(original_arr, noise_factor)
        denoised_arr = model.predict(noisy_arr, verbose=0)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Input (resized to 28x28)")
            st.image(original_arr.squeeze(), clamp=True, width=150)
        with col2:
            st.subheader("Noisy")
            st.image(noisy_arr.squeeze(), clamp=True, width=150)
        with col3:
            st.subheader("Denoised")
            st.image(denoised_arr.squeeze(), clamp=True, width=150)

        st.warning(
            "No MSE/PSNR/SSIM is shown here because there is no known clean "
            "'ground truth' version of your uploaded image to compare against — "
            "those metrics require a genuine reference image, which only the "
            "MNIST test set provides in this project. This mode is for a quick, "
            "qualitative look only, and quality will depend heavily on how close "
            "your image is to a real MNIST-style digit."
        )
    else:
        st.write("Upload an image above to try it.")
