"""
Convolutional Autoencoder model for image denoising.
Architecture: simple encoder-decoder with Conv2D + MaxPooling / UpSampling.
Kept small on purpose so it trains quickly on CPU/Colab for a mini-project.
"""

from tensorflow.keras import layers, models


def build_autoencoder(input_shape=(28, 28, 1)):
    """Builds and compiles a convolutional autoencoder.

    Args:
        input_shape: shape of a single input image, e.g. (28, 28, 1) for MNIST.

    Returns:
        A compiled tf.keras.Model.
    """
    inp = layers.Input(shape=input_shape, name="noisy_input")

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inp)
    x = layers.MaxPooling2D((2, 2), padding="same")(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    encoded = layers.MaxPooling2D((2, 2), padding="same", name="bottleneck")(x)

    # Decoder
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(
        input_shape[-1], (3, 3), activation="sigmoid", padding="same", name="denoised_output"
    )(x)

    autoencoder = models.Model(inp, decoded, name="conv_autoencoder")
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
    return autoencoder


if __name__ == "__main__":
    model = build_autoencoder()
    model.summary()
