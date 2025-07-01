"""
train_autoencoder.py

This script reads 64x64 seismic patches and trains a convolutional autoencoder model
to reconstruct input patches. The goal is to reduce noise and extract deep features
from seismic images, useful for applications such as anomaly detection and segmentation.

Author: Hande Çağatay
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam

# -------------------------------
# 🧩 Load Seismic Patches
# -------------------------------

# NOTE: Replace this line with your actual patch array loading
# For example: X = np.load("patches.npy")
# Here we simulate dummy data as an example
X = np.random.rand(1000, 64, 64, 1).astype("float32")

# Normalize values between 0 and 1
X = X / np.max(X)

# -------------------------------
# 🔧 Define Autoencoder Architecture
# -------------------------------

input_img = Input(shape=(64, 64, 1))  # Input patch size: 64x64x1

# Encoder
x = Conv2D(32, (3, 3), activation="relu", padding="same")(input_img)
x = MaxPooling2D((2, 2), padding="same")(x)
x = Conv2D(16, (3, 3), activation="relu", padding="same")(x)
encoded = MaxPooling2D((2, 2), padding="same")(x)
# Encoded shape: 16x16x16

# Decoder
x = Conv2D(16, (3, 3), activation="relu", padding="same")(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)

# Define and compile the model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer=Adam(), loss="mse")

# -------------------------------
# 🧠 Train the Model
# -------------------------------

history = autoencoder.fit(
    X, X,
    epochs=20,
    batch_size=32,
    shuffle=True,
    validation_split=0.1
)

# -------------------------------
# 🎨 Visualize Reconstruction Results
# -------------------------------

decoded_imgs = autoencoder.predict(X[:10])

n = 10  # number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Original patch
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(X[i].reshape(64, 64), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Reconstructed patch
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(64, 64), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.savefig("reconstruction_result.png")  # Save output as PNG
plt.show()
