import numpy as np
from tensorflow import keras
from keras import layers

# Define the same model architecture as used for training
input_shape = (32, 32, 1)
num_labels = 8

model_for_loading = keras.Sequential([
    keras.Input(shape=input_shape),
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="valid"),
    layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="valid"),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_labels, activation="softmax")
])

# Load saved Keras weights
model_for_loading.load_weights("my_model_weights.weights.h5")

# Loop through trainable layers and save weights and biases separately
for idx, layer in enumerate(model_for_loading.layers):
    if layer.weights:  # Only save layers that have weights
        weights, biases = layer.get_weights()
        np.save(f"layer{idx}_weights.npy", weights)
        np.save(f"layer{idx}_biases.npy", biases)
        print(f"Saved weights and biases for layer {idx} ({layer.name})")