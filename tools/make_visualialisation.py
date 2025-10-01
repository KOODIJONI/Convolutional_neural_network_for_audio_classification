import visualkeras
from tensorflow.keras import layers, models
import visualkeras
input_shape = (124, 129, 1) 
# Example model
model = models.Sequential([
    layers.Input(shape=(124, 129, 1)),
    layers.Resizing(32, 32),
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])

# Visualize
visualkeras.layered_view(model, to_file='better_model.png', legend=True)
