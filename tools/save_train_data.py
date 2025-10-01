import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Loading and Preprocessing ---

# Download and extract the mini_speech_commands dataset
data_dir = keras.utils.get_file(
    'mini_speech_commands.zip',
    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    extract=True,
    cache_dir='.', cache_subdir='data')
data_dir = 'data/mini_speech_commands_extracted/mini_speech_commands' 

# Load the data using audio_dataset_from_directory
train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
    directory=data_dir,
    batch_size=64,
    validation_split=0.2,
    seed=0,
    output_sequence_length=16000,
    subset='both')

label_names = np.array(train_ds.class_names)
print("label names:", label_names)

# Function to squeeze the extra channel dimension
def squeeze(audio, labels):
  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

# Apply the squeeze function to the datasets
train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

# Split the validation dataset into validation and test sets
test_ds = val_ds.shard(num_shards=2, index=0)
val_ds = val_ds.shard(num_shards=2, index=1)

# Function to convert a waveform to a spectrogram
def get_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

# Function to create spectrogram datasets from the audio datasets
def make_spec_ds(ds):
  return ds.map(
      map_func=lambda audio, label: (get_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE)

# Convert all datasets to spectrogram datasets
train_spectrogram_ds = make_spec_ds(train_ds)
val_spectrogram_ds = make_spec_ds(val_ds)
test_spectrogram_ds = make_spec_ds(test_ds)

# --- Save example spectrogram and labels to a file ---
# This step is crucial for running your custom model code from a separate script.
# It saves an example batch of spectrograms and their labels for testing.
for example_spectrograms, example_spect_labels in train_spectrogram_ds.take(1):
  break

np.save('spectrograms_data.npy', example_spectrograms.numpy())
np.save('labels.npy', example_spect_labels.numpy())

print("\nData loading and preprocessing complete. Spectrograms and labels saved to .npy files.")
print("The data is ready to be used by your model.")