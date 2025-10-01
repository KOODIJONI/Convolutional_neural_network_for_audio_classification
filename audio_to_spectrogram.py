import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import stft

def get_spectrogram(waveform):
  spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def plot_spectrogram(spectrogram, ax):
  if len(spectrogram.shape) > 2:
    assert len(spectrogram.shape) == 3
    spectrogram = np.squeeze(spectrogram, axis=-1)
  log_spec = np.log(spectrogram.T + np.finfo(float).eps)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)

def path_to_spectrogram(file_path):
    """
    Muuttaa wavtiedoston spectrogrammiksi

    Args:
        file_path: Polku tiedostoon.

    Returns:
       Spectrogram tensori .
    """
    audio_binary = tf.io.read_file(file_path)
    audio, sample_rate = tf.audio.decode_wav(audio_binary, desired_channels=1, desired_samples=16000)

    waveform = tf.squeeze(audio, axis=-1)
    spectrogram = get_spectrogram(waveform)

    plt.figure(figsize=(10, 4))
    plot_spectrogram(spectrogram.numpy(), plt.gca())
    plt.title(f'Spectrogram for: {file_path}')

    spectrogram = spectrogram[tf.newaxis, ...]
    return spectrogram.numpy()



