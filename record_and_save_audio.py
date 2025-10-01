import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import time
def list_audio_devices():
    """Listaa kaikki laitteet."""
    devices = sd.query_devices()
    input_devices = []
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:  
            print(f"{i}: {dev['name']} ({dev['max_input_channels']} channels)")
            input_devices.append((i, dev['name']))
    return input_devices
def record_audio_to_wav(duration, device_index, progress_callback=None, counter_callback=None, output_file="output.wav"):
    for i in range(3):
        if counter_callback:
            counter_callback(f"starting: {i+1}")
        time.sleep(1)

    recording = sd.rec(
        int(duration * 16000),
        samplerate=16000,
        channels=1,
        dtype="float32",
        device=device_index
    )

    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed >= duration:
            break
        if progress_callback:
            progress_callback(elapsed*100)
        if counter_callback:
            counter_callback(f"Counter: {int(elapsed)+1}")
        time.sleep(0.05)
        print(elapsed)

    sd.wait()

    recording_int16 = np.int16(recording * 32767)
    write(output_file, 16000, recording_int16)

    if counter_callback:
        counter_callback("Recording done")

    return output_file