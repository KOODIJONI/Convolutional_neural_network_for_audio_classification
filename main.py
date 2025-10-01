import tkinter as tk
from tkinter import ttk, messagebox
import threading
from record_and_save_audio import list_audio_devices, record_audio_to_wav
from machine_learning_model import predict_with_path
class AudioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Recorder & Predictor")
        self.keywords_label = tk.Label(root, text="Keywords: ['down', 'go', 'left', 'no', 'right', 'stop', 'up', 'yes']", font=("Arial", 12))
        self.keywords_label.pack(pady=5)

        tk.Label(root, text="Select Input Device:", font=("Arial", 11)).pack()
        self.devices = list_audio_devices()
        if not self.devices:
            messagebox.showerror("Device Error", "No input devices found!")
            self.root.destroy()
            return
        self.device_names = [f"{i}: {name}" for i, name in self.devices]

        self.device_combo = ttk.Combobox(root, values=self.device_names, state="readonly")
        self.device_combo.current(0)
        self.device_combo.pack(pady=5)

        self.counter_label = tk.Label(root, text="Counter: 0", font=("Arial", 12))
        self.counter_label.pack(pady=5)

        self.progress = ttk.Progressbar(root, orient="horizontal", length=250, mode="determinate")
        self.progress.pack(pady=5)

        self.prediction_label = tk.Label(root, text="Prediction: None", font=("Arial", 12, "bold"))
        self.prediction_label.pack(pady=5)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=10)

        self.record_button = tk.Button(button_frame, text="Record", command=self.start_recording)
        self.record_button.grid(row=0, column=0, padx=5)

        self.predict_button = tk.Button(button_frame, text="Predict", command=self.predict)
        self.predict_button.grid(row=0, column=1, padx=5)



    def start_recording(self, duration=1):
        threading.Thread(target=self._record, args=(duration,), daemon=True).start()

    def _record(self, duration):
        try:
            selected = self.device_combo.get().split(":")[0]
            device_index = int(selected)

            file_path = record_audio_to_wav(
                duration=duration,
                device_index=device_index,
                progress_callback=lambda elapsed: (
                    self.progress.config(value=elapsed),
                    self.progress.update_idletasks()   
                ),
                counter_callback=lambda text: (
                    self.counter_label.config(text=text),
                    self.counter_label.update_idletasks() 
                ),
                output_file="output.wav"
            )

            self.progress["value"] = 0
            self.predict()

        except Exception as e:
            messagebox.showerror("Recording Error", f"An error occurred:\n{e}")
        

    def predict(self):
        try:
            prediction = predict_with_path("output.wav")
            self.prediction_label.config(text=f"Prediction: {prediction}")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred:\n{e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioApp(root)
    root.mainloop()
