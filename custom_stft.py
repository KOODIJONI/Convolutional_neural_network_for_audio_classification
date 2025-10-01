import numpy as np
def short_time_fourier_transform(waveform,frame_length,frame_step):
    frames_fft = np.array([
        fft_iterative(waveform[i:i+frame_length] * np.hanning(frame_length))
        for i in range(0, len(waveform) - frame_length + 1, frame_step)
    ])
    return np.array(frames_fft)

def fft_iterative(x):
    #Cooley–Tukey FFT algorithm
    x = np.array(x, dtype=complex)
    n = x.shape[0]
    next_pow2 = int(2 ** np.ceil(np.log2(n)))
    if n != next_pow2:
        x = np.pad(x, (0, next_pow2 - n))
        n = next_pow2
    indices = np.arange(n)
    bits = int(np.log2(n))
    rev_indices = np.array([int(f"{i:0{bits}b}"[::-1], 2) for i in indices])
    X = x[rev_indices].copy()

    m = 2
    while m <= n:
        wm = np.exp(-2j * np.pi / m)
        for k in range(0, n, m):
            w = 1
            for j in range(m // 2):
                t = w * X[k + j + m // 2]
                u = X[k + j]
                X[k + j] = u + t
                X[k + j + m // 2] = u - t
                w *= wm
        m*= 2

    return X