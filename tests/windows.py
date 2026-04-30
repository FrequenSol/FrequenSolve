import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# helpers
# -----------------------


def cosine_tapered_gate(N, start, end, taper_len):
    """
    Cosine-tapered rectangular gate:
      0 ... (start-taper) -> ramp to 1 -> 1 between [start,end)
      -> ramp to 0 -> 0 ...
    start/end/taper_len are in *samples*.
    """
    w = np.zeros(N)
    start = max(0, int(start))
    end = min(N, int(end))
    if end <= start:
        return w
    w[start:end] = 1.0
    L = int(taper_len)
    if L > 0:
        # left taper
        L_left = min(L, start)
        if L_left > 0:
            idx = np.arange(start - L_left, start)
            n = np.arange(L_left)
            w[idx] = 0.5 - 0.5 * np.cos(np.pi * (n + 1) / (L_left + 1))
        # right taper
        L_right = min(L, N - end)
        if L_right > 0:
            idx = np.arange(end, end + L_right)
            n = np.arange(L_right)
            w[idx] = 0.5 + 0.5 * np.cos(np.pi * (n + 1) / (L_right + 1))
    return w


def hz_to_bin(f_hz, fs, N):
    """
    Map a frequency in Hz to the nearest DFT bin index in [0, N-1].
    Works for negative f as well (wraps into [0,N)).
    """
    k = int(np.round(f_hz * N / fs)) % N
    return k


def bins_from_hz(f_list_hz, fs, N):
    """Vectorized helper."""
    return [hz_to_bin(f, fs, N) for f in f_list_hz]


def add_hermitian_pairs(k_list, N):
    """
    Ensure Hermitian symmetry so time-domain result is (nearly) real:
    include each bin and its conjugate partner (-k) mod N.
    """
    keep = set()
    for k in k_list:
        kk = int(k) % N
        keep.add(kk)
        keep.add((-kk) % N)
    return sorted(keep)


def print_freq_table(k_list, fs, N, label="Kept bins"):
    def fk(k):
        # signed frequency in Hz associated to bin k
        k_signed = k if k <= N // 2 else k - N
        return k_signed * fs / N

    pairs = sorted({(k, fk(k)) for k in k_list})
    df = fs / N
    print(f"{label}: {len(pairs)} bins  |  Δf = {df:.6f} Hz  |  fs = {fs:.6f} Hz")
    head = min(len(pairs), 24)
    print("  (showing up to 24)  k  ->  f_k [Hz]")
    for k, f in pairs[:head]:
        print(f"  {k:6d} -> {f: .6f}")
    if len(pairs) > head:
        print("  ...")


def ricker_wavelet(t, f0):
    """Zero-phase Ricker wavelet centered at t=0 with peak frequency f0 (Hz)."""
    a = (np.pi * f0 * t) ** 2
    return (1 - 2 * a) * np.exp(-a)


def keep_center_k(W, K):
    """
    Keep K centered frequency bins (in the fftshifted sense); zero the rest.
    Works for real signals where low-freq content is centered.
    """
    N = len(W)
    if K >= N:
        return W.copy()
    W_shift = np.fft.fftshift(W)
    mid = N // 2
    half = K // 2
    if K % 2 == 0:
        sl = slice(mid - half, mid + half)
    else:
        sl = slice(mid - half, mid + half + 1)
    Wk_shift = np.zeros_like(W_shift)
    Wk_shift[sl] = W_shift[sl]
    return np.fft.ifftshift(Wk_shift)


def circ_conv_sparse(W_sparse, X):
    """
    Circular convolution in the frequency domain:
      Y[k] = sum_m W[m] * X[(k - m) mod N]
    Only sums over bins where W != 0.  (Scaling by 1/N is applied outside.)
    """
    N = len(W_sparse)
    Y = np.zeros(N, dtype=complex)
    m_keep = np.flatnonzero(np.abs(W_sparse) > 0)
    for m in m_keep:
        Y += W_sparse[m] * np.roll(X, m)  # X[k-m]
    return Y


# -----------------------
# main demo (tweak here)
# -----------------------


def main():
    N = 512  # number of samples
    dt = 0.002  # sample interval (s)
    fs = 1.0 / dt  # Hz
    gate_start_s = 0.2  # gate start time (s)
    gate_end_s = 0.3  # gate end time (s)
    taper_len_s = 0.05  # cosine taper length each side (s)
    f0 = 20.0  # Ricker peak frequency (Hz)
    noise_sigma = 0.01  # additive noise level
    ENFORCE_HERMITIAN = True

    t = np.arange(N) * dt

    # 1) Build the time mute window
    start = int(gate_start_s / dt)
    end = int(gate_end_s / dt)
    taper = int(taper_len_s / dt)
    w = cosine_tapered_gate(N, start, end, taper)

    W = np.fft.fft(w)

    # Plot the amplitude spectrum (positive frequencies only)
    freqs = np.fft.fftfreq(N, dt)
    pos_mask = (freqs >= 0) & (freqs < 50)
    plt.figure()
    plt.title("Window amplitude spectrum (positive frequencies)")
    plt.plot(freqs[pos_mask], np.abs(W[pos_mask]))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude |W(f)|")
    plt.grid(True)
    plt.show()

    # f_keep_hz = [0, 2, 4, 6, 8, 10, 12, 16, 20]   # edit me
    mask = (freqs >= 0) & (freqs < 3)
    f_keep_hz = freqs[mask]
    print(f_keep_hz)
    k_sel = bins_from_hz(f_keep_hz, fs, N)

    if ENFORCE_HERMITIAN:
        k_sel = add_hermitian_pairs(k_sel, N)

    # Build sparse spectrum and approximate window in time
    mask = np.zeros(N, dtype=bool)
    mask[np.array(k_sel, dtype=int) % N] = True
    W_sel = np.where(mask, W, 0.0)
    w_approx = np.fft.ifft(W_sel).real  # real because of Hermitian symmetry

    # -- Plot window vs approximation
    plt.figure()
    plt.title(f"Window in time: original vs. {len(k_sel)}-freq approximation")
    plt.plot(t, w, label="w[n] (original)")
    plt.plot(t, w_approx, linestyle="--", label=f"w~[n] (K={len(k_sel)})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3) Make a simple seismic-like signal
    x = np.zeros(N)
    centers = [0.1, 0.25, 0.3, 0.32, 0.36, 0.41, 0.55, 1.10]  # seconds
    amps = [1.0, -0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    for c, a in zip(centers, amps):
        x += a * ricker_wavelet(t - c, f0)
    rng = np.random.default_rng(1)
    x += noise_sigma * rng.standard_normal(N)

    # Exact time-domain mute (ground truth)
    y_time = w * x

    plt.figure()
    plt.title("Signal and time-domain mute (ground truth)")
    plt.plot(t, x, label="x[n] (signal)")
    plt.plot(t, y_time, linestyle="--", label="w[n]·x[n]")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 4) Frequency-domain convolution using only K bins
    #    Multiplication in time <-> convolution in frequency:
    #    FFT(y) = (1/N) * (FFT(w) ⊛ FFT(x))
    X = np.fft.fft(x)
    Y_freqconv = circ_conv_sparse(W_sel, X) / N
    y_freq = np.fft.ifft(Y_freqconv).real

    # Compare: exact (time multiply) vs. freq-domain approximation
    plt.figure()
    plt.title(f"Time mute: exact vs. freq-domain approx (K={len(k_sel)})")
    plt.plot(t, y_time, label="Exact (time multiply)")
    plt.plot(t, y_freq, linestyle="--", label="Approx (freq conv)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)

    # Quick error readouts
    err_w = np.sqrt(np.mean((w - w_approx) ** 2))
    err_y = np.sqrt(np.mean((y_time - y_freq) ** 2))
    print(f"Window RMS error (time): {err_w:.3e}")
    print(f"Mute   RMS error (time): {err_y:.3e}")

    plt.show()


if __name__ == "__main__":
    main()
