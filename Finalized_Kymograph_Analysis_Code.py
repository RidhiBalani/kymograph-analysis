# -*- coding: utf-8 -*-
"""
Kymograph Analysis Code
by Ridhi Balani

"""
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
from scipy.signal import hilbert, savgol_filter
from pathlib import Path
import glob
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import uniform_filter1d


# Constants
DT = 0.0078  # Time resolution
WINDOW_LENGTH = 5
POLYORDER = 2
plt.rcParams["figure.dpi"] = 300
PTh=1.3 #phase thresold
TF=0.3 #movement detection thresold factor
HILBERT_REMOVE=20

def fourier_denoise(signal, dt, f_low=4.0, f_high=1000.0, f_rel_low=0.3,f_rel_high=5.0, power_cut=0.1, debug=False):
    """
    FFT‐based denoising of a 2D kymograph (time × space).

    Parameters
    ----------
    signal : 2D array, shape (n_t, n_s)
        Input kymograph: axis 0 is time, axis 1 is space.
    dt : float
        Time step between samples.
    f_low : float
        Lower cutoff frequency in Hz.
    f_high : float
        Upper cutoff frequency in Hz.
    debug : bool
        If True, plot original vs. denoised image and their PSDs.

    Returns
    -------
    denoised : 2D array, same shape as `signal`
        Real‐valued reconstruction after masking low‐amplitude
        and out‐of‐band frequency components.
    """
    # FFT along time axis
    fft_sig = np.fft.fft(signal, axis=0)         # shape (n_t, n_s)
    n_t, n_s = signal.shape
    
    # Frequencies and amplitudes
    freq = np.fft.fftfreq(n_t, d=dt)            # length n_t
    amps = np.abs(fft_sig)                      # shape (n_t, n_s)

    # PSD across spatial axis (after initial FFT)
    psd = np.sum(amps**2, axis=1)  # shape (n_t,)

    # Initial coarse band mask
    coarse_band = (np.abs(freq) >= f_low) & (np.abs(freq) <= f_high)

    # Find peak frequency in positive range
    pos = (freq >= 0) & coarse_band
    peak_idx = np.argmax(psd[pos])
    peak_freq = freq[pos][peak_idx]

    band_lower = peak_freq * f_rel_low
    band_upper = peak_freq * f_rel_high
    narrow_band = (np.abs(freq) >= band_lower) & (np.abs(freq) <= band_upper)

    # Final mask = must be in both coarse and narrow band
    final_mask = (coarse_band & narrow_band)[:, None]  # shape (n_t, 1)
    fft_masked = fft_sig * final_mask

    # Inverse FFT → denoised kymograph
    denoised = np.fft.ifft(fft_masked, axis=0).real

    # stripe masking
    stripe_powers=np.sum(amps**2 * final_mask,axis=0)
    stripe_mask = stripe_powers > np.max(stripe_powers) * power_cut 
    denoised[:, ~stripe_mask] = 0
    denoised_removed = denoised[:, stripe_mask]



    # Debug plots
    if debug:
        masked_psd = np.sum(np.abs(fft_masked)**2, axis=1)

        # Only plot non-negative freqs
        pos = freq >= 0

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # Original vs denoised kymograph
        axs[0, 0].imshow(signal, aspect='auto',interpolation=None)
        axs[0, 0].set_title('Original Kymograph')
        axs[0, 0].set_xlabel('Space (px)')
        axs[0, 0].set_ylabel('Time (frames)')

        axs[0, 1].imshow(denoised, aspect='auto',interpolation=None)
        axs[0, 1].set_title('Denoised Kymograph')
        axs[0, 1].set_xlabel('Space (px)')

        # PSD: original
        axs[1, 0].plot(freq[pos], psd[pos])
        axs[1, 0].set_title('Original PSD')
        axs[1, 0].set_xlabel('Frequency (Hz)')
        axs[1, 0].set_ylabel('Power')

        # PSD: masked
        axs[1, 1].plot(freq[pos], masked_psd[pos])
        axs[1, 1].set_title(f'Masked PSD\n(band {f_low}–{f_high} Hz)')
        axs[1, 1].set_xlabel('Frequency (Hz)')

        plt.tight_layout()
        plt.show()

    return denoised,denoised_removed, peak_freq

def manual_slope(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)
    return np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)

def align_stripes(kymo, win=300, max_shift=10, debug=False):
    """
    Temporally align spatial stripes of a kymograph with a fully sliding window.

    For each time t, center a window of length win (clipped at boundaries),
    correlate each stripe to the window-mean over stripes,
    subtract the spatial mean of lags (zero net shift),
    and store into lag_map[t, :].

    Finally shifts each stripe frame-by-frame according to lag_map
    and returns the spatial average.

    Parameters
    ----------
    kymo : 2D array, shape (n_t, n_s)
        Input kymograph (time × space).
    win : int
        Window length (frames) for local correlation.
    max_shift : int
        Maximum lag (± frames) to test.
    debug : bool
        If True, show diagnostic plots.

    Returns
    -------
    aligned_avg : 1D array, shape (n_t,)
        Time series after alignment and spatial averaging.
    """
    n_t, n_s = kymo.shape
    half = win // 2
    d_range = np.arange(-max_shift, max_shift+1)

    # Build continuous lag_map (n_t × n_s)
    lag_map = np.zeros((n_t, n_s), dtype=float)
    for t in tqdm(range(n_t), desc="Aligning stripes"):
        # window bounds
        st = t - half
        en = st + win
        if st < 0:
            st = 0
            en = min(win, n_t)
        elif en > n_t:
            en = n_t
            st = max(0, n_t - win)

        window = kymo[st:en]               # shape (w_i, n_s)
        ref = np.nanmean(window, axis=1)   # mean over stripes → (w_i,)

        lags = np.zeros(n_s, dtype=float)
        for s in range(n_s):
            cur = window[:, s]
            best_r, best_d = -np.inf, 0
            for d in d_range:
                if d < 0:
                    a, b = cur[-d:], ref[: len(cur[-d:])]
                elif d > 0:
                    a, b = cur[:-d], ref[d:]
                else:
                    a, b = cur, ref
                if a.size > 1:
                    r = np.corrcoef(a, b)[0, 1]
                else:
                    r = -np.inf
                if r > best_r:
                    best_r, best_d = r, d
            lags[s] = best_d

        # zero‐mean over space
        lags = lags - int(np.mean(lags))
        lag_map[t] = lags

    # Apply per-frame shifts
    aligned = np.full_like(kymo, np.nan)
    for s in range(n_s):
        for t in range(n_t):
            d = int(round(lag_map[t, s]))
            t2 = t + d
            if 0 <= t2 < n_t:
                aligned[t, s] = kymo[t2, s]

    # Spatial average
    aligned_avg = np.nanmean(aligned, axis=1)
    orig_avg    = np.nanmean(kymo, axis=1)

    # Debug plots
    if debug:
        fig, axs = plt.subplots(2, 2, figsize=(14, 8))

        # Lag map
        im0 = axs[0, 0].imshow(
            lag_map, aspect='auto', cmap='RdBu_r',interpolation=None
        )
        axs[0, 0].set_title('Lag Map (frames)')
        axs[0, 0].set_ylabel('Time')
        axs[0, 0].set_xlabel('Stripe index')
        fig.colorbar(im0, ax=axs[0, 0], label='lag')

        # Shifted (aligned) kymograph
        im1 = axs[0, 1].imshow(aligned, aspect='auto',interpolation=None)
        axs[0, 1].set_title('Shifted Kymograph')
        axs[0, 1].set_ylabel('Time')
        axs[0, 1].set_xlabel('Stripe index')
        fig.colorbar(im1, ax=axs[0, 1], label='intensity')

        # Spatial average
        axs[1, 0].plot(orig_avg, label='Original avg')
        axs[1, 0].plot(aligned_avg, label='Aligned avg', alpha=0.8)
        axs[1, 0].set_title('Spatial Average')
        axs[1, 0].set_xlabel('Time')
        axs[1, 0].legend()

        # Difference trace
        axs[1, 1].imshow(kymo, aspect='auto',interpolation=None)
        axs[1, 1].set_title('Original Kymograph')
        axs[1, 1].set_ylabel('Time')

        plt.tight_layout()
        plt.show()

    return aligned_avg

def compute_phase_from_protophi(protophi: np.ndarray, nharm: int = 10) -> np.ndarray:
    """
    Compute a proper phase from a given protophase using harmonic correction.

    Parameters:
    - protophi (np.ndarray): Real-valued 1D array representing protophase angles (in radians).
    - nharm (int): Number of harmonics to use in correction (default: 10).

    Returns:
    - np.ndarray: Unwrapped corrected phase of the same shape as protophi.
    """
    if not isinstance(protophi, np.ndarray) or protophi.ndim != 1:
        raise ValueError("protophi must be a 1D NumPy array")
    if not np.issubdtype(protophi.dtype, np.floating):
        raise ValueError("protophi must be a real-valued array")

    k = np.arange(1, nharm + 1)
    exp_kphi = np.exp(-1j * np.outer(k, protophi))  # shape: (nharm, len(protophi))
    Sn_pos = np.mean(exp_kphi, axis=1)  # shape: (nharm,)

    phi = protophi.astype(np.complex128)
    correction_terms = Sn_pos[:, None] * (np.exp(1j * np.outer(k, protophi)) - 1) / (1j * k[:, None])
    phi += 2 * np.sum(correction_terms, axis=0)

    return np.unwrap(np.real(phi))


def process_signal(signal, dt=DT):
    signal = signal - np.mean(signal,axis=0)  # Remove DC offset
    denoised,denoised_removed, peak_freq = fourier_denoise(signal,dt,f_rel_low=0.3,f_rel_high=1.7, debug=True)
    #denoised_removed = denoised_removed - uniform_filter1d(denoised_removed, size=int(5/(peak_freq*DT)), axis=0, mode='nearest')
    #aligned=align_stripes(denoised_removed,win=int(5/(peak_freq*DT)), max_shift=int(0.5/(peak_freq*DT))+1, debug=True)
    aligned = np.mean(denoised_removed,axis=1)
    analytic = hilbert(aligned)
    analytic = analytic[HILBERT_REMOVE:-HILBERT_REMOVE] # Remove edge artifacts from Hilbert transform
    phase = np.unwrap(np.angle(analytic))
    phase=compute_phase_from_protophi(phase)
    freq = (phase[-1]-phase[0]) / (2.0 * np.pi * dt * phase.shape[0])

    return {
        'denoised': denoised,
        'analytic': analytic,
        'phase': phase,
        'freq_phase': freq,
        'freq_PSD': peak_freq,
    }

def plot_analysis(time, c_data, b_data, filename, title_prefix="", segment_img=None):
    # plt.figure(figsize=(14, 6))
    # plt.suptitle(f'{title_prefix}\n{filename}')
    # plt.subplot(1, 2, 1)
    # plt.plot(time, c_data['denoised'], color='purple', label='C (Left)')
    # plt.title(f'Left Half{f" (Cols 0–{middle_col-1})" if middle_col else ""}')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Averaged Intensity')
    # plt.legend()

    # plt.subplot(1, 2, 2)
    # plt.plot(time, b_data['denoised'], color='orange', label='B (Right)')
    # plt.title(f'Right Half{f" (Cols {middle_col}–...)" if middle_col else ""}')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Averaged Intensity')
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    time = time[HILBERT_REMOVE:-HILBERT_REMOVE]  # Adjust time to match analytic data length
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time, c_data['analytic'].real/np.std(c_data['analytic'].real), color='purple', label='Cilia')
    if b_data is not None:
        plt.plot(time, b_data['analytic'].real/np.std(b_data['analytic'].real), color='orange', label='Beam')
    plt.ylabel('Intensity')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time, c_data['phase'], color='purple', label=f'C frequency (phase): {c_data["freq_phase"]:.2f} Hz')
    if b_data is not None:
        plt.plot(time, b_data['phase'], color='orange', label=f'B frequency (phase): {b_data["freq_phase"]:.2f} Hz')
    plt.ylabel('Phase')
    plt.legend()

    plt.subplot(3, 1, 3)
    # Phase deviation from linear fit
    c_dev = c_data['phase'] - 2 * np.pi * c_data['freq_phase'] * (time - time[0]) - c_data['phase'][0]
    if b_data is not None:
        b_dev = b_data['phase'] - 2 * np.pi * b_data['freq_phase'] * (time - time[0]) - b_data['phase'][0]
        diff = c_data['phase'] - b_data['phase']

    plt.plot(time, c_dev, color='purple', label='C phase change')
    if b_data is not None:
        plt.plot(time, b_dev, color='orange', label='B phase change')
        

        # Determine y-range
        y_min = min(np.min(c_dev), np.min(b_dev), np.min(diff))
        y_max = max(np.max(c_dev), np.max(b_dev), np.max(diff))

        # Compute vertical line positions in the phase range
        phase_min = np.floor(y_min / (2 * np.pi))
        phase_max = np.ceil(y_max / (2 * np.pi))
        if np.abs(phase_min - phase_max) < 10:
            plt.plot(time, diff, color='blue', label='C phase - B phase')
            for n in np.arange(phase_min, phase_max + 0.5):
                pos = n * np.pi
                if n % 2 == 0:
                    plt.axhline(pos, color='black', linestyle='--', linewidth=0.5)  # integer multiples
                else:
                    plt.axhline(pos, color='black', linestyle=':', linewidth=0.5)   # half-integer multiples

    plt.xlabel('Time (s)')
    plt.ylabel('Phase difference')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(np.real(c_data['analytic']), np.imag(c_data['analytic']), color='purple')
    plt.title('C Hilbert Transform')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    if b_data is not None:
        plt.subplot(1, 2, 2)
        plt.plot(np.real(b_data['analytic']), np.imag(b_data['analytic']), color='orange')
        plt.title('B Hilbert Transform')
        plt.xlabel('Real')
        plt.ylabel('Imaginary')
        plt.tight_layout()
        plt.show()

    if segment_img is not None:
        plt.figure(figsize=(10, 5))
        plt.imshow(segment_img,interpolation=None)
        plt.title(f'{title_prefix}\n{filename}')
        plt.show()

def detect_movement_start(kymo: np.ndarray,
                          window_size: int = 5,
                          threshold_factor: float = TF,
                          spatial_agg: str = 'mean') -> int:
    n_t, n_s = kymo.shape
    if window_size >= n_t:
        raise ValueError("window_size must be smaller than n_t")

    # 1) Build sliding windows over the time axis
    #    shape → (n_windows, window_size, n_s), where n_windows = n_t - window_size + 1
    windows = sliding_window_view(kymo, window_shape=(window_size, n_s))[:,0,:, :]  # view hack
    # Alternative correct call:
    # windows = sliding_window_view(kymo, window_shape=window_size, axis=0)

    # 2) Compute std over the time‐axis for each window and each spatial position
    std_per_space = np.std(windows, axis=1)         # shape: (n_windows, n_s)

    # 3) Aggregate over space to get a 1D “activity” trace
    if spatial_agg == 'mean':
        activity = std_per_space.mean(axis=1)
    elif spatial_agg == 'median':
        activity = np.median(std_per_space, axis=1)
    else:
        raise ValueError("spatial_agg must be ‘mean’ or ‘median’")

    baseline = np.median(activity)
    baseline_std = np.std(activity)
    # 5) Find first window where activity spikes above threshold
    thresh = baseline + threshold_factor * baseline_std

    exceed = np.where(activity > thresh)[0]
    if exceed.size == 0:
        return n_t  # no movement detected
    # the index in activity corresponds to window‐start time
    return int(exceed[0])

def choose_split_bounds(img, bounds=None):
    """
    Let user interactively choose two column bounds on `img`.
    - bounds: tuple(float or int or None, float or int or None)
        If not None, each element can be:
          - float in (0,1): fraction of width
          - int: absolute column
          - None: interactive
        Defaults to (0.5, 0.5) (center, center).
    Returns
    -------
    left, right : int
        The two column indices, clipped to [1, n_cols-1] and ordered.
    """
    n_cols = img.shape[1]
    # resolve initial
    left, right = bounds if bounds is not None else (0.5, 0.5)
    def resolve(x):
        if x is None: return None
        return int(n_cols * x) if isinstance(x, float) else x
    left = resolve(left)
    right = resolve(right)

    if left is not None and right is not None:
        # both fixed
        return *sorted((left, right)), 

    # interactive
    fig, ax = plt.subplots()
    ax.imshow(img, aspect='auto', interpolation=None)
    # init lines
    left = left or n_cols//2
    right = right or n_cols//2
    l_line = ax.axvline(left, color='r', linestyle='--')
    r_line = ax.axvline(right, color='b', linestyle=':')
    plt.title("Single-click to set LEFT (red); double-click to set RIGHT (blue);\nClose when done")

    def onclick(event):
        if event.inaxes != ax or event.xdata is None: return
        x = int(event.xdata)
        if event.dblclick:
            # right boundary
            nonlocal right
            right = x
            r_line.set_xdata([x, x])
        else:
            # left boundary
            nonlocal left
            left = x
            l_line.set_xdata([x, x])
        fig.canvas.draw_idle()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    # clip and order
    left = max(1, min(left, n_cols-1))
    right = max(1, min(right, n_cols-1))
    return (min(left, right), max(left, right))

def plot_psd_signal(before_c, after_c, after_b):
    plt.figure(figsize=(10, 5))
    signals = {
        'Cilium before': before_c['denoised'],
        'Cilium after': after_c['denoised'],
        'Beam': after_b['denoised']
    }

    for key, s in signals.items():
        n_t = s.shape[0]
        fft = np.fft.fft(s, axis=0)
        freq = np.fft.fftfreq(n_t, d=DT)
        psd = np.sum(np.abs(fft) ** 2,axis=1)  # shape (n_t,)

        pos = freq >= 0
        plt.plot(freq[pos], psd[pos]/np.max(psd), label=key)

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (a.u.)")
    plt.title("PSD of Hilbert-Filtered Signals")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analyze_before_after_beam_oscillation(
    img,
    time,
    filename,
    show_plots=True,
    use_movement_detector=True,
    std_window_size=5,
    fallback_phase_threshold=1.5,
    split=0.5  # Default split at middle column (0.5 means middle of the image width)
):
    left_col, right_col = choose_split_bounds(img, bounds=(None, None))
    left  = img[:, :left_col]
    right = img[:, right_col:]

    # Initialize split_index using detect_movement_start
    split_index = None
    if use_movement_detector:
        try:
            split_index = detect_movement_start(
                img,
                window_size=std_window_size,
                threshold_factor=TF  # Adjust based on your noise level
            )
            print(f"{filename}: Movement detected at index {split_index} (time = {time[split_index]:.2f}s)")
        except Exception as e:
            print(f"{filename}: Movement detection failed. Error: {e}")
            split_index = None

    # Fallback to phase threshold if movement detection fails
    if split_index is None or split_index <= 0 or split_index >= len(time) - 1:
        print(f"{filename}: Falling back to phase threshold.")
        full_b = process_signal(right)
        cross_indices = np.where(full_b['phase'] > fallback_phase_threshold)[0]
        split_index = cross_indices[0] if len(cross_indices) > 0 else len(time) // 2  # Default midpoint

    # Ensure split_index is valid
    split_index = max(1, min(split_index, len(time) - 1))

    # --- Plot activity trace from detect_movement_start for diagnostics ---
    if use_movement_detector and show_plots:
        plt.figure(figsize=(10, 4))
        windows = sliding_window_view(img, window_shape=(std_window_size, img.shape[1]))[:, 0, :, :]
        std_per_space = np.std(windows, axis=1)
        activity = std_per_space.mean(axis=1)
        baseline = np.median(activity)
        baseline_std = np.std(activity)
        thresh = baseline + TF * baseline_std  # Match threshold_factor=0.3

        plt.plot(time[:len(activity)], activity, label='Activity (Mean STD)')
        plt.axhline(thresh, color='r', linestyle='--', label='Threshold')
        plt.axvline(time[split_index], color='k', linestyle=':', label='Detected Onset')
        plt.xlabel('Time (s)')
        plt.ylabel('Activity')
        plt.title(f'{filename}: Movement Detection')
        plt.legend()
        plt.show()

    # --- Split data and analyze ---
    before_time = time[:split_index]
    after_time = time[split_index:]

    before_c = process_signal(left[:split_index,:])
    #before_b = process_signal(right[:split_index,:])

    after_c = process_signal(left[split_index:,:])
    after_b = process_signal(right[split_index:,:])
    

    # --- Plot results ---
    if show_plots:
        if before_c :
            plot_analysis(
                before_time, before_c, None, filename,
                title_prefix="Before Beam Oscillations",
                segment_img=img[:split_index]
            )
        if after_c and after_b:
            plot_analysis(
                after_time, after_c, after_b, filename,
                title_prefix="After Beam Oscillations",
                segment_img=img[split_index:]
            )
        if before_c and after_c and after_b:
            plot_psd_signal(before_c,after_c,after_b)

def plot_full_kymograph_before_after(filepath_pattern, phase_threshold=PTh, show_plots=True):
    filepaths = glob.glob(filepath_pattern)
    for fpath in filepaths:
        img = skio.imread(fpath, plugin="tifffile")
        filename = Path(fpath).name
        time = np.arange(img.shape[0]) * DT
        analyze_before_after_beam_oscillation(
            img,
            time,
            filename,
            show_plots=show_plots,
            use_movement_detector=True,
            std_window_size=WINDOW_LENGTH,
            fallback_phase_threshold=phase_threshold,
            split=None
        )
        

if __name__ == "__main__":
    plot_full_kymograph_before_after("/home/max/Downloads/Kymographs/*.tif", show_plots=True)
