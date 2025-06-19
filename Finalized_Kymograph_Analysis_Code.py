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
from numpy.lib.stride_tricks import sliding_window_view


# Constants
DT = 0.0078  # Time resolution
WINDOW_LENGTH = 5
POLYORDER = 2
plt.rcParams["figure.dpi"] = 300
PTh=1.3 #phase thresold 
TF=0.3 #movement detection thresold factor

def fourier_denoise(signal, threshold_ratio=0.4):
    fft_signal = np.fft.fft(signal)
    amplitudes = np.abs(fft_signal)
    threshold = threshold_ratio * np.max(amplitudes)
    fft_signal[amplitudes < threshold] = 0
    return np.fft.ifft(fft_signal).real

def manual_slope(x, y):
    x_mean, y_mean = np.mean(x), np.mean(y)
    return np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)

def process_signal(signal, dt=DT):
    signal -= np.mean(signal)
    denoised = fourier_denoise(signal)
    smoothed = savgol_filter(denoised, window_length=WINDOW_LENGTH, polyorder=POLYORDER)
    analytic = hilbert(smoothed)
    phase = np.unwrap(np.angle(analytic))
    freq = np.diff(phase) / (2.0 * np.pi * dt)
    slope = manual_slope(np.arange(len(phase)) * dt, phase)
    smoothed_freq = savgol_filter(freq, window_length=WINDOW_LENGTH, polyorder=POLYORDER)
    return {
        'smoothed': smoothed,
        'analytic': analytic,
        'phase': phase,
        'freq': freq,
        'slope': slope,
        'smoothed_freq': smoothed_freq
    }

def plot_analysis(time, c_data, b_data, filename, title_prefix="", middle_col=None, segment_img=None):
    plt.figure(figsize=(14, 6))
    plt.suptitle(f'{title_prefix}\n{filename}')
    plt.subplot(1, 2, 1)
    plt.plot(time, c_data['smoothed'], color='purple', label='C (Left)')
    plt.title(f'Left Half{f" (Cols 0–{middle_col-1})" if middle_col else ""}')
    plt.xlabel('Time (s)')
    plt.ylabel('Averaged Intensity')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(time, b_data['smoothed'], color='orange', label='B (Right)')
    plt.title(f'Right Half{f" (Cols {middle_col}–...)" if middle_col else ""}')
    plt.xlabel('Time (s)')
    plt.ylabel('Averaged Intensity')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time, c_data['analytic'].real, color='purple', label='Cilia')
    plt.plot(time, b_data['analytic'].real, color='orange', label='Beam')
    plt.ylabel('Intensity')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(time, c_data['phase'], color='purple', label=f'C slope: {c_data["slope"]:.2f}')
    plt.plot(time, b_data['phase'], color='orange', label=f'B slope: {b_data["slope"]:.2f}')
    plt.ylabel('Phase')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(time[:-1], c_data['smoothed_freq'], color='purple', label=f'C avg freq: {np.mean(c_data["freq"]):.2f} Hz')
    plt.plot(time[:-1], b_data['smoothed_freq'], color='orange', label=f'B avg freq: {np.mean(b_data["freq"]):.2f} Hz')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(np.real(c_data['analytic']), np.imag(c_data['analytic']), color='purple')
    plt.title('C Hilbert Transform')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')

    plt.subplot(1, 2, 2)
    plt.plot(np.real(b_data['analytic']), np.imag(b_data['analytic']), color='orange')
    plt.title('B Hilbert Transform')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.tight_layout()
    plt.show()

    if segment_img is not None:
        plt.figure(figsize=(10, 5))
        plt.imshow(segment_img)
        plt.title(f'{title_prefix}\n{filename}')
        plt.show()

def analyze_and_plot_segment(segment_img, time, filename, title_prefix, show_plots=True):
    middle_col = segment_img.shape[1] // 2
    left_avg = np.mean(segment_img[:, :middle_col], axis=1)
    right_avg = np.mean(segment_img[:, middle_col:], axis=1)
    c_data = process_signal(left_avg)
    b_data = process_signal(right_avg)
    if show_plots:
        plot_analysis(time, c_data, b_data, filename, title_prefix, middle_col, segment_img)

#Movmement detection code provided by Maximilian Kotz
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


def analyze_before_after_beam_oscillation(
    img, 
    time, 
    filename, 
    show_plots=True,
    use_movement_detector=True,
    std_window_size=5,
    fallback_phase_threshold=1.5
):
    middle_col = img.shape[1] // 2
    left_avg = np.mean(img[:, :middle_col], axis=1)
    right_avg = np.mean(img[:, middle_col:], axis=1)
    
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
        full_b = process_signal(right_avg)
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
    
    # Process signals only if segments are long enough
    min_length = WINDOW_LENGTH * 2  # Ensure Savitzky-Golay can be applied
    if len(before_time) >= min_length:
        before_c = process_signal(left_avg[:split_index])
        before_b = process_signal(right_avg[:split_index])
    else:
        print(f"{filename}: 'Before' segment too short for analysis.")
        before_c, before_b = None, None
    
    if len(after_time) >= min_length:
        after_c = process_signal(left_avg[split_index:])
        after_b = process_signal(right_avg[split_index:])
    else:
        print(f"{filename}: 'After' segment too short for analysis.")
        after_c, after_b = None, None
    
    # --- Plot results ---
    if show_plots:
        if before_c and before_b:
            plot_analysis(
                before_time, before_c, before_b, filename,
                title_prefix="Before Beam Oscillations",
                segment_img=img[:split_index]
            )
        if after_c and after_b:
            plot_analysis(
                after_time, after_c, after_b, filename,
                title_prefix="After Beam Oscillations",
                segment_img=img[split_index:]
            )
def analyze_kymographs(filepath_pattern, show_plots=True, increment=100):
    filepaths = glob.glob(filepath_pattern)
    if not filepaths:
        print(f"No files found matching pattern: {filepath_pattern}")
        return

    for fpath in filepaths:
        img = skio.imread(fpath, plugin="tifffile")
        filename = Path(fpath).name
        total_rows = img.shape[0]
        num_segments = int(np.ceil(total_rows / increment))

        # Segment-wise analysis
        for segment in range(num_segments):
            start_row = segment * increment
            end_row = min((segment + 1) * increment, total_rows)
            segment_img = img[start_row:end_row]
            time = np.arange(start_row, end_row) * DT
            analyze_and_plot_segment(
                segment_img=segment_img,
                time=time,
                filename=filename,
                title_prefix=f'Segment {segment+1} (Rows {start_row}-{end_row-1})',
                show_plots=show_plots
            )

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
            fallback_phase_threshold=phase_threshold
        )

def plot_full_kymograph_analysis(filepath_pattern, show_plots=True):
    filepaths = glob.glob(filepath_pattern)
    for fpath in filepaths:
        img = skio.imread(fpath, plugin="tifffile")
        filename = Path(fpath).name
        time = np.arange(img.shape[0]) * DT
        analyze_and_plot_segment(img, time, filename, "Full Kymograph", show_plots=show_plots)


#change location based on where your kymographs are stored and how they're named ( For ex:"C:/Users/ridhi/Downloads/*.tif")
# Analyze each segment of each kymograph
#analyze_kymographs("C:/Users/ridhi/Downloads/*.tif", increment=100)

# Analyze the full kymograph
#plot_full_kymograph_analysis("C:/Users/ridhi/Downloads/*.tif", show_plots=True)

# Analyze full kymograph split into before/after beam oscillation
plot_full_kymograph_before_after("/home/max/Downloads/Kymographs/*.tif", show_plots=True)


