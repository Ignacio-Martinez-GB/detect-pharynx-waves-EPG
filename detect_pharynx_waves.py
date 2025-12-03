
# Force Matplotlib to use TkAgg backend to prevent rendering issues in some environments.
import matplotlib
matplotlib.use('TkAgg')

import pyabf
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, medfilt
from matplotlib.widgets import Slider
import argparse
import os
import sys
from scipy.signal import lombscargle
from scipy.stats import sem
import scipy.stats as st
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def bandpass_filter(data, fs, lowcut, highcut, order=4):
    """
    Applies a zero-phase bandpass filter to the input signal.

    Args:
        data (np.ndarray): The input signal array.
        fs (float): The sampling frequency of the signal in Hz.
        lowcut (float): The low cutoff frequency of the filter in Hz.
        highcut (float): The high cutoff frequency of the filter in Hz.
        order (int): The order of the Butterworth filter.

    Returns:
        np.ndarray: The filtered signal.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)


def detect_events_adaptive(signal, fs, window_sec=1.0, r_factor=5, e_factor=2, min_interval=0.05):
    """
    Detects E (trough) and R (peak) events using an adaptive thresholding
    approach based on a sliding window.

    The algorithm calculates the standard deviation of the signal within a sliding
    window and uses it to set local thresholds for peak and trough detection.

    Args:
        signal (np.ndarray): The input signal.
        fs (float): The sampling frequency in Hz.
        window_sec (float): The size of the sliding window in seconds.
        r_factor (float): The multiplier for the R peak detection threshold (std * r_factor).
        e_factor (float): The multiplier for the E trough detection threshold (std * e_factor).
        min_interval (float): The minimum interval between consecutive events in seconds.

    Returns:
        list: A list of tuples, where each tuple contains the indices (e_index, r_index)
              of a detected E-R event pair.
    """
    win_pts = int(window_sec * fs)
    step = win_pts // 2
    all_events = []
    min_dist = int(min_interval * fs)

    # Process the signal in overlapping sliding windows.
    for start in range(0, len(signal), step):
        end = min(start + win_pts, len(signal))
        segment = signal[start:end]
        noise_std = np.std(segment)
        r_prom = noise_std * r_factor
        e_prom = noise_std * e_factor

        r_peaks, _ = find_peaks(segment, prominence=r_prom, distance=min_dist)
        r_peaks = [i + start for i in r_peaks]

        for r in r_peaks:
            win_e = int(min_interval * fs * 2)
            s = max(start, r - win_e)
            seg2 = signal[s:r]
            if len(seg2) < 3:
                continue
            e_tr, _ = find_peaks(-seg2, prominence=e_prom, distance=1)
            if e_tr.size:
                all_events.append((s + e_tr[-1], r))

    # Sort events by R-peak time and filter to ensure minimum distance.
    all_events.sort(key=lambda x: x[1])
    filtered = []
    for e, r in all_events:
        if not filtered or (r - filtered[-1][1] > min_dist):
            filtered.append((e, r))

    # If no events are found, attempt a global detection approach.
    if not filtered:
        noise_std = np.std(signal)
        r_prom = noise_std * r_factor
        e_prom = noise_std * e_factor
        r_peaks, _ = find_peaks(signal, prominence=r_prom, distance=min_dist)

        for r in r_peaks:
            start0 = max(0, r - win_pts)
            seg0 = signal[start0:r]
            e_tr, _ = find_peaks(-seg0, prominence=e_prom, distance=1)
            if e_tr.size:
                filtered.append((start0 + e_tr[-1], r))

    return filtered


def compute_cadence(events, fs):
    """
    Computes the mean and standard deviation of event intervals (cadence).

    Args:
        events (list): A list of (e_index, r_index) tuples.
        fs (float): The sampling frequency in Hz.

    Returns:
        tuple: A tuple containing (mean_interval, std_interval) in seconds.
               Returns None if fewer than two events are provided.
    """
    intervals = np.diff([r for _, r in events]) / fs
    if len(intervals) == 0:
        return None
    return np.mean(intervals), np.std(intervals)


def plot_events(time, raw, filt, smoothed, events, title="Event Detection"):
    """
    Generates a plot showing the raw, filtered, and smoothed signals, along with
    detected E and R events.

    Args:
        time (np.ndarray): The time vector for the signals.
        raw (np.ndarray): The raw input signal.
        filt (np.ndarray): The filtered signal.
        smoothed (np.ndarray): The smoothed signal.
        events (list): A list of (e_index, r_index) tuples for detected events.
        title (str): The title for the plot.

    Returns:
        tuple: A tuple containing the Matplotlib figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, raw, label='Raw signal', alpha=0.3)
    ax.plot(time, filt, label='Filtered signal', alpha=0.5)
    ax.plot(time, smoothed, label='Smoothed signal', alpha=0.7)

    e_idx = [e for e, _ in events]
    r_idx = [r for _, r in events]

    ax.scatter(time[e_idx], filt[e_idx], c='red', label='E (trough)', zorder=5)
    ax.scatter(time[r_idx], filt[r_idx], c='green', label='R (peak)', zorder=5)

    # Add a raster plot of R-events at the top of the figure.
    y_min, y_max = ax.get_ylim()
    line_start = y_max * 0.9
    line_end = y_max
    for rt in time[r_idx]:
        ax.vlines(rt, line_start, line_end, color='black', linewidth=1)
    ax.set_ylim(y_min, y_max)

    ax.set_title(title)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    plt.tight_layout()

    return fig, ax


def export_r_times_to_csv(r_times, filename='r_times.csv'):
    """
    Exports the times of detected R events to a CSV file.

    Args:
        r_times (np.ndarray): An array of R event times in seconds.
        filename (str): The path for the output CSV file.

    Returns:
        bool: True if the export was successful, False otherwise.
    """
    try:
        import pandas as pd
        df = pd.DataFrame({'r_time_s': r_times})
        df.to_csv(filename, index=False)
        print(f"Exported {len(r_times)} R events to '{filename}'.")
        return True
    except ImportError:
        try:
            np.savetxt(filename, r_times, delimiter=',', header='r_time_s', comments='')
            print(f"Exported {len(r_times)} R events to '{filename}' using numpy.")
            return True
        except Exception as e:
            print(f"Error exporting to CSV: {e}")
            return False


class InteractiveDetector:
    """
    A class for interactively detecting signal events using adjustable thresholds.

    This class provides a Matplotlib-based GUI with sliders to control the
    detection thresholds for E (trough) and R (peak) events in real-time.
    """

    def __init__(self, time, filt):
        """
        Initializes the interactive detector with signal data.

        Args:
            time (np.ndarray): The time vector for the signal.
            filt (np.ndarray): The filtered signal data.
        """
        self.time = time
        self.filt = filt
        self.fs = 1.0 / (time[1] - time[0])
        self.paired_E = []
        self.paired_R = []
        self.setup_plot()

    def setup_plot(self):
        """Sets up the interactive plot, including sliders and event markers."""
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.25)

        self.ax.plot(self.time, self.filt, label='Filtered signal', color='orange', alpha=0.7)

        self.pos_init = np.max(self.filt) * 0.5
        self.neg_init = np.min(self.filt) * 0.5

        self.hline_pos = self.ax.axhline(self.pos_init, color='green', linestyle='--', label='R Threshold')
        self.hline_neg = self.ax.axhline(self.neg_init, color='red', linestyle='--', label='E Threshold')

        self.scat_R = self.ax.scatter([], [], color='green', s=30, label='R')
        self.scat_E = self.ax.scatter([], [], color='red', s=30, label='E')

        self.ax.legend()
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Amplitude')

        self.axpos = plt.axes([0.25, 0.15, 0.65, 0.03])
        self.axneg = plt.axes([0.25, 0.10, 0.65, 0.03])
        self.s_pos = Slider(self.axpos, 'R Threshold', np.min(self.filt), np.max(self.filt), valinit=self.pos_init)

        min_dist = int(0.05 * self.fs)
        self.all_R, _ = find_peaks(self.filt, distance=min_dist)
        self.s_neg = Slider(self.axneg, 'E Threshold', np.min(self.filt), np.max(self.filt), valinit=self.neg_init)
        self.all_E, _ = find_peaks(-self.filt, distance=min_dist)

        self.s_pos.on_changed(self.update)
        self.s_neg.on_changed(self.update)

        self.update(None)

    def update(self, val):
        """Callback function to update the plot when a slider value changes."""
        pos_th = self.s_pos.val
        neg_th = self.s_neg.val

        self.hline_pos.set_ydata([pos_th, pos_th])
        self.hline_neg.set_ydata([neg_th, neg_th])

        idx_R = self.all_R[self.filt[self.all_R] >= pos_th]
        idx_E = self.all_E[self.filt[self.all_E] <= neg_th]

        paired_E, paired_R = [], []
        prev_r = 0
        for r in idx_R:
            candidates = [e for e in idx_E if prev_r <= e < r]
            if candidates:
                best_e = min(candidates, key=lambda e_idx: self.filt[e_idx])
                paired_E.append(best_e)
                paired_R.append(r)
            prev_r = r

        self.scat_R.set_offsets(np.c_[self.time[paired_R], self.filt[paired_R]])
        self.scat_E.set_offsets(np.c_[self.time[paired_E], self.filt[paired_E]])

        self.paired_E, self.paired_R = paired_E, paired_R
        self.fig.canvas.draw_idle()

    def get_results(self):
        """
        Returns the final threshold values and the detected event pairs.

        Returns:
            tuple: A tuple containing (pos_threshold, neg_threshold, e_indices, r_indices).
        """
        return self.s_pos.val, self.s_neg.val, self.paired_E, self.paired_R


def static_threshold_plot(time, raw, filt, pos_th, neg_th):
    """
    Generates a plot of detection results using fixed, static thresholds.

    Args:
        time (np.ndarray): The time vector.
        raw (np.ndarray): The raw signal.
        filt (np.ndarray): The filtered signal.
        pos_th (float): The positive threshold for R peak detection.
        neg_th (float): The negative threshold for E trough detection.

    Returns:
        tuple: A tuple containing the indices of paired (E, R) events.
    """
    fs = 1.0 / (time[1] - time[0])

    above = np.where(filt >= pos_th)[0]
    R_events = []
    if above.size:
        splits = np.where(np.diff(above) > 1)[0]
        starts = np.concatenate(([0], splits + 1))
        ends = np.concatenate((splits, [above.size - 1]))
        for s, e in zip(starts, ends):
            seg = above[s:e + 1]
            R_events.append(seg[np.argmax(filt[seg])])

    below = np.where(filt <= neg_th)[0]
    E_events = []
    if below.size:
        splits = np.where(np.diff(below) > 1)[0]
        starts = np.concatenate(([0], splits + 1))
        ends = np.concatenate((splits, [below.size - 1]))
        for s, e in zip(starts, ends):
            seg = below[s:e + 1]
            E_events.append(seg[np.argmin(filt[seg])])

    paired_E, paired_R = [], []
    prev_r = 0
    for r in R_events:
        candidates = [e for e in E_events if prev_r <= e < r]
        if candidates:
            best_e = min(candidates, key=lambda e_idx: filt[e_idx])
            paired_E.append(best_e)
            paired_R.append(r)
        prev_r = r

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 4]}, sharex=True)
    plt.subplots_adjust(hspace=0.05)

    raster_ax = axes[0]
    if paired_R:
        raster_ax.eventplot([time[paired_R]], orientation='horizontal', colors='black', linewidths=1)
    raster_ax.set_yticks([])
    raster_ax.set_ylabel('Events')
    raster_ax.set_title('Raster Plot of R-Events')

    main_ax = axes[1]
    main_ax.plot(time, raw, label='Raw Signal', alpha=0.3)
    main_ax.plot(time, filt, label='Filtered Signal', color='orange')
    main_ax.axhline(pos_th, color='green', linestyle='--', label='R Threshold')
    main_ax.axhline(neg_th, color='red', linestyle='--', label='E Threshold')
    main_ax.scatter(time[paired_E], filt[paired_E], c='red', label='E', s=30)
    main_ax.scatter(time[paired_R], filt[paired_R], c='green', label='R', s=30)
    main_ax.legend()
    main_ax.set_xlabel('Time (s)')
    main_ax.set_ylabel('Amplitude')

    fig.suptitle('Static Threshold Detection', fontsize=14)
    plt.tight_layout()

    return paired_E, paired_R


def plot_raster(time, r_times):
    """
    Displays a raster plot of R-event times.

    Args:
        time (np.ndarray): The full time vector to set plot limits.
        r_times (np.ndarray): An array of R-event times in seconds.

    Returns:
        tuple: A tuple containing the Matplotlib figure and axes objects.
    """
    fig_r, ax_r = plt.subplots(figsize=(12, 2))
    ax_r.eventplot(r_times, orientation='horizontal', colors='black')
    ax_r.set_title('Raster Plot of R-Events')
    ax_r.set_xlabel('Time (s)')
    ax_r.set_yticks([])
    ax_r.set_xlim(time[0], time[-1])
    plt.tight_layout()
    return fig_r, ax_r


def plot_advanced_analysis(time, raw, filt, r_times, window_size=5):
    """
    Generates a three-panel figure for advanced analysis of pharyngeal rhythm,
    including the filtered signal, instantaneous frequency, and a raster plot.

    Args:
        time (np.ndarray): The time vector.
        raw (np.ndarray): The raw signal.
        filt (np.ndarray): The filtered signal.
        r_times (np.ndarray): An array of R-event times in seconds.
        window_size (int): The window size for calculating the moving average of frequency.

    Returns:
        tuple: A tuple containing the Matplotlib figure and a tuple of the three axes objects.
    """
    ieis = np.diff(r_times)
    freqs = 1.0 / ieis
    mean_freq = np.mean(freqs)
    std_freq = np.std(freqs)
    cv = std_freq / mean_freq

    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1], hspace=0.05)

    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time, raw, label='Raw Signal', alpha=0.3)
    ax1.plot(time, filt, label='Filtered Signal', color='orange')
    for rt in r_times:
        ax1.axvline(rt, color='green', alpha=0.3, linestyle='--')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Pharyngeal Rhythm Analysis', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.set_xticklabels([])

    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(r_times[1:], freqs, 'o-', label='Instantaneous Frequency', alpha=0.7)
    if len(freqs) > window_size:
        rolling_window = np.ones(window_size) / window_size
        rolling_mean = np.convolve(freqs, rolling_window, mode='valid')
        offset = window_size // 2
        rolling_x = r_times[1 + offset:1 + offset + len(rolling_mean)]
        ax2.plot(rolling_x, rolling_mean, 'r-', linewidth=2, label='Moving Average')

        if len(freqs) > window_size * 2:
            rolling_std = np.array([np.std(freqs[i:i + window_size]) for i in range(len(freqs) - window_size + 1)])
            ax2.fill_between(rolling_x, rolling_mean - rolling_std, rolling_mean + rolling_std, color='r', alpha=0.2, label='Standard Deviation')
            for i, (x, y) in enumerate(zip(r_times[1:], freqs)):
                if i >= offset and i < len(freqs) - offset:
                    idx = i - offset
                    if abs(y - rolling_mean[idx]) > 2 * rolling_std[idx]:
                        ax2.plot(x, y, 'r*', markersize=10)

    if len(freqs) > 3:
        ax_inset = ax2.inset_axes([0.65, 0.65, 0.3, 0.3])
        ax_inset.axvline(cv, color='r', linestyle='--', linewidth=2)
        ax_inset.text(0.5, 0.6, f'CV = {cv:.2f}', transform=ax_inset.transAxes, ha='center', va='center')
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_title('Coefficient of Variation')

    ax2.set_ylabel('Frequency (Hz)')
    ax2.legend(loc='upper left')
    ax2.set_xticklabels([])

    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.eventplot(r_times, orientation='horizontal', colors='black')
    ax3.set_yticks([])
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Events')

    ax1.set_xlim(time[0], time[-1])
    plt.tight_layout()
    return fig, (ax1, ax2, ax3)


def batch_process_folder(folder_path, lowcut=1.0, highcut=50.0, interactive=True, export_csv=True):
    """
    Processes all .abf files in a specified directory, performing event detection,
    metric calculation, and visualization.

    Args:
        folder_path (str): The path to the directory containing .abf files.
        lowcut (float): The low cutoff frequency for the bandpass filter.
        highcut (float): The high cutoff frequency for the bandpass filter.
        interactive (bool): If True, enables interactive threshold selection for each file.
        export_csv (bool): If True, exports the summary of results to a CSV file.

    Returns:
        list: A list of dictionaries, where each dictionary contains the analysis
              results for a single file.
    """
    import glob
    import pandas as pd
    import re
    results = []
    folder_name = os.path.basename(folder_path.rstrip(os.sep))
    m = re.search(r'[dD]ia\s*(\d+)', folder_name)
    day_val = int(m.group(1)) if m else ''
    parts = re.split(r'\s+[dD]ia\s*\d+', folder_name)
    strain_val = parts[0] if parts else folder_name

    abf_files = glob.glob(os.path.join(folder_path, '*.abf'))
    for filepath in abf_files:
        file_name = os.path.basename(filepath)
        abf = pyabf.ABF(filepath)
        abf.setSweep(0)
        raw = abf.sweepY
        time = abf.sweepX
        fs = abf.dataRate

        max_time = 180.0
        if time[-1] > max_time:
            idx_end = np.searchsorted(time, max_time)
            raw = raw[:idx_end]
            time = time[:idx_end]

        filt = bandpass_filter(raw, fs, lowcut, highcut)
        kernel = int(0.05 * fs)
        if kernel % 2 == 0: kernel += 1
        smoothed = medfilt(filt, kernel_size=kernel)

        if interactive:
            det = InteractiveDetector(time, filt)
            plt.show()
            pos_th, neg_th, paired_E, paired_R = det.get_results()
            paired_E, paired_R = static_threshold_plot(time, raw, filt, pos_th, neg_th)
            fig_static = plt.gcf()
            svg_static = os.path.join(folder_path, f"{os.path.splitext(file_name)[0]}_static_threshold.svg")
            fig_static.savefig(svg_static, format='svg')
            print(f"Saved static threshold figure for {file_name} to {svg_static}")
        else:
            events = detect_events_adaptive(filt, fs)
            paired_E = [e for e, _ in events]
            paired_R = [r for _, r in events]

        r_times = time[paired_R]
        if paired_E:
            pre = int(0.1 * fs); post = int(0.3 * fs)
            segments = []
            for e in paired_E:
                start, end = e - pre, e + post
                if start < 0 or end > len(filt):
                    seg = filt[max(0, start):min(len(filt), end)]
                    pad_pre = max(0, -start)
                    pad_post = max(0, end - len(filt))
                    seg = np.pad(seg, (pad_pre, pad_post), mode='constant', constant_values=np.nan)
                else:
                    seg = filt[start:end]
                segments.append(seg)
            
            segs_arr = np.vstack(segments)
            mean_wave = np.nanmean(segs_arr, axis=0)
            std_wave = np.nanstd(segs_arr, axis=0)
            tseg = (np.arange(len(mean_wave)) - pre) / fs
            plt.figure(figsize=(6, 4))
            plt.fill_between(tseg, mean_wave - std_wave, mean_wave + std_wave, alpha=0.3)
            plt.plot(tseg, mean_wave, color='blue')
            plt.title(f'Average Waveform ± STD: {file_name}')
            plt.xlabel('Time Relative to E (s)')
            plt.ylabel('Amplitude')
            plt.tight_layout()
            fig = plt.gcf()
            avg_svg = os.path.join(folder_path, f"{os.path.splitext(file_name)[0]}_avg_waveform.svg")
            fig.savefig(avg_svg, format='svg')
            print(f"Saved average waveform figure for {file_name} to {avg_svg}")
            plt.show()

        intervals = np.diff(r_times)
        mask = intervals > (1.0 / fs)
        ieis = intervals[mask]
        freqs = 1.0 / ieis if ieis.size > 0 else np.array([])
        
        from statsmodels.tsa.stattools import adfuller
        adf_stat, adf_p = (adfuller(freqs)[0:2]) if freqs.size > 3 else (np.nan, np.nan)
        sd1, sd2 = (np.sqrt(np.var(freqs[1:] - freqs[:-1]) / 2.0), np.sqrt(np.var(freqs[1:] + freqs[:-1]) / 2.0)) if freqs.size > 1 else (np.nan, np.nan)

        def _approx_entropy(U, m, r):
            N = len(U)
            def _phi(m_):
                x = np.array([U[i:i+m_] for i in range(N-m_+1)])
                C = [np.mean(np.max(np.abs(x - xi), axis=1) <= r) for xi in x]
                return np.log(np.mean(C)) if np.mean(C) > 0 else np.nan
            return abs(_phi(m) - _phi(m+1))

        def _sample_entropy(U, m, r):
            N = len(U)
            def _count(m_):
                x = np.array([U[i:i+m_] for i in range(N-m_+1)])
                return sum(np.sum(np.max(np.abs(x - xi), axis=1) <= r) - 1 for xi in x)
            B, A = _count(2), _count(3)
            return -np.log(A/B) if B > 0 and A > 0 else np.nan

        r_tol = 0.2 * np.std(freqs) if freqs.size else np.nan
        ap_en = _approx_entropy(freqs, 2, r_tol)
        samp_en = _sample_entropy(freqs, 2, r_tol)
        acf_lag1 = np.corrcoef(freqs[:-1], freqs[1:])[0, 1] if freqs.size > 1 else np.nan
        
        durations = np.array([(r - e) / fs for e, r in zip(paired_E, paired_R)]) if paired_E else np.array([])
        mean_duration = float(np.mean(durations)) if durations.size else np.nan
        
        total_events = len(paired_R)
        recording_duration_s = time[-1] - time[0]
        events_per_min = total_events * 60.0 / recording_duration_s if recording_duration_s > 0 else np.nan
        mean_freq = float(np.mean(freqs)) if freqs.size else np.nan
        median_freq = float(np.median(freqs)) if freqs.size else np.nan
        cv_freq = float(np.std(freqs) / mean_freq) if mean_freq > 0 else np.nan
        cv2 = np.mean(2 * np.abs(np.diff(ieis)) / (ieis[1:] + ieis[:-1])) if ieis.size > 1 else np.nan
        burst_index = np.sum(ieis < (np.mean(ieis) / 2)) / ieis.size if ieis.size else np.nan
        
        time_for_freq = r_times[1:][mask]
        slope = np.polyfit(time_for_freq, freqs, 1)[0] if ieis.size > 1 else np.nan
        
        if freqs.size > 0 and time_for_freq.size > 0:
            f_ls = np.linspace(0.1, fs / 2, 500)
            pgram = lombscargle(time_for_freq, freqs, 2 * np.pi * f_ls)
            dom_peak = f_ls[np.argmax(pgram)]
        else:
            dom_peak = np.nan
            
        f_fft = np.fft.rfftfreq(len(filt), 1 / fs)
        psd = np.abs(np.fft.rfft(filt)) ** 2
        
        result = {
            'file': file_name, 'strain': strain_val, 'day': day_val, 'total_events': total_events,
            'mean_freq': mean_freq, 'median_freq': median_freq, 'cv_freq': cv_freq, 'cv2': cv2,
            'burst_index': burst_index, 'mean_duration': mean_duration, 'events_per_min': events_per_min,
            'trend_slope': slope, 'dom_peak_ls': dom_peak, 'r_times': r_times, 'psd_freqs': f_fft,
            'psd_values': psd, 'freqs': freqs, 'time_for_freq': time_for_freq, 'adf_stat': adf_stat,
            'adf_p': adf_p, 'sd1': sd1, 'sd2': sd2, 'ap_en': ap_en, 'samp_en': samp_en, 'acf_lag1': acf_lag1,
        }
        results.append(result)

    df = pd.DataFrame([{k: v for k, v in res.items() if not isinstance(v, np.ndarray)} for res in results])
    df['mean_duration_ms'] = df['mean_duration'] * 1000

    if len(results) > 1 and any(len(res['r_times']) > 0 for res in results):
        data = [res['r_times'] for res in results if len(res['r_times']) > 0]
        offsets = list(range(len(data)))
        fig, ax = plt.subplots(figsize=(12, len(offsets) * 0.5))
        ax.eventplot(data, orientation='horizontal', lineoffsets=offsets, linelengths=0.8, colors='black')
        ax.set_yticks(offsets)
        ax.set_yticklabels([res['file'] for res in results if len(res['r_times']) > 0])
        ax.set_xlabel('Time (s)')
        ax.set_title('Raster Comparison of Files')
        plt.tight_layout()
        raster_svg = os.path.join(folder_path, 'raster_comparison.svg')
        fig.savefig(raster_svg, format='svg')
        print(f"Saved raster comparison figure to {raster_svg}")
        plt.show()

    metrics = [
        'total_events', 'mean_freq', 'cv_freq', 'cv2', 'mean_duration_ms', 'events_per_min',
        'adf_stat', 'adf_p', 'sd1', 'sd2', 'ap_en', 'samp_en', 'acf_lag1'
    ]
    fig, axs = plt.subplots(len(metrics), 1, figsize=(8, len(metrics) * 2.5))
    fig.subplots_adjust(left=0.25, hspace=0.6)
    for ax, metric in zip(axs, metrics):
        vals = df[metric].dropna().values
        mu, se = np.nanmean(vals), st.sem(vals, nan_policy='omit')
        ax.bar(0, mu, yerr=se, capsize=5, color='C0')
        ax.scatter(np.random.uniform(-0.2, 0.2, size=len(vals)), vals, alpha=0.3, color='C1', s=20)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([0])
        ax.set_xticklabels([metric], fontsize=10)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)
        ax.set_ylabel('Duration (ms)' if metric == 'mean_duration_ms' else 'Events/min' if metric == 'events_per_min' else metric)
    plt.tight_layout()
    global_svg = os.path.join(folder_path, 'global_summary.svg')
    fig.savefig(global_svg, format='svg')
    print(f"Saved global summary figure to {global_svg}")
    plt.show()

    valid = [res for res in results if res['time_for_freq'].size > 0]
    if valid:
        dt = 0.1
        max_common = min(res['time_for_freq'][-1] for res in valid)
        t_grid = np.arange(0, max_common, dt)
        freq_matrix = np.array([np.interp(t_grid, res['time_for_freq'], res['freqs'], left=np.nan, right=np.nan) for res in valid])
        mean_grid = np.nanmean(freq_matrix, axis=0)
        sem_grid = sem(freq_matrix, axis=0, nan_policy='omit')
        
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t_grid, mean_grid, 'r-', label='Global Mean')
        ax.fill_between(t_grid, mean_grid - sem_grid, mean_grid + sem_grid, color='r', alpha=0.2, label=f'±SE (n={len(valid)})')
        
        all_freqs = np.hstack([res['freqs'] for res in valid])
        cv_global = np.std(all_freqs) / np.mean(all_freqs) if all_freqs.size else np.nan
        ax_in = ax.inset_axes([0.65, 0.65, 0.3, 0.3])
        ax_in.hist(all_freqs, bins='auto', alpha=0.7)
        ax_in.axvline(np.mean(all_freqs), color='r', linestyle='--', linewidth=2)
        ax_in.text(0.5, 0.8, f'CV = {cv_global:.2f}', transform=ax_in.transAxes, ha='center')
        ax_in.set_xlabel('Frequency (Hz)')
        ax_in.set_ylabel('Count')
        ax_in.set_title('Frequency Distribution')
        
        ax.set_title('Grouped Instantaneous Frequency Analysis')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (Hz)')
        ax.legend()
        plt.tight_layout()
        grouped_svg = os.path.join(folder_path, 'grouped_frequency.svg')
        fig.savefig(grouped_svg, format='svg')
        print(f"Saved grouped frequency panel figure to {grouped_svg}")
        plt.show()

    if 't_grid' in locals() and 'mean_grid' in locals() and locals()['t_grid'].size > 1:
        t_interp, freq_interp = locals()['t_grid'], locals()['mean_grid']
        dt_interp = t_interp[1] - t_interp[0]
        
        adf_stats = [res['adf_stat'] for res in results if not np.isnan(res['adf_stat'])]
        adf_ps = [res['adf_p'] for res in results if not np.isnan(res['adf_p'])]
        adf_mu, adf_se, adf_p_mu = (np.mean(adf_stats), sem(adf_stats, nan_policy='omit'), np.mean(adf_ps)) if adf_stats else (np.nan, np.nan, np.nan)

        sd1_vals = [res['sd1'] for res in results if not np.isnan(res['sd1'])]
        sd2_vals = [res['sd2'] for res in results if not np.isnan(res['sd2'])]
        sd1_mu, sd2_mu = (np.mean(sd1_vals), np.mean(sd2_vals)) if sd1_vals else (np.nan, np.nan)

        freq_clean = freq_interp[np.isfinite(freq_interp)]
        r_tol = 0.2 * np.std(freq_clean) if freq_clean.size else np.nan
        ap_en = _approx_entropy(freq_clean, 2, r_tol)
        samp_en = _sample_entropy(freq_clean, 2, r_tol)

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        
        axs[0,0].plot(t_interp, freq_interp)
        axs[0,0].set_title('ADF Test on Global Mean Frequency')
        axs[0,0].set_xlabel('Time (s)')
        axs[0,0].set_ylabel('Frequency (Hz)')
        axs[0,0].text(0.05, 0.95, f'Stat={adf_mu:.2f}±{adf_se:.2f}, p={adf_p_mu:.3f}', transform=axs[0,0].transAxes, va='top', bbox=dict(facecolor='white', alpha=0.7))

        x_pc, y_pc = freq_clean[:-1], freq_clean[1:]
        axs[0,1].scatter(x_pc, y_pc, alpha=0.5)
        mn, mx = np.min(freq_clean), np.max(freq_clean)
        axs[0,1].plot([mn, mx], [mn, mx], 'r--')
        axs[0,1].set_title('Poincaré Plot')
        axs[0,1].set_xlabel('x[n]')
        axs[0,1].set_ylabel('x[n+1]')
        axs[0,1].text(0.05, 0.95, f'SD1={sd1_mu:.2f}\nSD2={sd2_mu:.2f}', transform=axs[0,1].transAxes, va='top', bbox=dict(facecolor='white', alpha=0.7))

        axs[1,0].bar(['ApEn', 'SampEn'], [ap_en, samp_en], alpha=0.7)
        axs[1,0].set_title('Entropy Metrics')
        axs[1,0].set_ylabel('Value')

        max_lag = int(1.0 / dt_interp)
        acf_vals = [np.corrcoef(freq_clean[:-lag], freq_clean[lag:])[0,1] if len(freq_clean) > lag else np.nan for lag in range(1, max_lag + 1)]
        lags = np.arange(1, max_lag + 1) * dt_interp
        ci = 1.96 / np.sqrt(max(1, len(freq_clean)))
        axs[1,1].stem(lags, acf_vals, linefmt='C4-', markerfmt='C4o', basefmt='k-')
        axs[1,1].hlines([ci, -ci], xmin=0, xmax=lags[-1], colors='r', linestyles='--')
        axs[1,1].set_title('Autocorrelation (1s lag)')
        axs[1,1].set_xlabel('Lag (s)')
        axs[1,1].set_ylabel('ACF')

        plt.tight_layout()
        ts_svg = os.path.join(folder_path, 'timeseries_analysis.svg')
        fig.savefig(ts_svg, format='svg')
        print(f"Saved time-series analysis figure to {ts_svg}")
        plt.show()

    metrics_umap = ['total_events', 'mean_freq', 'cv_freq', 'cv2', 'mean_duration_ms', 'events_per_min', 'adf_stat', 'adf_p', 'sd1', 'sd2', 'ap_en', 'samp_en', 'acf_lag1']
    X_raw = df[metrics_umap].values
    X_imp = SimpleImputer(strategy='mean').fit_transform(X_raw)
    X_scaled = StandardScaler().fit_transform(X_imp)
    embedding = umap.UMAP(n_neighbors=10, min_dist=0.1, n_components=2, random_state=42).fit_transform(X_scaled)
    
    fig_umap, ax_umap = plt.subplots(figsize=(6,6))
    ax_umap.scatter(embedding[:,0], embedding[:,1], alpha=0.8)
    ax_umap.set_title('UMAP of Summary Metrics')
    ax_umap.set_xlabel('UMAP 1')
    ax_umap.set_ylabel('UMAP 2')
    plt.tight_layout()
    umap_svg = os.path.join(folder_path, 'umap_embedding.svg')
    fig_umap.savefig(umap_svg, format='svg')
    print(f"Saved UMAP embedding figure to {umap_svg}")
    plt.show()

    if export_csv:
        folder_name = os.path.basename(folder_path.rstrip(os.sep))
        csv_path = os.path.join(folder_path, f"{folder_name}_summary.csv")
        df.to_csv(csv_path, index=False)
        
        feature_cols = [
            'file', 'strain', 'day', 'total_events', 'mean_freq', 'median_freq', 'cv_freq', 'cv2', 
            'burst_index', 'mean_duration_ms', 'events_per_min', 'trend_slope', 'dom_peak_ls',
            'adf_stat', 'adf_p', 'sd1', 'sd2', 'ap_en', 'samp_en', 'acf_lag1'
        ]
        cols_to_export = [col for col in feature_cols if col in df.columns]
        features_full_csv = os.path.join(folder_path, f"{folder_name}_features_full.csv")
        df[cols_to_export].to_csv(features_full_csv, index=False)
        print(f"Exported full feature set CSV for UMAP to: {features_full_csv}")

    if input("Export summary data to CSV? (y/n): ").strip().lower() == 'y':
        summary_rows = []
        for metric in metrics:
            vals = df[metric].dropna().values.astype(float)
            summary_rows.append({'metric': metric, 'mean': np.mean(vals) if vals.size else np.nan, 
                                 'std': np.std(vals, ddof=1) if vals.size else np.nan, 
                                 'stderr': sem(vals, nan_policy='omit') if vals.size else np.nan})
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(folder_path, f"{os.path.basename(folder_path.rstrip(os.sep))}_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"Global summary saved to: {summary_path}")

    return results


def parse_arguments():
    """Parses command-line arguments for the script."""
    parser = argparse.ArgumentParser(description='Detect pharyngeal waves from ABF recordings.')
    parser.add_argument('--file', type=str, help='Path to the ABF file to analyze.',
                        default="/Users/ignaciomgb/Documents/OpenEphysAnalisis2025/AnlisisDatosFer/Datos Fer/Dia 1/ATM1 dia 1/25318037.abf")
    parser.add_argument('--lowcut', type=float, default=1.0, help='Low cutoff frequency (Hz) for the bandpass filter.')
    parser.add_argument('--highcut', type=float, default=50.0, help='High cutoff frequency (Hz) for the bandpass filter.')
    parser.add_argument('--interactive', action='store_true', help='Enable interactive mode for event validation.')
    parser.add_argument('--export', action='store_true', help='Automatically export results to CSV without prompting.')
    return parser.parse_args()


def main():
    """Main execution function for pharyngeal wave detection from a single file."""
    args = parse_arguments()

    if not os.path.isfile(args.file):
        print(f"Error: File not found: {args.file}")
        return 1

    try:
        print(f"Loading ABF file: {args.file}")
        abf = pyabf.ABF(args.file)
        abf.setSweep(0)
        raw = abf.sweepY
        time = abf.sweepX
        fs = abf.dataRate

        max_time = 180.0
        if time[-1] > max_time:
            idx_end = np.searchsorted(time, max_time)
            raw, time = raw[:idx_end], time[:idx_end]

        print(f"Data loaded: {len(raw)} samples, {time[-1]:.2f} seconds, {fs} Hz")

        print(f"Applying bandpass filter ({args.lowcut}-{args.highcut} Hz)...")
        filt = bandpass_filter(raw, fs, args.lowcut, args.highcut)

        kernel = int(0.05 * fs)
        if kernel % 2 == 0: kernel += 1
        smoothed = medfilt(filt, kernel_size=kernel)

        r_times = None
        if args.interactive or input("Generate interactive validation plot? (y/n): ").strip().lower() == 'y':
            print("Launching interactive detector...")
            detector = InteractiveDetector(time, filt)
            plt.show()

            pos_th, neg_th, paired_E, paired_R = detector.get_results()
            paired_E, paired_R = static_threshold_plot(time, raw, filt, pos_th, neg_th)
            plt.show()

            events = list(zip(paired_E, paired_R))
            refractory = int(0.05 * fs)
            events_sorted = sorted(events, key=lambda x: x[0])
            events_cons = []
            for e, r in events_sorted:
                if not events_cons or (r - events_cons[-1][1] > refractory):
                    events_cons.append((e, r))
                elif filt[e] < filt[events_cons[-1][0]]:
                    events_cons[-1] = (e, r)
            
            r_times = time[[r for _, r in events_cons]]

            if len(r_times) > 1:
                ieis = np.diff(r_times)
                freqs = 1.0 / ieis
                print(f"Manual detection: {len(events_cons)} events detected.")
                print(f"Manual mean frequency: {np.mean(freqs):.2f} Hz (CV: {np.std(freqs) / np.mean(freqs) * 100:.1f}%)")

                print("Generating advanced visualization...")
                plot_advanced_analysis(time, raw, filt, r_times)
                plt.show()

                plt.figure(figsize=(8, 4))
                plt.plot(r_times[1:], freqs, '-o')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.title('Instantaneous Frequency')
                plt.tight_layout()
                plt.show()

                plot_raster(time, r_times)
                plt.show()

        if r_times is not None and len(r_times) > 0:
            print("\n" + "=" * 50 + "\nDATA EXPORT\n" + "=" * 50)
            if args.export or input("Save R-event times to a CSV file? (y/n): ").strip().lower() == 'y':
                default_filename = 'r_times.csv'
                filename = input(f"Enter filename (default: '{default_filename}'): ").strip() or default_filename
                if not filename.endswith('.csv'): filename += '.csv'
                export_r_times_to_csv(r_times, filename)
                print(f"CSV saved as: {filename}")
        else:
            print("\nNo events were detected for CSV export.")

        return 0

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    # This block allows the script to be used for batch processing when executed directly.
    # Modify the 'folder_path' variable to point to your directory of .abf files.
    folder_path = '/Users/ignaciomgb/Documents/OpenEphysAnalisis2025/AnlisisDatosFer/Datos Fer/Dia 15/N2'
    
    # Execute batch processing.
    results = batch_process_folder(folder_path,
                                   lowcut=1.0,
                                   highcut=50.0,
                                   interactive=True,
                                   export_csv=True)
    
    # Print a summary of the results.
    for res in results:
        print(f"File: {res['file']}, Events: {res['total_events']}, Mean Frequency: {res.get('mean_freq', float('nan')):.2f} Hz")
    
    sys.exit(0)
