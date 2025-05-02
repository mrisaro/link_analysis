import numpy as np
import matplotlib.pyplot as plt
import allantools

def moving_robust_std(data, window_size, max_std=None):
    """
    Compute the robust standard deviation (based on MAD) in a moving window over data.

    Parameters:
        data (array-like): The input signal.
        window_size (int): Size of the moving window (must be odd for symmetry).

    Returns:
        np.ndarray: Array of robust std estimates, same size as data.
    """
    data = np.asarray(data)
    half_win = window_size // 2

    # Use 'reflect' to prevent artificially flat edges
    padded = np.pad(data, (half_win, half_win), mode='reflect')
    result = np.zeros_like(data, dtype=float)

    for i in range(len(data)):
        window = padded[i:i + window_size]
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        result[i] = 1.4826 * mad
        
    if max_std is not None:
        result = np.clip(result, None, max_std) 
        
    return result

def detect_unlocked_regions(data, m_std, outlier_window, f0=21.4e6, threshold=3, outlier_fraction=0.1):
    """
    Detect unlocked regions based on local outlier rate.

    Parameters:
        data (array-like): Input signal.
        m_std (array-like): Moving robust standard deviation.
        outlier_window (int): Window size for moving outlier rate.
        f0 (float): Expected central frequency.
        threshold (float): Threshold in units of robust std to count as outlier.
        outlier_fraction (float): Fraction of outliers above which region is considered unlocked.

    Returns:
        locked_mask (np.ndarray): Boolean array, True = locked, False = unlocked.
    """
    data = np.asarray(data)
    residuals = np.abs(data - f0)
    outliers = residuals > threshold * m_std

    # Use reflect padding
    half_win = outlier_window // 2
    padded_outliers = np.pad(outliers.astype(int), (half_win, half_win), mode='reflect')
    outlier_rate = np.zeros_like(data, dtype=float)

    for i in range(len(data)):
        window = padded_outliers[i:i + outlier_window]
        outlier_rate[i] = np.mean(window)

    locked_mask = outlier_rate <= outlier_fraction
    return locked_mask

def detect_glitches(
    data,
    m_std,
    locked_mask,
    f0=21.4e6,
    thr=3,
    pre=1,
    pos=1,
    glitch_window=500,
    glitch_rate_threshold=0.05
):
    """
    Detect glitches (outliers) in locked regions of the signal, with an additional
    criterion based on local glitch rate within a moving window.

    Parameters:
        data (array-like): Input signal.
        m_std (array-like): Moving robust standard deviation.
        locked_mask (array-like): Boolean mask indicating locked regions.
        f0 (float): Expected central frequency.
        thr (float): Threshold multiplier for standard deviation.
        pre (int): Number of preceding points to include as outliers.
        pos (int): Number of succeeding points to include as outliers.
        glitch_window (int): Size of the moving window to evaluate glitch rate.
        glitch_rate_threshold (float): Maximum allowed glitch rate within the window.

    Returns:
        valid_mask (np.ndarray): Boolean array, True = valid data, False = outlier.
        outlier_mask (np.ndarray): Boolean array, True = outlier, False = valid data.
        uptime (float): Ratio of valid data points to total data points.
    """
    data = np.asarray(data)
    m_std = np.asarray(m_std)
    locked_mask = np.asarray(locked_mask)

    # Compute absolute deviation from expected value
    deviation = np.abs(data - f0)

    # Identify initial outliers in locked regions
    initial_outliers = (deviation > thr * m_std) & locked_mask

    # Initialize outlier mask
    outlier_mask = initial_outliers.copy()

    # Include neighboring points based on pre and pos
    for i in range(1, pre + 1):
        outlier_mask |= np.roll(initial_outliers, i)
    for i in range(1, pos + 1):
        outlier_mask |= np.roll(initial_outliers, -i)

    # Ensure outlier_mask is only True within locked regions
    outlier_mask &= locked_mask

    # Additional criterion: moving window glitch rate using convolution
    if glitch_window > 1 and glitch_rate_threshold > 0:
        # Create a kernel for convolution
        kernel = np.ones(glitch_window) / glitch_window

        # Compute the glitch rate using convolution
        glitch_rate = np.convolve(outlier_mask.astype(float), kernel, mode='same')

        # Identify regions where glitch rate exceeds the threshold
        high_glitch_regions = glitch_rate > glitch_rate_threshold

        # Update outlier mask to include high glitch regions
        outlier_mask |= high_glitch_regions

    # Valid data mask is the complement of outlier mask within locked regions
    valid_mask = locked_mask & ~outlier_mask

    # Compute uptime
    uptime = np.sum(valid_mask) / len(data)

    return valid_mask, outlier_mask, uptime

def analyze_signal(data, window_size, outlier_window, f0=21.4e6, threshold=3, outlier_fraction=0.1, thr=3, pre=1, pos=1):
    """
    Analyze the signal to determine locked regions, valid data points, outliers, uptime, and moving robust std.

    Parameters:
        data (array-like): Input signal.
        window_size (int): Size of the moving window for robust std.
        outlier_window (int): Window size for moving outlier rate.
        f0 (float): Expected central frequency.
        threshold (float): Threshold in units of robust std to count as outlier for unlocking.
        outlier_fraction (float): Fraction of outliers above which region is considered unlocked.
        thr (float): Threshold multiplier for standard deviation for glitch detection.
        pre (int): Number of preceding points to include as outliers.
        pos (int): Number of succeeding points to include as outliers.

    Returns:
        locked_mask (np.ndarray): Boolean array, True = locked, False = unlocked.
        valid_mask (np.ndarray): Boolean array, True = valid data, False = outlier.
        outlier_mask (np.ndarray): Boolean array, True = outlier, False = valid data.
        uptime (float): Ratio of valid data points to total data points.
        m_std (np.ndarray): Moving robust standard deviation.
    """
    m_std = moving_robust_std(data, window_size, max_std=1.2)
    locked_mask = detect_unlocked_regions(data, m_std, outlier_window, f0, threshold, outlier_fraction)
    valid_mask, outlier_mask, uptime = detect_glitches(data, m_std, locked_mask, f0, thr, pre, pos)
    return locked_mask, valid_mask, outlier_mask, uptime, m_std

def plot_data_with_std_and_unlocks(data, window_size, locked_mask, f_name, f0=21.4e6, thr=3):
    """
    Plots the signal with moving robust std and highlights unlocked regions below.

    Parameters:
        data (array-like): Input signal.
        window_size (int): Size of the moving window for smoothing and robust std.
        locked_mask (array-like): Boolean mask, True = locked, False = unlocked.
        f0 (float): Expected central frequency.
    """
    data = np.asarray(data)
    locked_mask = np.asarray(locked_mask)
    m_std = moving_robust_std(data, window_size, max_std=1.2)
    time_arr = np.arange(len(data))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # Upper plot: signal + thresholds around f0
    ax1.plot(time_arr, data, '-o', linewidth=0.5, markersize=3, color='C0', label='Link')
    ax1.fill_between(time_arr, f0 - thr * m_std, f0 + thr * m_std,
                     color='C1', alpha=0.8, label=f'Threshold ({thr:.1f} Sigma)')
    ax1.set_ylim(f0 - 10, f0 + 10)
    ax1.set_ylabel(r'freq (Hz)', fontsize=12)
    ax1.legend(loc='best', fontsize=12)
    ax1.grid(linestyle='--')
    ax1.set_title('Signal and Unlock Regions')

    # Lower plot: unlocked regions
    unlock_mask = ~locked_mask  # invert to get unlocked regions
    ax2.plot(time_arr, unlock_mask.astype(int), drawstyle='steps-post', color='C3')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Locked', 'Unlocked'])
    ax2.set_xlabel('Time (sec)', fontsize=12)
    ax2.grid(linestyle='--')
    
    plt.tight_layout()
    plt.show()
    fig.savefig(f_name)
    #plt.close(fig)
    
def plot_signal_and_allan(data, m_std, valid_mask, outlier_mask, name, f0=21.4e6, thr=3, sample_rate=1.0):
    """
    Plot the signal with glitches and compute Allan deviation.

    Parameters:
        data (array-like): Input signal.
        m_std (array-like): Moving robust standard deviation.
        valid_mask (array-like): Boolean mask indicating valid data points.
        outlier_mask (array-like): Boolean mask indicating outlier data points.
        f0 (float): Expected central frequency.
        thr (float): Threshold multiplier for standard deviation.
        sample_rate (float): Sampling rate in Hz.
    """
    data = np.asarray(data)
    m_std = np.asarray(m_std)
    valid_mask = np.asarray(valid_mask)
    outlier_mask = np.asarray(outlier_mask)
    time_arr = np.arange(len(data)) / sample_rate

    # Compute uptime
    uptime = np.sum(valid_mask) / len(data) * 100

    # Plot signal with glitches
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False,
                                    gridspec_kw={'height_ratios': [3, 2]})
    ax1.plot(time_arr[valid_mask], data[valid_mask], '-o', linewidth=0.5, markersize=3, color='C0', label='Signal')
    ax1.fill_between(time_arr[valid_mask], f0 - thr * m_std[valid_mask], f0 + thr * m_std[valid_mask],
                     color='C1', alpha=0.5, label=f'Threshold (±{thr}σ)')
    ax1.plot(time_arr[outlier_mask], data[outlier_mask], 'ko', markersize=2, alpha=0.3, label='Glitches')
    ax1.set_ylim(f0-10,f0+10)
    ax1.set_xlim(-500, 87000)
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_title('Signal with Detected Glitches')
    ax1.legend()
    ax1.grid(True)

    # Annotate uptime
    ax1.text(0.01, 0.95, f'Uptime: {uptime:.2f}%', transform=ax1.transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

    # Compute Allan deviation
    (tau,adev,inu,inu) = allantools.mdev(data[valid_mask]/194e12,rate=1.0,data_type='freq')

    # Plot Allan deviation
    ax2.loglog(tau, adev, '-o')
    ax2.set_ylim(0.5e-18, 3e-15)
    ax2.set_xlabel('Tau (s)')
    ax2.set_ylabel('MDEV (@194 THz)')
    ax2.set_title('Allan Deviation of Valid Data')
    ax2.grid(True, which='both', ls='--')

    plt.tight_layout()
    plt.show()
    fig.savefig(name)
    #plt.close(fig)  