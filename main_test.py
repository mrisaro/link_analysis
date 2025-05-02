import numpy as np
import os
from auxiliary import analyze_signal

def load_frequency_data(file_path):
    """
    Load and clean frequency data from a text file.

    Parameters:
        file_path (str): Path to the data file.

    Returns:
        data (np.ndarray): 1D array containing the desired frequency data.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        # Load the data, skipping the first 14 header lines
        M = np.loadtxt(file_path, skiprows=14)
        # Extract columns from the 4th column onwards
        dat = M[:, 3:]
        # Extract the 3rd column from the sliced data
        data = dat[:, 2]
        return data
    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {e}")

file_path = 'data/250320_1_Frequ_L_G4L6_2.txt'
data = load_frequency_data(file_path)

# Analyze the signal
locked_mask, valid_mask, outlier_mask, uptime, m_std = analyze_signal(
    data,
    window_size=3601,
    outlier_window=1001,
    f0=21.4e6,
    threshold=5,
    outlier_fraction=0.1,
    thr=5,
    pre=1,
    pos=1
)

print(f"System uptime: {uptime * 100:.2f}%")