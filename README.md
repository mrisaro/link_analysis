# Data analysis of glitches for OFM-NPL

The idea of the repository is to keep track of the data analysi I've been doing on the data from the campaign. 
Mainly focused on the beat information of the optical fiber links. Make a characterization of the fiber link noise and try to define a criteria for the detection of glitches and determinations of the valid points for the optical clock comparison.

# Signal Analysis Toolkit

In the project there is a main_test.py file which is a simple implementation of the function. The only necessary thing is to import the function from the auxiliary file and then apply it to the data. The parameters already set to the function are the ones I think are the best for the data analysis.

## Overview

The analyze_signal function is designed to process time-series frequency data, identifying stable (locked) regions and detecting anomalies (glitches) within those regions. This tool is particularly useful for analyzing beatnote frequency data in precision measurement systems.

### Features

 *Robust Moving Standard Deviation: Utilizes a moving window to compute a robust estimate of the standard deviation, mitigating the influence of outliers.

 *Locked Region Detection: Identifies periods where the system is considered 'locked' based on the frequency stability.

 *Glitch Detection: Detects sudden deviations (glitches) in the frequency data within locked regions, accounting for adjacent data points affected by such anomalies.

 *Uptime Calculation: Computes the proportion of valid (non-glitch) data points within locked regions, providing a measure of system stability.

### Methodology

1. Compute Moving Robust Standard Deviation:

    - A moving window is applied to the frequency data to calculate the robust standard deviation at each point.

    - This approach reduces the impact of transient outliers on the standard deviation estimate.

2. Identify Locked Regions:

    - The deviation of each data point from the expected frequency (f0) is compared against a threshold (e.g., threshold * m_std).

    - A moving window assesses the fraction of outliers; if this fraction exceeds outlier_fraction, the region is marked as 'unlocked'.

3. Detect Glitches within Locked Regions:

    - Within the identified locked regions, data points deviating more than thr * m_std from f0 are flagged as glitches.

    - To account for potential measurement artifacts, adjacent points (pre before and pos after) are also marked as glitches.

    - A moving window evaluates the glitch rate; regions exceeding glitch_rate_threshold are considered invalid.

4. Compute Uptime:

    - The ratio of valid data points (non-glitch) within locked regions to the total number of data points provides the system's uptime.

### function signature

<pre> ```python 
# Your Python code here 
def analyze_signal(
    data,
    window_size,
    outlier_window,
    f0=21.4e6,
    threshold=3,
    outlier_fraction=0.1,
    thr=3,
    pre=1,
    pos=1,
    glitch_window=500,
    glitch_rate_threshold=0.01
):
    """
    Analyze the signal to determine locked regions, valid data points, outliers, uptime, and moving robust std.

    Parameters:
        data (array-like): Input frequency data.
        window_size (int): Window size for computing moving robust std.
        outlier_window (int): Window size for assessing outlier fraction.
        f0 (float): Expected central frequency.
        threshold (float): Threshold multiplier for outlier detection in locked region identification.
        outlier_fraction (float): Maximum allowed fraction of outliers in a window to consider region as locked.
        thr (float): Threshold multiplier for glitch detection within locked regions.
        pre (int): Number of preceding points to include as glitches.
        pos (int): Number of succeeding points to include as glitches.
        glitch_window (int): Window size for computing glitch rate.
        glitch_rate_threshold (float): Maximum allowed glitch rate in a window to consider data as valid.

    Returns:
        locked_mask (np.ndarray): Boolean array indicating locked regions.
        valid_mask (np.ndarray): Boolean array indicating valid data points.
        outlier_mask (np.ndarray): Boolean array indicating detected glitches.
        uptime (float): Proportion of valid data points within locked regions.
        m_std (np.ndarray): Moving robust standard deviation of the data.
    """
``` </pre>