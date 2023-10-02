"""Resources for getting phase and amplitude information out of time-series data."""

import numpy as np
from typing import Tuple
import scipy
import pandas as pd


def get_fft_amplitude_and_phase_scipy(values: np.ndarray, times: np.ndarray, fs: float, n_samples: int = 1024, apply_filter: bool = False) -> Tuple[np.ndarray]:
    """Apply an fft to get phase, amplitude, period, and frequency information.

    Args:
        values: time series data
        times: timestamps for each data point
        fs: sampling frequency
        n_samples: number of sample points (default 1024). Should generally be a power of 2
        apply_filter: apply a 5th order butterworth filter to the data 

    Returns:
        Tuple of freq, amplitude, phase, period
        freq: frequency values for amplitude and phase vectors
        amplitude: amplitude of the response from the fft at different frequencies
        phase: phase of the response from the fft at different frequencies
        period: 1/frequency
    """
    assert len(values) == len(times)
    # Filter requires 3*order of filter number of samples
    if apply_filter:
        assert len(values) > 15
        raise NotImplementedError('Filter not yet included.')

    if pd.api.types.is_timedelta64_ns_dtype(times):
        dt_time = times
    else:
        dt_time = pd.to_timedelta(times)

    data = pd.Series(data=values, index=dt_time)
    data = data.fillna(method='bfill')
    data = data.resample('H').sum()
    freq = scipy.fft.fftfreq(n_samples, 1 / fs)
    fft_vals = scipy.fft.fft(data.values - np.mean(data.values), n=n_samples)
    mask = freq > 0
    freq = freq[mask]
    period = 1 / freq
    # calc the amplitude and phase
    amplitude = np.abs(fft_vals[mask]) / n_samples
    phase = np.angle(fft_vals[mask])
    return freq, amplitude, phase, period


def make_phase_df(df: pd.DataFrame, value_column: str, time_column: str = 'relative_exp_time', groups: str = 'Unique_animal', trim_start: int = 0, trim_end: int = 0, apply_filter: bool = False) -> pd.DataFrame:
    """Generate a phase response dataframe given an input time series dataframe.

    Args:
        df: Input dataframe
        value_column: column name for the values to run phase analysis on
        time_column: column name for the values
        groups: column name for the groupings of values
        trim_start: count of number of samples to remove from the beginning of each group
        trim_end: count of the number of samples to remove from the end of each group
        apply_filter: Apply the filter from `get_fft_amplitude_and_phase_scipy`

    Returns:
        Data frame containing the following columns:
            freq: frequency value for phase and amplitude
            amplitude: amplitude response
            phase: phase response
            period: 1/freq
            group: value from groups
    """
    assert value_column in list(df.columns)
    assert time_column in list(df.columns)
    assert len(df) > trim_start + trim_end

    # TODO: detect sample frequency from time column
    # Currently hard coded to 1hr (default from generate_behavior_tables)
    fs = 1

    # Trim time
    df_trimmed = df.sort_values(by=[groups, time_column]).reset_index(drop=True)
    df_trimmed = df_trimmed.groupby(groups).apply(lambda x: x.iloc[trim_start:-trim_end]).reset_index(drop=True)

    results = {}
    for group, sub_df in df_trimmed.groupby(groups):
        freq, amplitude, phase, period = get_fft_amplitude_and_phase_scipy(sub_df[value_column].values, sub_df[time_column].values, fs, apply_filter=apply_filter)

        results[group] = {'freq': freq, 'amplitude': amplitude, 'phase': phase, 'period': period, 'group': group}

    results = pd.DataFrame.from_dict(results, orient='index')
    results = results.explode(['freq', 'amplitude', 'phase', 'period'])
    results[['freq', 'amplitude', 'phase', 'period']] = results[['freq', 'amplitude', 'phase', 'period']].astype(float)
    return results


def to_fraction_str(x):
    """Prints a fraction as a string.

    Args:
        x: tuple of (num, den)
    """
    if x[0] == 0: 
        return '0'
    else: 
        return '{}/{}'.format(x[0], x[1])
