# -*- coding: utf-8 -*-
# Narrow bandpass filtering for a time series using a Gaussian function.
# Original Author: Hongrui Qiu, CUG
# Modified by: Yusong Yuan, CUG, 2026-03-26

import numpy as np
from scipy.fftpack import fft, ifft, next_fast_len

def opt_fft(data):
    '''
    Compute the Fast Fourier Transform (FFT) of the input data with optimal zero padding.
    
    The length of the FFT output will be the next power-of-two size, or a size that 
    optimizes the computation for the input length.
    
    Note: The length of the output is different from the input.
    '''
    nmin = data.shape[-1]*2
    nopt = next_fast_len(nmin)
    spec = fft(data, n=nopt)
    return spec

def narrow_bandpass(wf, per, dt, alpha=20):
    '''
    Perform narrow bandpass filtering using a Gaussian function for a specific period.
    
    Args:
    - wf (1D array): Input time series (single cycle of the waveform).
    - per (float): Center period of the bandpass filter (in the same units as the data).
    - dt (float): Time step between data points.
    - alpha (float, optional): Controls the width of the Gaussian bandpass filter. Default is 20.

    Returns:
    - (1D array): Filtered waveform, corresponding to the input signal but passed through the narrow bandpass filter.
    '''
    # Perform FFT on the waveform
    sf = opt_fft(wf)
    ns = len(sf)
    dom = 2 * np.pi / (ns * dt)  # frequency resolution
    om_k = 2 * np.pi / per  # angular frequency for the central period

    # Create Gaussian bandpass filter in frequency domain
    b = np.exp(-((dom * np.arange(ns) - om_k) / om_k) ** 2 * alpha)

    # Apply the filter in frequency domain
    filt_sf = b * sf

    # Set half of the spectrum to zero for Hilbert transformation
    for m in range(ns // 2 + 1, ns, 1):
        filt_sf[m] = 0.0
    
    filt_sf[0] /= 2.0
    filt_sf[ns // 2] = np.real(filt_sf[ns // 2])

    # Perform inverse FFT to get the filtered time series
    tmp = ifft(filt_sf)
    
    # Return the real part of the filtered waveform for the single period
    filt_wf = np.real(tmp[:len(wf)])

    return filt_wf