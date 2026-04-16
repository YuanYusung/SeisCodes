#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:35:07 2026

@author: yusong
"""
import numpy as np
from scipy.fftpack import ifft, fftfreq
import pylab as pl

def synthetic_surface_wave(if_addnoise=False):
    
    npts = 2401
    t0, te = -60, 60
    dist = 1.5  # km
    noise_level = 0.1
    taxis = np.linspace(t0, te, npts)
    delta = taxis[1] - taxis[0]
    freq = fftfreq(npts, d=delta)
    amp_spec = np.zeros(npts)
    phase = np.zeros(npts)
    fmin, fmax = 0.2, 3.0

    indf = (freq >= fmin) & (freq <= fmax)
    # Synthetic amplitude spectrum
    #amp_spec[indf] = np.sin(2 * np.pi * (freq[indf] - fmin) / (fmax - fmin))**2
    amp_spec[indf] = np.exp(-0.5 * ((freq[indf] - (fmin + fmax) / 2) / (fmax - fmin) * 6)**2)
    # Synthetic phase spectrum
    Vph = 0.25 + 1.25 * ((freq[indf] - fmax) / (fmin - fmax))**4
    tph = dist / Vph
    phase[indf] = 2 * np.pi * freq[indf] * (-tph - taxis[0])

    # Build synthetic signal
    spec = amp_spec * np.exp(1j * phase)
    if if_addnoise:
        noise = amp_spec[indf] * noise_level * np.exp(1j * 2 * np.pi * phase[indf] * np.random.randn(len(freq[indf])))
        spec[indf] += noise
    syndata = ifft(spec).real * 2

    # Plot raw synthetic waveform
    fig, ax = pl.subplots(1, 1, figsize=(12, 8))
    ax.tick_params(labelsize=18)
    ax.plot(taxis, syndata, 'k-',lw=2.5)
    ax.set_xlim(dist / max(Vph) - 1 / fmin, dist / min(Vph) + 5 / fmin)
    ax.set_ylim(-0.2,0.2)
    ax.set_xlabel('Time (s)',fontsize=18)
    fig.tight_layout()
    fig.savefig("Surface_wave.png",format="png",dpi=1000)
    
    
synthetic_surface_wave(if_addnoise=False)