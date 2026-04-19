#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for processing ambient noise cross-correlation (ANC) data from 
seismic arrays, handled as Stream objects. 
Created at 2026/4/17 by Yusong Yuan
"""

import os
import numpy as np
from obspy import Trace, Stream
from geopy.distance import geodesic
import pickle

def dat2pickle():
    """
    The function reads the ANCs in .dat format (from NoiseCorr_SAC by Huajian Yao) and stores it as an ObsPy Stream object
    """
    dir = "../CFs/T-T/"
    fout = "../CFs/ANCs_TT.pickle"

    if os.path.exists(fout):
        with open(fout,'rb') as handle:
            st = pickle.load(handle)
        return st
    
    st = Stream()
    for file in os.listdir(dir):
        if not (file.startswith('TT') and file.endswith('dat')):
            continue

        evnm, stnm = file.split('_')[1].split('-')

        data = np.loadtxt(f"{dir}{file}")
        evlo, evla, evel=data[0]
        stlo, stla, stel=data[1]

        cc = np.concatenate((data[2:, 2][::-1], data[2:, 1][1:]))
        #taxis = np.concatenate((-1 * taxis_pos[::-1], taxis_pos[1:]))

        tr = Trace(cc)
        tr.stats.dist =  geodesic((evla, evlo), (stla, stlo)).kilometers
        tr.stats.delta = data[2:, 0][1] - data[2:, 0][0]
        tr.stats.b = -data[2:, 0][-1]
        tr.stats.kstnm = stnm
        tr.stats.kevnm = evnm

        st += tr

    with open(fout,'wb') as handle:
        pickle.dump(st,handle,protocol=pickle.HIGHEST_PROTOCOL)
    return st

def tapering(st,vmin,vmax):
    '''
    subroutine for tapering utilizing obspy
    
    tr - obspy trace object
    taxis - time axis of the input trace
    vmin,vmax - the moving out velocities of the tapering window
    '''
    npts = st[0].stats.npts; t0 = st[0].stats.b; delta = st[0].stats.delta
    taxis = np.linspace(t0,t0+npts*delta-delta,npts)
    st_copy = st.copy()
    for i in range(len(st)):
        tr = st[i]
        dist = tr.stats.dist
        tmin = dist/vmax; tmax = dist/vmin
        ind_taper = (taxis >= tmin) & (taxis <= tmax)
    
        tr.data += tr.data[::-1]
        tr_copy = tr.copy()
        tr_copy.data = tr.data[ind_taper]
        tr_copy.taper(max_percentage=0.05)
    
        data = np.zeros_like(tr.data)
        data[ind_taper] = tr_copy.data
        tr_copy.data = data

        st_copy[i] = tr_copy
    
    return st_copy

def plot_st(st):
    """
    Plots the wavefield of the input Stream object.
    """
    if len(st) == 0:
        print("Error: Waveform stream is empty.")
        return

    #st = tapering(st, vmin=1.0, vmax=6.0)

    wfs = []; dist_lst = []
    nt = len(st)
    for tr in st:
        dist = tr.stats.dist
        dist_lst.append(dist)
        wf = tr.data.copy()
        wf /= np.max(np.abs(wf))
        wfs.append(wf)
        
    dist_arr = np.array(dist_lst)

    tr_ref = st[0]
    t0 = tr_ref.stats.b
    npts = tr_ref.stats.npts; delta = tr_ref.stats.delta
    taxis = np.linspace(t0, t0 + npts * delta - delta, npts)
    
    ind = np.argsort(dist_arr)
    dist_sorted = dist_arr[ind]
    wfs_sorted = np.array(wfs)[ind, :]
    

    X, Y = np.meshgrid(taxis, dist_sorted)
    wfs_plot = wfs_sorted

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    example_idx = nt // 2 
    img = ax.pcolormesh(X, Y, wfs_plot, cmap='coolwarm', rasterized=True, shading='auto', vmin=-0.8, vmax=0.8) 
    
    dist_exp = dist_sorted[example_idx]
    ax.plot(taxis, wfs_plot[example_idx, :] / 10. + dist_exp, 'k-', lw=2.0, 
            label='Example Trace')
        
    ax.tick_params(labelsize=14) 
    ax.set_title('ANCs', fontsize=16)
    ax.set_xlim(-5, 50)
    ax.set_xlabel('Correlation time (s)', fontsize=16)
    ax.set_ylabel('Interstation distance (km)', fontsize=16)

    fig.tight_layout()
    os.makedirs('Figures', exist_ok=True)
    output_path = 'Figures/ANCs_ZZ.png'
    fig.savefig(output_path, format='png', dpi=300)
    plt.close(fig)
