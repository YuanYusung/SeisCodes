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

    print(len(st))
    return st

if __name__ == "__main__":
    dat2pickle()