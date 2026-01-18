# -*- coding: utf-8 -*-
'''
Calculating the seismic phase travel time using obspy.taup based on a custom 1-D velocity model with tvel format.

First, you should add the line "from .taup_create import *" to 
"path/to/your/obspy/taup/__init__.py" to ensure that the custom model 
creation functionality is available.

Created by Yuan Yusong, CUG, 1/18/2026
'''
import obspy.taup as taup

# If this is the first time running the code, please uncomment the code line below. 
# Running this line will generate a model file in NPZ format in the current directory.

# obspy.taup.taup_create.build_taup_model("modified_model.tvel", "./") 

model = taup.TauPyModel(model="./modified_model.npz")

arrivals_P = model.get_travel_times(10, 10, phase_list=["P"])
arrivals_S = model.get_travel_times(10, 10, phase_list=["S"])

tP = arrivals_P[0].time
tS = arrivals_S[0].time

print(f"First travel time for phase 'P' : {tP}.")
print(f"First travel time for phase 'S' : {tS}.")
