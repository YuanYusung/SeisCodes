# SeisCodes

SeisCodes is a collection of Python scripts and Jupyter notebooks for seismological data processing and waveform analysis.

This repository has grown out of my ongoing seismology research and will continue to evolve as my work progresses. The current version mainly focuses on dense-array ambient-noise processing, surface-wave dispersion inversion, shear-wave velocity imaging, and InSight Marsquake analysis.

<img width="576" height="384" alt="seismic_codes" src="https://github.com/user-attachments/assets/fa5b4f3f-aed2-459f-82bb-e7d5d562232d" />
  
> ⚠️ **Current status**  
> This repository is under active development. Most scripts are research-oriented and may require minor path or parameter modifications before use. The code is not yet organized as a formal Python package, but the main functions are documented in individual scripts.

## Repository structure
```text
SeisCodes/
├── marsquake/                     # Utilities for InSight Marsquake waveform analysis
│   ├── load_mars_event.py         # Load, trim, rotate, and cache Marsquake waveforms
│   ├── mars_event_info.py         # Extract event metadata from QuakeML/XML catalogs
│   └── plot_utilities.py          # Waveform, spectrogram, particle-motion, and FTAN-related plots
├── EikTomo4LinArr.py              # Eikonal tomography for 1-D linear seismic arrays
├── disp_inversion_evodcinv.py     # 1-D dispersion inversion using evodcinv and disba
├── concat_1D_Vs.py                # Assemble station-wise 1-D Vs models into 2-D sections
├── narrow_bandpass.py             # Gaussian narrow-bandpass filtering
├── sensitivity_disba.py           # Surface-wave sensitivity kernels using disba
├── decon.py                       # Iterative and water-level deconvolution utilities
├── syn_surface_wave.py            # Synthetic dispersive surface-wave example
├── DAS_EQ.ipynb                   # Notebook for DAS-related analysis
└── pygrt_ccf_synth.ipynb          # Notebook for synthetic cross-correlation tests
```

## Requirements

The scripts are mainly written in Python and were tested in a research environment using:

- Python >= 3.9
- NumPy
- SciPy
- Matplotlib
- ObsPy
- geopy
- disba
- evodcinv
- h5py / pickle for local data storage when needed

## Data notice

Large seismic datasets are not included in this repository. Some scripts use local paths for waveform data, dispersion files, velocity models, or event catalogs. Before running the examples, users should modify the input and output paths according to their own data structure.
