"""
Created by Yusong Yuan, China University of Geosciences, 12/03/2025

This script performs 1D phase velocity dispersion curve inversion using the `evodcinv` library
(Luu, 2023; https://doi.org/10.5281/zenodo.5775193). It is designed to perform inversion to 
estimate the shear velocity structure of the Earth's subsurface based on phase velocity measurements
for my personal research.

Dependencies:
-------------
- evodcinv (Luu, 2023): Used for inversion of the phase velocity dispersion curves and Earth model construction.
- disba : Used for phase velocity dispersion curve prediction.
- numpy: For numerical operations and data handling.
- matplotlib: For plotting inversion results and model predictions.
- pickle: For saving and loading model layers and inversion results.
- os: For ensuring directory creation before saving files.

Usage:
------
1. This script processes phase velocity data for each station in the defined range.
2. It retrieves phase velocity data, performs inversion, and plots the resulting models.
3. Results (inversion models, dispersion curves, and misfit plots) are saved in the respective directories.

Functionality:
--------------
- `get_phvel(sta)`: Loads phase velocity data for a given station.
- `invert_1D(sta, if_plot=True)`: Performs 1D inversion for phase velocity curves and saves results.
- `save_empty_results(sta)`: Saves empty results for stations with missing or invalid data.
- `initialize_model()`: Initializes a default Earth model with predefined layers.
- `save_results(sta, layers, res, disp_curve)`: Saves inversion results, model layers, and dispersion curves.
- `plot_invert(layers, res, disp_curve, sta, zmax=2.)`: Plots the inversion results including the model and misfit curves.
- `plot_para_range(layers, ax=None)`: Plots the range of model parameters.
- `threshold(res, num=1000)`: Selects the top N models based on misfit values.
- `cal_mean_model(res, dz=0.01, zmax=None)`: Computes the mean velocity model from the inversion results.

Run this script to process seismic data from multiple stations and generate inversion models for 1D phase velocity profiles.

Note: Ensure that the required data files and libraries are correctly installed and available before execution.
"""


import numpy as np
from evodcinv import EarthModel, Layer, Curve, InversionResult, factory
from disba import resample, PhaseDispersion, depthplot
import copy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle
import os

def get_phvel(sta):
    """
    Retrieve phase velocity data based on station number.
    """
    try:
        fdisp = '../phase_vel/Seg1_Vph_nonsmooth.pkl' if sta < 345 else '../phase_vel/Seg2_Vph_nonsmooth.pkl'
        disp = np.load(fdisp, allow_pickle=True)

        pers = disp["period"]
        sta_ind = np.where(disp["sta_oth"] == sta)[0][0]
        vels = disp["mean_velocity"][sta_ind, :]
        uncers = disp["std_velocity"][sta_ind, :]

        return np.array(pers), np.array(vels), np.array(uncers)
    except Exception as e:
        print(f"Error in getting phase velocity for station {sta}: {e}")
        return None, None, None

def invert_1D(sta, if_plot=True):
    """
    Perform 1D phase velocity dispersion curve inversion.
    """
    print(f"Processing station {sta}!")

    pers, vels, uncers = get_phvel(sta)
    if pers is None or vels is None:
        print(f"Skipping station {sta} due to data retrieval error.")
        save_empty_results(sta)
        return None

    if np.all(np.isnan(vels)):
        print(f"Station {sta} has all NaN velocity data, skipping.")
        save_empty_results(sta)
        return None

    # Clean invalid data
    valid_mask = ~np.isnan(vels)
    pers, vels, uncers = pers[valid_mask], vels[valid_mask], uncers[valid_mask]
    uncers[uncers < 0.03] = 0.03

    # Initialize model
    model = initialize_model()

    # Model configuration
    model.configure(
        optimizer="cpso",
        misfit="rmse",
        increasing_velocity=True,
        extra_terms=[lambda x: factory.smooth(x, alpha=1.0e-3)],
        density=lambda vp: 2.0,
        dc=0.001,
        optimizer_args={"popsize": 200, "maxiter": 100, "workers": -1, "seed": 0}
    )

    # Define inversion curve
    curves = [Curve(pers, vels, 1, "rayleigh", "phase", uncertainties=uncers)]

    # Inversion execution
    res = model.invert(curves, maxrun=3)
    best_model = copy.deepcopy(res.models[np.argmin(res.misfits)])

    # Predict phase velocity from best model
    pd = PhaseDispersion(*best_model.T, dc=0.001)
    cpr_pred = pd(pers, mode=1, wave="rayleigh")

    # Store results
    disp_curve = {'period': pers, 'obs': vels, 'obs_std': uncers, 'fit': cpr_pred.velocity}
    save_results(sta, model.layers, res, disp_curve)

    if if_plot:
        plot_invert(model.layers, res, disp_curve, sta)

    return model.layers, res, disp_curve

def save_empty_results(sta):
    """
    Save empty results in case of missing data.
    """
    empty_model = EarthModel()
    empty_res = InversionResult([])
    empty_disp_curve = {'period': [], 'obs': [], 'obs_std': [], 'fit': []}
    
    # Ensure directories exist
    os.makedirs('./res', exist_ok=True)
    os.makedirs('./layers', exist_ok=True)
    os.makedirs('./disps', exist_ok=True)
    
    empty_res.write(f'./res/Vs_res_{sta}.json', file_format='json')
    with open(f'./layers/layers_{sta}.pkl', "wb") as f1:
        pickle.dump(empty_model.layers, f1)
    with open(f'./disps/disps_{sta}.pkl', "wb") as f2:
        pickle.dump(empty_disp_curve, f2)

def initialize_model():
    """
    Initialize the Earth model with predefined layers.
    """
    model = EarthModel()
    model.add(Layer([0.02, 0.02], [0.00, 0.00]))
    
    for i in range(3):
        model.add(Layer([0.05 + 0.01*i, 0.15 + 0.015*i], [0.1 , 0.4 + 0.5*i], [0.45 - i*0.03, 0.49]))
    
    for i in range(4):
        model.add(Layer([0.2 + 0.02*i, 0.3 + 0.02*i], [0.4 , 1.4 + 0.5*i], [0.4 - i*0.05, 0.45]))
    
    model.add(Layer([1.0, 1.0], [1.2 , 3.5], [0.3, 0.4]))
    
    return model

def save_results(sta, layers, res, disp_curve):
    """
    Save inversion results to files.
    """
    res.write(f'./res/Vs_res_{sta}.json', file_format='json')
    with open(f'./layers/layers_{sta}.pkl', "wb") as f1:
        pickle.dump(layers, f1)
    with open(f'./disps/disps_{sta}.pkl', "wb") as f2:
        pickle.dump(disp_curve, f2)

def plot_invert(layers, res, disp_curve, sta, zmax=2.):
    """
    Plot inversion results.
    """
    res = threshold(res, 10000)
    res_sel = threshold(res, 3000)
    mean_model = cal_mean_model(res_sel, dz=0.02)
    pd = PhaseDispersion(*mean_model.T, dc=0.001)
    cpr_pred = pd(disp_curve["period"], mode=1, wave="rayleigh")

    fig, ax = plt.subplots(1, 3, figsize=(14, 6))
    fig.subplots_adjust(bottom=0.25)

    plot_para_range(layers, ax=ax[0])
    depthplot(mean_model[:, 0], mean_model[:, 2], zmax=zmax, ax=ax[0], plot_args={"color": "blue", "linestyle": "--", "label": "Mean of Top 3000 models"})
    res.plot_model("vs", zmax=zmax, show="best", ax=ax[0], plot_args={"color": "red", "linestyle": "--", "label": "Best out of 60000 models"})
    ax[0].legend()

    res.plot_curve(disp_curve["period"], 1, "rayleigh", "phase", show="best", ax=ax[1], plot_args={"type": "line", "xaxis": "period", "color": "red", "linestyle": "-", "label": "Best Fitting"})
    
    sampled_period = np.concatenate(([disp_curve["period"][0]], disp_curve["period"][::5], [disp_curve["period"][-1]]))
    sampled_obs = np.concatenate(([disp_curve["obs"][0]], disp_curve["obs"][::5], [disp_curve["obs"][-1]]))
    sampled_obs_std = np.concatenate(([disp_curve["obs_std"][0]], disp_curve["obs_std"][::5], [disp_curve["obs_std"][-1]]))
    
    ax[1].plot(cpr_pred.period, cpr_pred.velocity, label="Mean_model", color='b')
    ax[1].errorbar(sampled_period, sampled_obs, yerr=sampled_obs_std, fmt='o', color='gray', capsize=5, label="Obs")
    ax[1].legend()

    res.plot_misfit(ax=ax[2])
    ax[2].set_ylim(0., 2.)
    
    norm = Normalize(vmin=res.misfits.min(), vmax=res.misfits.max())
    smap = ScalarMappable(norm=norm, cmap="viridis_r")
    axins = inset_axes(ax[1], width="150%", height="6%", loc="lower center", borderpad=-6.0)
    cb = plt.colorbar(smap, cax=axins, orientation="horizontal")
    cb.set_label("Misfit value")

    plt.savefig(f"./Figures/Vs_{sta}.png", dpi=300)

def plot_para_range(layers, ax=None):
    """
    Plot the parameter ranges for the layers.
    """
    d1, vs1, d2, vs2 = np.array([]), np.array([]), np.array([]), np.array([])

    for layer in layers:
        d1 = np.append(d1, layer.thickness[1])
        vs1 = np.append(vs1, layer.velocity_s[0])
        d2 = np.append(d2, layer.thickness[0])
        vs2 = np.append(vs2, layer.velocity_s[1])
    
    d2[-1] = np.sum(d1) - np.sum(d2[:-1])
    depthplot(d1, vs1, None, plot_args={"color": "mediumpurple", "linewidth": 2, "label": "Bounds"}, ax=ax)
    depthplot(d2, vs2, None, plot_args={"color": "mediumpurple", "linewidth": 2}, ax=ax)

    return ax

def threshold(res, num=1000):
    """
    Thresholding function for selecting top N models based on misfits.
    """
    sorted_idx = np.argsort(res.misfits)
    top_idx = sorted_idx[:num]
    return InversionResult(xs=res.xs[top_idx], models=res.models[top_idx], misfits=res.misfits[top_idx], global_misfits=res.global_misfits, maxiter=res.maxiter, popsize=res.popsize)

def cal_mean_model(res, dz=0.01, zmax=None):
    """
    Calculate the mean model from the inversion results.
    """
    models = res.models
    misfits = res.misfits

    def profile(thickness, parameter, z):
        parameter = np.atleast_2d(parameter).T
        zp, fp = resample(thickness, parameter, dz)
        zp = zp.cumsum()
        return np.interp(z, zp, fp[:, 0])

    if zmax is None:
        zmax = max([model[:, 0].sum() for model in models])

    nz = int(np.ceil(zmax / dz))
    z = dz * np.arange(nz)

    n_params = models[0].shape[1] - 1
    mean_model = np.column_stack([
        np.average(
            [profile(model[:, 0], model[:, i + 1], z) for model in models],
            axis=0,
            weights=1.0 / misfits
        )
        for i in range(n_params)
    ])
    
    d = np.full_like(z, dz)
    mean_model = np.column_stack((d, mean_model))

    merged_z, merged_model = [z[0]], [mean_model[0]]
    for i in range(1, len(mean_model)):
        if np.allclose(mean_model[i, 1:], mean_model[i - 1, 1:]):
            merged_z[-1] += dz
            merged_model[-1][1:] = np.average([merged_model[-1][1:], mean_model[i, 1:]], axis=0)
        else:
            merged_z.append(z[i] - z[i-1])
            merged_model.append(mean_model[i])

    merged_z = np.array(merged_z)
    merged_model = np.array(merged_model)
    return np.column_stack((merged_z, merged_model[:, 1:]))  # First column is thickness, not cumulative

if __name__ == "__main__":
    for sta in range(250, 400+1):
        invert_1D(sta)


