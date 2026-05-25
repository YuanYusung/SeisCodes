# This script computes and plots the sensitivity kernels for a given velocity model.
# It uses the PhaseSensitivity class from the `disba` library to compute the kernels 
# The velocity model is gridified and smoothed to obtain smoother sensitivity kernel curves.

# Author: Yuan Yusong
# Created on: 2026-03-19

import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndimage
from scipy.interpolate import interp1d
from disba import PhaseSensitivity


# Function to convert the velocity model into a grid model with finer depth intervals
def grid_model(model, dz):
    """
    Converts a given velocity model into a grid model with finer depth intervals.
    """
    # Create a depth grid from 0 to the total depth of the model
    depth_grid = np.arange(0.0, np.sum(model[:, 0]) + dz, dz)

    # Extract thickness, Vp, Vs, and density from the model
    thickness = model[:, 0]
    vp = model[:, 1] 
    vs = model[:, 2]
    rho = model[:, 3]

    # Calculate cumulative depth for each layer
    depths = np.cumsum(thickness)
    depths = np.insert(depths, 0, 0.0)  # Add depth 0 at the start
    
    # Interpolation setup
    model_depths, model_vp, model_vs, model_rho = [], [], [], []
    
    for i in range(len(thickness)):
        model_depths.extend([depths[i], depths[i+1]])
        model_vp.extend([vp[i], vp[i]])
        model_vs.extend([vs[i], vs[i]])
        model_rho.extend([rho[i], rho[i]])

    # Interpolate properties to match grid depths
    vp_fn = interp1d(model_depths, model_vp, kind='previous', fill_value="extrapolate")
    vs_fn = interp1d(model_depths, model_vs, kind='previous', fill_value="extrapolate")
    rho_fn = interp1d(model_depths, model_rho, kind='previous', fill_value="extrapolate")
    
    # Adjust thickness array to match grid depth intervals
    thickness = np.diff(depth_grid)
    thickness = np.append(thickness, thickness[-1])  # Extend last value
    
    # Construct grid model with thickness, Vp, Vs, and density
    grid_model = np.column_stack((thickness, vp_fn(depth_grid), vs_fn(depth_grid), rho_fn(depth_grid)))
    
    return grid_model


# Function to calculate sensitivity kernels for specified periods
def cal_skr(model, pers, mode=0, wave="rayleigh", parameter="velocity_s"):
    """
    Calculates the sensitivity kernels for a given period for each model layer.
    """
    # Initialize PhaseSensitivity object
    ps = PhaseSensitivity(*model.T)
    
    kernels = []  # List to store kernels for each period
    for period in pers:
        # Compute sensitivity kernel for the given period
        skr = ps(period, mode=mode, wave=wave, parameter=parameter)

        # Exclude the last grid point because it is located near the model bottom boundary
        # and may produce unstable or abnormal sensitivity values.
        kernels.append(skr.kernel[:-1])
        # Adjust depth for midpoint of each layer
        depth = skr.depth[:-1] + 0.5 * model[:-1, 0]
    
    return kernels, depth


# Function to plot the sensitivity kernels
def plot_skr(kernels, depth, pers, zmax=85):
    """
    Plots the sensitivity kernels for the given periods.
    """
    fig, ax = plt.subplots()
    
    # Plot each kernel with normalized values
    for kernel, per, color in zip(kernels, pers, ["black", "blue", "red", "green"]):
        ax.plot(kernel, depth, label=f"{per}s", color=color)
    
    # Set plot labels, title, and legend
    ax.set_xlabel("Normalized Sensitivity")
    ax.set_ylabel("Depth (km)")
    ax.set_title("Sensitivity Kernel")
    plt.ylim(zmax, 0)  # Set depth limits
    plt.legend()
    
    # Save the plot as a PNG file
    fig.tight_layout()
    plt.savefig("sensitivity_kernel.png", dpi=300)

def plot_skr_2d(kernels, depth, pers, zmax=85, normalize_each_period=False):
    """
    Plots 2D sensitivity kernels using pcolormesh.

    X-axis: period
    Y-axis: depth
    Color: sensitivity kernel value
    """

    K = np.asarray(kernels)  # shape: (n_periods, n_depths)

    if normalize_each_period:
        max_abs = np.max(np.abs(K), axis=1, keepdims=True)
        max_abs[max_abs == 0] = 1.0
        K = K / max_abs
        cbar_label = "Normalized Sensitivity"
    else:
        cbar_label = "Sensitivity Kernel"

    pers = np.asarray(pers)
    depth = np.asarray(depth)

    fig, ax = plt.subplots(figsize=(6.5, 5))

    vmax = np.nanmax(np.abs(K))
    vmin = -vmax

    im = ax.pcolormesh(
        pers,
        depth,
        K.T,
        shading="auto",
        cmap="coolwarm",
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xlabel("Period (s)")
    ax.set_ylabel("Depth (km)")
    ax.set_title("2D Sensitivity Kernel")
    ax.set_ylim(zmax, 0)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    fig.tight_layout()
    plt.savefig("sensitivity_kernel_2d.png", dpi=300)
    plt.show()
# An example velocity model with thickness, Vp, Vs, and density (units: km, km/s, km/s, g/cm3)
velocity_model = np.array([
   [10.0, 7.00, 3.50, 2.00],
   [10.0, 6.80, 3.40, 2.00],
   [10.0, 7.00, 3.50, 2.00],
   [10.0, 7.60, 3.80, 2.00],
   [10.0, 8.40, 4.20, 2.00],
   [10.0, 9.00, 4.50, 2.00],
   [10.0, 9.40, 4.70, 2.00],
   [10.0, 9.60, 4.80, 2.00],
   [10.0, 9.50, 4.75, 2.00],
])

# Convert the velocity model to a grid model with a small depth increment
model = grid_model(velocity_model, dz=2)

# Apply Gaussian smoothing to the model for better continuity
smoothed_model = ndimage.gaussian_filter1d(model, sigma=5, axis=0)

# Define periods for 1D sensitivity kernel curves
pers = [10.0, 20.0, 25.0, 30.0]

# Compute the sensitivity kernels for each period
kernels, depth = cal_skr(
    smoothed_model,
    pers,
    mode=0,
    wave="rayleigh",
    parameter="velocity_s"
)

# Plot 1D sensitivity kernels
plot_skr(kernels, depth, pers)


# ---------------------------------------------------------
# 2D sensitivity kernel plot
# Use denser period sampling for a smoother 2D image
# ---------------------------------------------------------
pers_2d = np.linspace(10, 30.0, 21)

kernels_2d, depth_2d = cal_skr(
    smoothed_model,
    pers_2d,
    mode=0,
    wave="rayleigh",
    parameter="velocity_s"
)

plot_skr_2d(
    kernels_2d,
    depth_2d,
    pers_2d,
    zmax=85,
    normalize_each_period=False
)
