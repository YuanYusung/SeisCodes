# Script: Eikonal Tomography for 1D Linear Array
#
# Author: Yuan Yusong, China University of Geosciences, Wuhan
# Date: 2025/12/06
#
# This script is desig/ed for eikonal tomography to model phase velocity based on phase travel time data from a 1D linear array of stations. 
#
# File Structure:
# - "Data/station_locations.txt": Contains the station IDs and locations in longitude and latitude.
# - "Data/Disp_M1": Contains travel time data files for each virtual source-receiver pair.

from geopy.distance import geodesic
import os
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Function to get the distance to a reference station (RR01)
def get_distoRR01(sta_id, station_file="Data/station_locations.txt", step_size=50):
    sta_ids, sta_locs = [], []
    with open(station_file, 'r') as file:
        for line in file:
            parts = line.split()
            sta_ids.append(int(parts[0]))
            sta_locs.append((float(parts[1]), float(parts[2])))

    start_station = sta_locs[0]
    end_station = sta_locs[-1]

    lat_diff = end_station[0] - start_station[0]
    lon_diff = end_station[1] - start_station[1]
    total_distance = geodesic(start_station, end_station).meters
    num_points = int(total_distance // step_size)
    
    # Generate grid points along the line between the first and last station
    grid_points = [(start_station[0] + (lat_diff * i) * (step_size / total_distance),
                    start_station[1] + (lon_diff * i) * (step_size / total_distance))
                   for i in range(num_points + 1)]

    sta_loc = sta_locs[sta_ids.index(sta_id)]

    nearest_point = min(grid_points, key=lambda point: geodesic(sta_loc, point).meters)
    distance_km = geodesic(start_station, nearest_point).km  # Convert to kilometers
    return round(distance_km, 2)

# Function for interpolating travel time data
def interp_travel_time(vir_src):
    fmin = 1.5
    fmax = 3.1
    df = 0.05
    freqs_grid = np.arange(fmin, fmax + df, df)

    tph_arr = []
    locs = []
    loc_src = get_distoRR01(vir_src)

    locs_grid = np.arange(0, 1.6 + 0.05, 0.05)
    tph_grid = np.full((len(freqs_grid), len(locs_grid)), np.nan)

    files = [f for f in os.listdir("Data/Disp_M1") if str(vir_src) in f.split('_')[2].split('-')]
    if len(files) == 0:
        vel_grid = tph_grid.copy()
        return locs_grid, freqs_grid, vel_grid

    for file_name in files:
        file_path = os.path.join("Data/Disp_M1", file_name)
        with open(file_path, 'r') as f:
            header = f.readline().strip().split(',')
            i0, j0, dist_raw = int(header[0]), int(header[1]), float(header[2])
            freqs = []
            tphs = []
            for line in f:
                freq, tph = line.strip().split(',')
                freqs.append(float(freq))
                tphs.append(float(tph))

        interp_func = interp1d(freqs, tphs, kind='linear', bounds_error=False, fill_value="extrapolate")
        interp = interp_func(freqs_grid)

        valid_mask = (freqs_grid >= min(freqs)-df) & (freqs_grid <= fmax+df)
        interp[~valid_mask] = np.nan

        dist_new = abs(get_distoRR01(i0)-get_distoRR01(j0))
        interp_corr = interp / dist_raw * dist_new

        if j0 == vir_src:
            interp_corr *= -1
            dist_new *= -1

        loc = dist_new + loc_src

        if loc in locs:
            index = locs.index(loc)
            tph_arr[index] = (tph_arr[index] + np.array(interp_corr)) / 2
        else:
            locs.append(loc)
            tph_arr.append(np.array(interp_corr))

    tph_arr = np.array(tph_arr)
    locs = np.array(locs)

    for i, loc in enumerate(locs):
        loc_index = np.abs(locs_grid - loc).argmin()
        tph_grid[loc_index] = np.array(tph_arr[i])

    gap_index = np.abs(locs_grid - 0.85).argmin()
    tph_grid[gap_index, :] = (tph_grid[gap_index - 1, :] + tph_grid[gap_index + 1, :]) / 2

    d_travel = nan_safe_gradient(tph_grid, step=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        vel_grid = dx / d_travel
        vel_grid[np.isnan(vel_grid)] = np.nan

    return locs_grid, freqs_grid, vel_grid

# Safe gradient calculation handling NaN values
def nan_safe_gradient(data, step=1):
    grad = np.full_like(data, np.nan, dtype=float)
    for i in range(step, data.shape[0] - step):
        for j in range(data.shape[1]):
            if np.isnan(data[i - step, j]) or np.isnan(data[i + step, j]):
                grad[i, j] = np.nan
            else:
                grad[i, j] = (data[i + step, j] - data[i - step, j]) / (2 * step)
    return grad

# Stack and plot phase velocity data from multiple virtul sources.
def stack_and_plot():
    if_run = 0
    if if_run:
        src_list = range(1, 47 + 1)
        velomaps = []
        locs, freqs = None, None

        for src in src_list:
            locs, freqs, vel_grid = interp_travel_time(src)
            print(f"Sucessfully processed virtual source {src}")
            velomaps.append(vel_grid)

        velocity_stack = np.stack(velomaps, axis=0)

        valid_mask = ~np.all(np.isnan(velocity_stack), axis=0)

        mean_velocity = np.full(valid_mask.shape, np.nan, dtype=float)
        std_velocity = np.full(valid_mask.shape, np.nan, dtype=float)

        mean_velocity[valid_mask] = np.nanmean(velocity_stack[:, valid_mask], axis=0)
        std_velocity[valid_mask] = np.nanstd(velocity_stack[:, valid_mask], axis=0)

        save_data = {
            "locs": locs,
            "freqs": freqs,
            "mean_velocity": mean_velocity,
            "std_velocity": std_velocity
        }

        with open("Data/Eikonal_tomo_res.pkl", "wb") as f:
            pickle.dump(save_data, f)

    data = np.load("Data/Eikonal_tomo_res.pkl", allow_pickle=True)
    locs = data["locs"]
    freqs = data["freqs"]
    mean_velocity = data['mean_velocity']
    std_velocity = data['std_velocity']

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    im1 = axes[0].imshow(mean_velocity.T, cmap='jet_r', rasterized=True,
                         extent=[locs.min(), locs.max(), freqs.min(), freqs.max()],
                         aspect='auto', origin='lower', interpolation='kaiser', vmin=0.7, vmax=2.)

    x1, x2, x3 = get_distoRR01(21), get_distoRR01(31), get_distoRR01(42)
    xes = [x1, x2, x3]
    axes[0].vlines(x=xes, ymin=freqs.min(), ymax=freqs.max(), linestyle='dashed', colors='k', lw=2.)

    cbar1 = plt.colorbar(im1, ax=axes[0])
    cbar1.set_label("Velocity (km/s)", fontsize=14)
    cbar1.ax.tick_params(labelsize=12)

    axes[0].set_ylabel("Frequency (Hz)", fontsize=16)
    axes[0].set_title("Phase Velocity of Mode 1 from Eik Tomo", fontsize=16)
    axes[0].tick_params(labelsize=14)

    im2 = axes[1].imshow(std_velocity.T / mean_velocity.T * 100, rasterized=True,
                         extent=[locs.min(), locs.max(), freqs.min(), freqs.max()],
                         cmap='coolwarm', aspect='auto', origin='lower', interpolation='kaiser', vmin=0., vmax=5)

    cbar2 = plt.colorbar(im2, ax=axes[1])
    cbar2.set_label("Std./Velocity (%)", fontsize=14)
    cbar2.ax.tick_params(labelsize=14)

    axes[1].set_xlabel("Distance to RR01 (km)", fontsize=16)
    axes[1].set_ylabel("Frequency (Hz)", fontsize=16)
    axes[1].set_title("Uncertainty", fontsize=16)
    axes[1].tick_params(labelsize=12)

    plt.tight_layout()
    plt.savefig("Figures/Figure_eiktomo_map.png")
    plt.show()

if __name__ == "__main__":
    stack_and_plot()
