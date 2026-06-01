# -*- coding: utf-8 -*-
"""
Straight-ray 2-D seismic tomography demo on an X-Y grid
======================================================

This script is a compact learning-oriented example of 2-D straight-ray travel-time
seismic tomography. It includes:

1. Forward modelling using a checkerboard velocity model on an x-y plane.
2. User-controllable 2-D station distribution.
3. All station pairs connected as straight rays.
4. Linear inversion for slowness perturbations with smoothing and damping.
5. Release-style plotting of true model, ray geometry, ray density, recovered model,
   model error, and travel-time fitting.

Assumptions
-----------
- The model is a 2-D x-y plane.
- Rays are straight; ray bending is not considered.
- The inversion solves for slowness perturbation relative to a homogeneous initial model.

Dependencies
------------
    numpy
    matplotlib

Run
---
    python straight_ray_tomography_xy_release.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Configuration
# =============================================================================

StationMode = Literal["random", "grid", "custom"]


@dataclass
class TomographyConfig:
    """Configuration for the straight-ray tomography demo on an x-y grid."""

    # Model grid
    nx: int = 10
    ny: int = 10
    lx: float = 30.0  # km
    ly: float = 30.0  # km

    # Checkerboard true model
    v0: float = 4.0  # km/s
    checkerboard_amp: float = 0.10
    checkerboard_blocks_x: int = 3
    checkerboard_blocks_y: int = 3

    # Station geometry
    station_mode: StationMode = "random"
    n_stations: int = 60
    station_seed: int = 42
    grid_stations_x: int = 8
    grid_stations_y: int = 5
    station_margin: float = 0.5  # km, used for random stations
    custom_stations: tuple[tuple[float, float], ...] = ()

    # Ray-pair selection
    min_station_distance: float = 3.0  # km
    max_station_distance: float | None = None  # km

    # Data noise
    noise_std: float = 0.1  # s
    noise_seed: int = 42

    # Inversion regularization
    lambda_smooth: float = 1.0
    lambda_damp: float = 0.2

    # Plotting
    velocity_clim: tuple[float, float] = (3.2, 4.8)
    ray_plot_step: int = 1
    output_figure: str = "straight_ray_tomography_xy.png"
    figure_dpi: int = 300

    @property
    def n_model_cells(self) -> int:
        """Total number of model cells."""
        return self.nx * self.ny


# =============================================================================
# Model construction
# =============================================================================


def build_checkerboard_velocity(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
    v0: float,
    amp: float,
    block_nx: int,
    block_ny: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a 2-D checkerboard velocity model on an x-y grid.

    Parameters
    ----------
    nx, ny
        Number of grid cells in x and y directions.
    lx, ly
        Model size in x and y directions, in km.
    v0
        Background velocity in km/s.
    amp
        Relative checkerboard perturbation, for example 0.10 means +/-10%.
    block_nx, block_ny
        Number of checkerboard blocks in x and y directions.

    Returns
    -------
    x, y
        Cell-center coordinates.
    velocity
        True velocity model with shape (ny, nx).
    """

    dx = lx / nx
    dy = ly / ny

    x = np.linspace(0.0, lx, nx, endpoint=False) + dx / 2.0
    y = np.linspace(0.0, ly, ny, endpoint=False) + dy / 2.0
    X, Y = np.meshgrid(x, y)

    ix_block = np.floor(X / (lx / block_nx)).astype(int)
    iy_block = np.floor(Y / (ly / block_ny)).astype(int)

    checker_pattern = ((ix_block + iy_block) % 2) * 2 - 1
    velocity = v0 * (1.0 + amp * checker_pattern)

    return x, y, velocity


# =============================================================================
# Station and ray geometry
# =============================================================================


def make_random_stations(
    lx: float,
    ly: float,
    n_stations: int,
    seed: int = 42,
    margin: float = 0.5,
) -> list[tuple[float, float]]:
    """Generate randomly distributed stations inside the 2-D x-y model."""

    if lx <= 2.0 * margin or ly <= 2.0 * margin:
        raise ValueError("Model size must be larger than twice the station margin.")

    rng = np.random.default_rng(seed)
    xs = rng.uniform(margin, lx - margin, n_stations)
    ys = rng.uniform(margin, ly - margin, n_stations)
    return list(zip(xs, ys))


def make_grid_stations(
    lx: float,
    ly: float,
    nx_station: int,
    ny_station: int,
    margin: float = 1.0,
) -> list[tuple[float, float]]:
    """Generate regularly spaced stations inside the 2-D x-y model."""

    if nx_station < 1 or ny_station < 1:
        raise ValueError("nx_station and ny_station must be positive integers.")

    xs = np.linspace(margin, lx - margin, nx_station)
    ys = np.linspace(margin, ly - margin, ny_station)
    return [(float(x), float(y)) for y in ys for x in xs]


def get_stations(config: TomographyConfig) -> list[tuple[float, float]]:
    """Create stations according to the selected station mode."""

    if config.station_mode == "random":
        return make_random_stations(
            lx=config.lx,
            ly=config.ly,
            n_stations=config.n_stations,
            seed=config.station_seed,
            margin=config.station_margin,
        )

    if config.station_mode == "grid":
        return make_grid_stations(
            lx=config.lx,
            ly=config.ly,
            nx_station=config.grid_stations_x,
            ny_station=config.grid_stations_y,
        )

    if config.station_mode == "custom":
        if len(config.custom_stations) < 2:
            raise ValueError("custom_stations must contain at least two stations.")
        return [(float(x), float(y)) for x, y in config.custom_stations]

    raise ValueError(f"Unsupported station mode: {config.station_mode}")


def make_rays_from_stations(
    stations: Iterable[tuple[float, float]],
    min_distance: float = 0.0,
    max_distance: float | None = None,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """
    Connect all valid station pairs to form straight rays.

    Parameters
    ----------
    stations
        Station coordinates, each station as (x, y), in km.
    min_distance
        Minimum allowed inter-station distance in km.
    max_distance
        Maximum allowed inter-station distance in km. If None, no upper limit is used.

    Returns
    -------
    rays
        A list of straight rays, each ray as ((x1, y1), (x2, y2)).
    """

    station_list = list(stations)
    rays: list[tuple[tuple[float, float], tuple[float, float]]] = []

    for i in range(len(station_list)):
        for j in range(i + 1, len(station_list)):
            p0 = station_list[i]
            p1 = station_list[j]
            distance = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))

            if distance < min_distance:
                continue
            if max_distance is not None and distance > max_distance:
                continue

            rays.append((p0, p1))

    if not rays:
        raise ValueError("No rays were generated. Check station distribution and distance limits.")

    return rays


# =============================================================================
# Forward operator
# =============================================================================


def ray_to_matrix_row(
    p0: tuple[float, float],
    p1: tuple[float, float],
    nx: int,
    ny: int,
    lx: float,
    ly: float,
) -> np.ndarray:
    """
    Compute the path length of one straight ray in each model cell.

    The forward equation is
        t_i = sum_j G_ij * s_j,
    where G_ij is the ray length in cell j and s_j is slowness.
    """

    x0, y0 = p0
    x1, y1 = p1
    dx = lx / nx
    dy = ly / ny

    tau_values = [0.0, 1.0]

    if x1 != x0:
        for xb in np.linspace(0.0, lx, nx + 1):
            tau = (xb - x0) / (x1 - x0)
            if 0.0 < tau < 1.0:
                tau_values.append(float(tau))

    if y1 != y0:
        for yb in np.linspace(0.0, ly, ny + 1):
            tau = (yb - y0) / (y1 - y0)
            if 0.0 < tau < 1.0:
                tau_values.append(float(tau))

    tau_values = np.array(sorted(set(np.round(tau_values, 12))))
    row = np.zeros(nx * ny, dtype=float)
    total_ray_length = float(np.hypot(x1 - x0, y1 - y0))

    for tau_a, tau_b in zip(tau_values[:-1], tau_values[1:]):
        if tau_b <= tau_a:
            continue

        tau_mid = 0.5 * (tau_a + tau_b)
        xm = x0 + tau_mid * (x1 - x0)
        ym = y0 + tau_mid * (y1 - y0)

        ix = int(np.floor(xm / dx))
        iy = int(np.floor(ym / dy))
        ix = min(max(ix, 0), nx - 1)
        iy = min(max(iy, 0), ny - 1)

        segment_length = (tau_b - tau_a) * total_ray_length
        row[iy * nx + ix] += segment_length

    return row


def build_forward_matrix(
    rays: Iterable[tuple[tuple[float, float], tuple[float, float]]],
    nx: int,
    ny: int,
    lx: float,
    ly: float,
) -> np.ndarray:
    """Build the straight-ray path-length matrix G."""

    return np.vstack([ray_to_matrix_row(p0, p1, nx, ny, lx, ly) for p0, p1 in rays])


# =============================================================================
# Inversion
# =============================================================================


def build_smoothness_matrix(nx: int, ny: int) -> np.ndarray:
    """Build a first-order finite-difference smoothing matrix on an x-y grid."""

    rows: list[np.ndarray] = []

    for iy in range(ny):
        for ix in range(nx):
            k = iy * nx + ix

            if ix < nx - 1:
                row = np.zeros(nx * ny, dtype=float)
                row[k] = -1.0
                row[k + 1] = 1.0
                rows.append(row)

            if iy < ny - 1:
                row = np.zeros(nx * ny, dtype=float)
                row[k] = -1.0
                row[k + nx] = 1.0
                rows.append(row)

    return np.vstack(rows)


def invert_slowness_perturbation(
    G: np.ndarray,
    t_obs: np.ndarray,
    v0: float,
    nx: int,
    ny: int,
    lambda_smooth: float,
    lambda_damp: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Invert for slowness perturbation relative to a homogeneous initial model.

    Objective function:
        ||G dm - r||^2 + lambda_smooth^2 ||L dm||^2 + lambda_damp^2 ||dm||^2
    """

    n_model = nx * ny
    s0_vec = np.full(n_model, 1.0 / v0, dtype=float)
    residual = t_obs - G @ s0_vec

    L = build_smoothness_matrix(nx, ny)
    A = np.vstack((G, lambda_smooth * L, lambda_damp * np.eye(n_model)))
    b = np.concatenate((residual, np.zeros(L.shape[0]), np.zeros(n_model)))

    dm_est = np.linalg.lstsq(A, b, rcond=None)[0]
    s_est_vec = np.maximum(s0_vec + dm_est, 1e-6)
    v_est = 1.0 / s_est_vec.reshape(ny, nx)

    t_pred = G @ s_est_vec
    rms_misfit = float(np.sqrt(np.mean((t_obs - t_pred) ** 2)))

    return v_est, t_pred, rms_misfit


# =============================================================================
# Plotting
# =============================================================================


def station_arrays(stations: Iterable[tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert station coordinate list to x and y arrays."""

    station_list = list(stations)
    return np.array([p[0] for p in station_list]), np.array([p[1] for p in station_list])


def overlay_stations(ax: plt.Axes, stations: Iterable[tuple[float, float]], label: bool = False) -> None:
    """Overlay station locations on a model plot."""

    station_x, station_y = station_arrays(stations)
    ax.scatter(
        station_x,
        station_y,
        s=35,
        marker="^",
        c="yellow",
        edgecolors="black",
        linewidths=0.5,
        label="Stations" if label else None,
        zorder=5,
    )
    if label:
        ax.legend(loc="upper right", frameon=True)


def setup_xy_axis(ax: plt.Axes, lx: float, ly: float) -> None:
    """Apply common axis settings for x-y model plots."""

    ax.set_xlim(0.0, lx)
    ax.set_ylim(0.0, ly)
    ax.set_aspect("equal")
    ax.set_xlabel("X distance (km)")
    ax.set_ylabel("Y distance (km)")


def plot_results(
    v_true: np.ndarray,
    v_est: np.ndarray,
    ray_density: np.ndarray,
    rays: list[tuple[tuple[float, float], tuple[float, float]]],
    stations: list[tuple[float, float]],
    lx: float,
    ly: float,
    t_obs: np.ndarray,
    t_pred: np.ndarray,
    output_figure: str | Path = "straight_ray_tomography_xy.png",
    figure_dpi: int = 300,
    velocity_clim: tuple[float, float] = (3.2, 4.8),
    ray_plot_step: int = 1,
) -> None:
    """Plot the true model, ray geometry, ray density, recovered model, error, and fit."""

    vmin, vmax = velocity_clim
    error = v_est - v_true
    error_abs = float(np.nanmax(np.abs(error)))
    if error_abs == 0.0:
        error_abs = 1.0

    station_x, station_y = station_arrays(stations)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # 1. True model with stations
    im0 = axes[0, 0].imshow(
        v_true,
        extent=[0, lx, 0, ly],
        origin="lower",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        cmap="seismic",
        interpolation="bilinear",
    )
    axes[0, 0].set_title("True checkerboard velocity + stations")
    setup_xy_axis(axes[0, 0], lx, ly)
    overlay_stations(axes[0, 0], stations, label=True)
    plt.colorbar(im0, ax=axes[0, 0], label="Velocity (km/s)")

    # 2. Station-pair ray geometry
    axes[0, 1].set_title("Station-pair ray geometry")
    ray_plot_step = max(int(ray_plot_step), 1)
    for i, (p0, p1) in enumerate(rays):
        if i % ray_plot_step == 0:
            axes[0, 1].plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                linewidth=0.2,
                alpha=0.45,
                color="black",
                zorder=1,
            )
    axes[0, 1].scatter(
        station_x,
        station_y,
        s=35,
        marker="^",
        c="yellow",
        edgecolors="black",
        linewidths=0.5,
        label="Stations",
        zorder=3,
    )
    setup_xy_axis(axes[0, 1], lx, ly)
    axes[0, 1].legend(loc="upper right", frameon=True)

    # 3. Ray density
    im2 = axes[0, 2].imshow(
        ray_density,
        extent=[0, lx, 0, ly],
        origin="lower",
        aspect="equal",
        cmap="hot_r",
        interpolation="nearest",
    )
    axes[0, 2].set_title("Ray density")
    setup_xy_axis(axes[0, 2], lx, ly)
    overlay_stations(axes[0, 2], stations, label=False)
    plt.colorbar(im2, ax=axes[0, 2], label="Number of rays per cell")

    # 4. Recovered model with stations
    im3 = axes[1, 0].imshow(
        v_est,
        extent=[0, lx, 0, ly],
        origin="lower",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
        cmap="seismic",
        interpolation="bilinear",
    )
    axes[1, 0].set_title("Recovered velocity + stations")
    setup_xy_axis(axes[1, 0], lx, ly)
    overlay_stations(axes[1, 0], stations, label=True)
    plt.colorbar(im3, ax=axes[1, 0], label="Velocity (km/s)")

    # 5. Velocity error
    im4 = axes[1, 1].imshow(
        error,
        extent=[0, lx, 0, ly],
        origin="lower",
        aspect="equal",
        vmin=-error_abs,
        vmax=error_abs,
        cmap="seismic",
        interpolation="bilinear",
    )
    axes[1, 1].set_title("Velocity error: recovered - true")
    setup_xy_axis(axes[1, 1], lx, ly)
    overlay_stations(axes[1, 1], stations, label=False)
    plt.colorbar(im4, ax=axes[1, 1], label="Error (km/s)")

    # 6. Travel-time fitting
    axes[1, 2].scatter(t_obs, t_pred, s=8, alpha=0.6)
    min_t = float(min(t_obs.min(), t_pred.min()))
    max_t = float(max(t_obs.max(), t_pred.max()))
    axes[1, 2].plot([min_t, max_t], [min_t, max_t], "k--", linewidth=1)
    axes[1, 2].set_title("Travel-time fitting")
    axes[1, 2].set_xlabel("Observed travel time (s)")
    axes[1, 2].set_ylabel("Predicted travel time (s)")
    axes[1, 2].set_aspect("equal", adjustable="box")
    axes[1, 2].grid(True, linewidth=0.3, alpha=0.5)

    plt.tight_layout()
    output_figure = Path(output_figure)
    fig.savefig(output_figure, dpi=figure_dpi, bbox_inches="tight")
    plt.show()


# =============================================================================
# Main workflow
# =============================================================================


def run_tomography(config: TomographyConfig) -> dict[str, object]:
    """Run forward modelling, inversion, and plotting."""

    _, _, v_true = build_checkerboard_velocity(
        nx=config.nx,
        ny=config.ny,
        lx=config.lx,
        ly=config.ly,
        v0=config.v0,
        amp=config.checkerboard_amp,
        block_nx=config.checkerboard_blocks_x,
        block_ny=config.checkerboard_blocks_y,
    )

    s_true_vec = (1.0 / v_true).ravel()

    stations = get_stations(config)
    rays = make_rays_from_stations(
        stations,
        min_distance=config.min_station_distance,
        max_distance=config.max_station_distance,
    )

    G = build_forward_matrix(rays, config.nx, config.ny, config.lx, config.ly)
    t_true = G @ s_true_vec

    rng = np.random.default_rng(config.noise_seed)
    t_obs = t_true + rng.normal(0.0, config.noise_std, size=t_true.shape)

    v_est, t_pred, rms_misfit = invert_slowness_perturbation(
        G=G,
        t_obs=t_obs,
        v0=config.v0,
        nx=config.nx,
        ny=config.ny,
        lambda_smooth=config.lambda_smooth,
        lambda_damp=config.lambda_damp,
    )

    ray_density = np.count_nonzero(G > 0.0, axis=0).reshape(config.ny, config.nx)

    print("\nStraight-ray tomography demo on an X-Y grid")
    print("-" * 43)
    print(f"Model size              : {config.lx:.1f} km × {config.ly:.1f} km")
    print(f"Grid                    : {config.nx} × {config.ny} = {config.n_model_cells} cells")
    print(f"Station mode            : {config.station_mode}")
    print(f"Number of stations      : {len(stations)}")
    print(f"Number of rays          : {len(rays)}")
    print(f"Travel-time range       : {t_true.min():.3f} s to {t_true.max():.3f} s")
    print(f"Noise standard deviation: {config.noise_std:.3f} s")
    print(f"lambda_smooth           : {config.lambda_smooth:.3f}")
    print(f"lambda_damp             : {config.lambda_damp:.3f}")
    print(f"RMS travel-time residual: {rms_misfit:.4f} s")
    print(f"Output figure           : {config.output_figure}\n")

    plot_results(
        v_true=v_true,
        v_est=v_est,
        ray_density=ray_density,
        rays=rays,
        stations=stations,
        lx=config.lx,
        ly=config.ly,
        t_obs=t_obs,
        t_pred=t_pred,
        output_figure=config.output_figure,
        figure_dpi=config.figure_dpi,
        velocity_clim=config.velocity_clim,
        ray_plot_step=config.ray_plot_step,
    )

    return {
        "config": config,
        "v_true": v_true,
        "v_est": v_est,
        "ray_density": ray_density,
        "stations": stations,
        "rays": rays,
        "G": G,
        "t_true": t_true,
        "t_obs": t_obs,
        "t_pred": t_pred,
        "rms_misfit": rms_misfit,
    }


def main() -> None:
    """Entry point for command-line execution."""

    config = TomographyConfig(
        # You can modify the parameters here for learning experiments.
        nx=10,
        ny=10,
        lx=30.0,
        ly=30.0,
        v0=4.0,
        checkerboard_amp=0.10,
        checkerboard_blocks_x=3,
        checkerboard_blocks_y=3,
        station_mode="random",  # options: "random", "grid", "custom"
        n_stations=60,
        station_seed=42,
        min_station_distance=3.0,
        max_station_distance=None,
        noise_std=0.1,
        lambda_smooth=1.0,
        lambda_damp=0.2,
        velocity_clim=(3.2, 4.8),
        ray_plot_step=1,
        output_figure="straight_ray_tomography_xy.png",
        figure_dpi=300,
    )

    # Example for custom stations:
    # config.station_mode = "custom"
    # config.custom_stations = (
    #     (2.0, 2.0), (8.0, 4.5), (12.0, 9.0),
    #     (16.0, 14.0), (22.0, 18.0), (27.0, 26.0),
    # )

    run_tomography(config)


if __name__ == "__main__":
    main()
