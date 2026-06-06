"""
Clean F-J / MFJ / phase-shift dispersion spectrum demo.

This script builds synthetic ambient-noise cross-correlation functions
containing one fundamental mode and one higher mode, then compares:

1. Phase-shift dispersion imaging
2. Classical frequency-Bessel (F-J) transform
3. Optional modified F-J (MFJ) transform with Hankel kernels

Units
-----
distance: km
phase velocity: km/s
frequency: Hz
time: s
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import hankel1,  j0


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SyntheticConfig:
    """Configuration for synthetic cross-correlation functions."""

    nr: int = 200
    r_min: float = 0.1
    r_max: float = 3.0
    dt: float = 0.01
    nt: int = 8192
    noise_level: float = 0.0
    seed: int = 2026
    add_distance_decay: bool = False
    normalize_each_trace: bool = False
    apply_time_taper: bool = True

    # Modal amplitude settings.
    # Increase a1_amp if you want the higher mode to be stronger.
    a0_amp: float = 1.00
    a1_amp: float = 0.50
    a0_center_hz: float = 1.35
    a1_center_hz: float = 3.25
    a0_width_hz: float = 2.05
    a1_width_hz: float = 1.55

    # Broadband tapers.
    low_taper_hz: float = 0.35
    high_taper_hz: float = 6.0


@dataclass
class ImagingConfig:
    """Configuration for dispersion imaging."""

    c_min: float = 0.6
    c_max: float = 2.6
    nc: int = 500
    fmin: float = 0.5
    fmax: float = 5.0

    use_mfj_for_right_panel: bool = True

    # Classical F-J options.
    fj_use_real_spectrum: bool = True
    fj_positive_only: bool = False
    fj_normalize_each_frequency: bool = True

    # Modified F-J options.
    mfj_use_positive_branch: bool = True
    mfj_normalize_each_frequency: bool = True
    mfj_use_real_part: bool = False
    mfj_taper_frac: float = 0.05

    # Phase-shift options.
    ps_positive_branch: bool = True
    ps_phase_only: bool = True
    ps_taper_frac: float = 0.05


@dataclass
class PlotConfig:
    """Configuration for plotting and output."""

    output_path: Path = Path("phase_shift_vs_fj_spectrum.png")
    dpi: int = 300
    waveform_time_window_s: float = 20.0
    waveform_trace_count: int = 30
    waveform_scale: float = 0.4
    cmap: str = "jet"


# =============================================================================
# Basic utilities
# =============================================================================


def _as_1d_float_array(x: np.ndarray, name: str) -> np.ndarray:
    """Convert input to a 1-D float array and validate it."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return arr


def _validate_ccf_inputs(
    cc_time: np.ndarray,
    r: np.ndarray,
    c: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate CCF, distance, and optional velocity arrays."""
    cc_time = np.asarray(cc_time, dtype=float)
    r = _as_1d_float_array(r, "r")

    if cc_time.ndim != 2:
        raise ValueError("cc_time must have shape (nr, nt)")
    if cc_time.shape[0] != r.size:
        raise ValueError("cc_time.shape[0] must match len(r)")
    if np.any(r <= 0):
        raise ValueError("r must be positive")

    if c is not None:
        c = _as_1d_float_array(c, "c")
        if np.any(c <= 0):
            raise ValueError("phase velocities must be positive")

    return cc_time, r, c


def _sort_by_distance(
    cc_time: np.ndarray,
    r: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sort traces by increasing interstation distance."""
    if np.any(np.diff(r) <= 0):
        order = np.argsort(r)
        return cc_time[order], r[order]
    return cc_time, r


def _trapz_distance_weights(r: np.ndarray) -> np.ndarray:
    """Return trapezoidal integration weights for an irregular distance axis."""
    r = _as_1d_float_array(r, "r")

    if r.size < 2:
        raise ValueError("r must contain at least two distances")
    if np.any(np.diff(r) <= 0):
        raise ValueError("r must be strictly increasing")

    dr = np.empty_like(r)
    dr[0] = 0.5 * (r[1] - r[0])
    dr[-1] = 0.5 * (r[-1] - r[-2])
    dr[1:-1] = 0.5 * (r[2:] - r[:-2])
    return dr


def _cosine_taper(n: int, taper_frac: float) -> np.ndarray:
    """Build a two-sided cosine taper."""
    win = np.ones(n)

    if taper_frac <= 0:
        return win

    m = int(taper_frac * n)
    if m > 1:
        taper = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, m)))
        win[:m] = taper
        win[-m:] = taper[::-1]

    return win


def _positive_lag_branch(cc_time: np.ndarray) -> np.ndarray:
    """
    Extract the positive-lag branch from FFT-convention CCFs.

    Input zero lag is assumed to be at index 0. The function first shifts
    zero lag to the center and then keeps the non-negative lag part.
    """
    nt = cc_time.shape[1]
    cc_shift = np.fft.fftshift(cc_time, axes=1)
    return cc_shift[:, nt // 2 :]


def _normalize_columns(image: np.ndarray) -> np.ndarray:
    """Normalize each frequency column by its maximum absolute amplitude."""
    scale = np.max(np.abs(image), axis=0, keepdims=True)
    scale[scale == 0.0] = 1.0
    return image / scale


# =============================================================================
# Dispersion imaging methods
# =============================================================================


def fj_transform(
    cc_time: np.ndarray,
    r: np.ndarray,
    dt: float,
    c: np.ndarray,
    fmin: float,
    fmax: float,
    use_real_spectrum: bool = True,
    positive_only: bool = True,
    normalize_each_frequency: bool = True,
    use_abs_amplitude: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a classical frequency-Bessel dispersion image.

    The discrete approximation is

        I(f, c) = sum_r C(f, r) J0(2*pi*f*r/c) r dr
    """
    cc_time, r, c = _validate_ccf_inputs(cc_time, r, c)
    cc_time, r = _sort_by_distance(cc_time, r)

    nt = cc_time.shape[1]
    freqs_all = np.fft.rfftfreq(nt, dt)
    spec = np.fft.rfft(cc_time, axis=1)

    if use_real_spectrum:
        spec = spec.real

    fmask = (freqs_all >= fmin) & (freqs_all <= fmax)
    freqs = freqs_all[fmask]
    spec = spec[:, fmask]

    weights = _trapz_distance_weights(r) * r
    image_amp = np.zeros((c.size, freqs.size), dtype=float)

    for jf, freq in enumerate(freqs):
        kernel = j0(2.0 * np.pi * freq * r[:, None] / c[None, :])
        vals = spec[:, jf]

        # F-J integral: shape = (nc,)
        integral = (weights * vals) @ kernel

        if use_abs_amplitude:
            image_amp[:, jf] = np.abs(integral)
        else:
            image_amp[:, jf] = np.real(integral)

    if positive_only and not use_abs_amplitude:
        image_amp = np.maximum(image_amp, 0.0)

    if normalize_each_frequency:
        image_amp = _normalize_columns(image_amp)

    return freqs, image_amp

def mfj_transform(
    cc_time: np.ndarray,
    r: np.ndarray,
    dt: float,
    c: np.ndarray,
    fmin: float,
    fmax: float,
    use_positive_branch: bool = True,
    normalize_each_frequency: bool = True,
    use_real_part: bool = False,
    taper_frac: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a modified F-J dispersion image using a Hankel-function kernel.

    For NumPy FFT convention and a one-way synthetic wavefield

        C(f, r) ~ A(f) exp(-i * omega * r / c),

    ``hankel1`` is usually the more suitable matching kernel because its
    asymptotic phase approximately compensates the propagation phase.

    Parameters
    ----------
    cc_time
        Cross-correlation functions or wavefield records, shape = (nr, nt).
    r
        Interstation distances in km, shape = (nr,).
    dt
        Sampling interval in seconds.
    c
        Trial phase velocity grid in km/s, shape = (nc,).
    fmin, fmax
        Frequency band in Hz.
    use_positive_branch
        If True, use the positive-lag branch.
    normalize_each_frequency
        If True, normalize each frequency column.
    use_real_part
        If True, use the real part of the complex integral.
        If False, use the absolute value.
    taper_frac
        Fraction of cosine taper applied to the selected time branch.

    Returns
    -------
    freqs
        Frequency samples in Hz, shape = (nf,).
    image
        Normalized MFJ power image, shape = (nc, nf).
    """

    cc_time, r, c = _validate_ccf_inputs(cc_time, r, c)
    cc_time, r = _sort_by_distance(cc_time, r)

    nt = cc_time.shape[1]
    data = _positive_lag_branch(cc_time) if use_positive_branch else cc_time.copy()
    data = data * _cosine_taper(data.shape[1], taper_frac)[None, :]

    # Zero-pad back to nt so that the frequency samples are comparable
    # with the original F-J transform.
    freqs_all = np.fft.rfftfreq(nt, dt)
    spec = np.fft.rfft(data, n=nt, axis=1)

    fmask = (freqs_all >= fmin) & (freqs_all <= fmax)
    freqs = freqs_all[fmask]
    spec = spec[:, fmask]

    weights = _trapz_distance_weights(r) * r
    image_amp = np.zeros((c.size, freqs.size), dtype=float)
    hankel = hankel1 

    for jf, freq in enumerate(freqs):
        kr = 2.0 * np.pi * freq * r[:, None] / c[None, :]
        kr = np.maximum(kr, 1e-6)

        integral = (weights * spec[:, jf]) @ hankel(0, kr)

        if use_real_part:
            image_amp[:, jf] = np.real(integral)
        else:
            image_amp[:, jf] = np.abs(integral)

    if normalize_each_frequency:
        image_amp = _normalize_columns(image_amp)

    return freqs, image_amp


def phase_shift_transform(
    cc_time: np.ndarray,
    r: np.ndarray,
    dt: float,
    c_grid: np.ndarray,
    fmin: float,
    fmax: float,
    positive_branch: bool = True,
    phase_only: bool = True,
    taper_frac: float = 0.05,
    zero_pad_factor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute a dispersion image using phase-shift stacking.

    Parameters
    ----------
    cc_time
        Cross-correlation functions, shape = (nr, nt).
    r
        Interstation distances in km, shape = (nr,).
    dt
        Sampling interval in seconds.
    c_grid
        Trial phase velocity grid in km/s, shape = (nc,).
    fmin, fmax
        Frequency range in Hz.
    positive_branch
        If True, use the positive-lag branch.
    phase_only
        If True, normalize each complex spectrum to unit amplitude.
    taper_frac
        Fraction of cosine taper applied at both ends.
    zero_pad_factor
        Zero-padding factor before FFT. For example, 2.0 means padding the
        selected time branch to about twice its original length. This increases
        frequency sampling density but does not improve the true physical
        frequency resolution.

    Returns
    -------
    freqs
        Frequency samples in Hz, shape = (nf,).
    image
        Normalized phase-shift dispersion image, shape = (nc, nf).
    """
    cc_time, r, c_grid = _validate_ccf_inputs(cc_time, r, c_grid)
    cc_time, r = _sort_by_distance(cc_time, r)

    # ------------------------------------------------------------
    # 1. Select positive-lag branch if needed.
    # ------------------------------------------------------------
    data = _positive_lag_branch(cc_time) if positive_branch else cc_time.copy()

    # ------------------------------------------------------------
    # 2. Apply time-domain taper before zero padding.
    # ------------------------------------------------------------
    n_data = data.shape[1]
    data = data * _cosine_taper(n_data, taper_frac)[None, :]

    # ------------------------------------------------------------
    # 3. Zero padding before FFT.
    #    Use the next power of 2 for efficient FFT computation.
    # ------------------------------------------------------------
    if zero_pad_factor is None or zero_pad_factor <= 1.0:
        nfft = n_data
    else:
        n_target = int(np.ceil(n_data * zero_pad_factor))
        nfft = int(2 ** np.ceil(np.log2(n_target)))

    freqs_all = np.fft.rfftfreq(nfft, dt)
    spec = np.fft.rfft(data, n=nfft, axis=1)

    # ------------------------------------------------------------
    # 4. Select target frequency band.
    # ------------------------------------------------------------
    fmask = (freqs_all >= fmin) & (freqs_all <= fmax)
    freqs = freqs_all[fmask]
    spec = spec[:, fmask]

    # ------------------------------------------------------------
    # 5. Optional phase-only normalization.
    # ------------------------------------------------------------
    if phase_only:
        spec = spec / (np.abs(spec) + 1e-12)

    # ------------------------------------------------------------
    # 6. Phase-shift stacking.
    # ------------------------------------------------------------
    image = np.zeros((c_grid.size, freqs.size), dtype=float)

    for jf, freq in enumerate(freqs):
        omega = 2.0 * np.pi * freq
        steering = np.exp(1j * omega * r[:, None] / c_grid[None, :])
        stack = np.sum(spec[:, jf][:, None] * steering, axis=0)
        image[:, jf] = np.abs(stack)

    return freqs, _normalize_columns(image)

# =============================================================================
# Synthetic data
# =============================================================================


def build_synthetic_ccf(
    r: np.ndarray,
    config: SyntheticConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build synthetic ambient-noise CCFs with two dispersive modes.

    The synthetic frequency-domain CCF follows

        C(f, r) ~ sum_m A_m(f) J0(2*pi*f*r/c_m(f)).

    Returns
    -------
    cc_time
        Synthetic CCFs in FFT convention, shape = (nr, nt).
    freqs
        Frequency samples in Hz.
    c0
        True fundamental-mode phase velocity curve in km/s.
    c1
        True first-higher-mode phase velocity curve in km/s.
    """
    rng = np.random.default_rng(config.seed)
    r = _as_1d_float_array(r, "r")

    freqs = np.fft.rfftfreq(config.nt, config.dt)

    # True dispersion curves.
    c0 = 0.95 + 0.55 * np.exp(-freqs / 1.40)
    c1 = 1.55 + 0.70 * np.exp(-freqs / 1.60)

    # Modal amplitude spectra.
    a0 = config.a0_amp * np.exp(
        -0.5 * ((freqs - config.a0_center_hz) / config.a0_width_hz) ** 2
    )
    a1 = config.a1_amp * np.exp(
        -0.5 * ((freqs - config.a1_center_hz) / config.a1_width_hz) ** 2
    )

    low_taper = 1.0 - np.exp(-(freqs / config.low_taper_hz) ** 4)
    high_taper = np.exp(-(freqs / config.high_taper_hz) ** 8)

    a0 *= low_taper * high_taper
    a1 *= low_taper * high_taper

    # Build frequency-domain CCFs.
    k0 = 2.0 * np.pi * freqs[None, :] / c0[None, :]
    k1 = 2.0 * np.pi * freqs[None, :] / c1[None, :]

    spectrum = (
        a0[None, :] * j0(k0 * r[:, None])
        + a1[None, :] * j0(k1 * r[:, None])
    )
    spectrum = (
        a0[None, :] * np.exp(-1j * k0 * r[:, None])
        + a1[None, :] * np.exp(-1j * k1 * r[:, None])
    )

    if config.add_distance_decay:
        spectrum *= 1.0 / np.sqrt(r[:, None] + 1.0)

    # Transform to time domain.
    cc_time = np.fft.irfft(spectrum, n=config.nt, axis=1)
    cc_time += cc_time[:, ::-1]  # Make the CCF symmetric in time.
    # Add trace-wise RMS-scaled Gaussian noise.
    if config.noise_level > 0:
        rms = np.sqrt(np.mean(cc_time**2, axis=1, keepdims=True))
        cc_time += config.noise_level * rms * rng.standard_normal(cc_time.shape)

    # Normalize traces.
    if config.normalize_each_trace:
        cc_time /= np.max(np.abs(cc_time), axis=1, keepdims=True) + 1e-12
    else:
        cc_time /= np.max(np.abs(cc_time)) + 1e-12

    # Apply lag-domain taper with zero lag at the center.
    if config.apply_time_taper:
        cc_shift = np.fft.fftshift(cc_time, axes=1)
        cc_shift *= np.hanning(config.nt)[None, :]
        cc_time = np.fft.ifftshift(cc_shift, axes=1)

    return cc_time, freqs, c0, c1


# =============================================================================
# Plotting
# =============================================================================


def plot_waveform_and_spectra(
    cc_time: np.ndarray,
    r: np.ndarray,
    dt: float,
    c_grid: np.ndarray,
    f_true: np.ndarray,
    c0_true: np.ndarray,
    c1_true: np.ndarray,
    f_ps: np.ndarray,
    ps_img: np.ndarray,
    f_right: np.ndarray,
    right_img: np.ndarray,
    right_title: str,
    imaging_config: ImagingConfig,
    plot_config: PlotConfig,
) -> None:
    """Plot CCF waveforms, phase-shift spectrum, and F-J/MFJ spectrum."""
    nt = cc_time.shape[1]

    lag = (np.arange(nt) - nt // 2) * dt
    cc_show = np.fft.fftshift(cc_time, axes=1)

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.5],
        hspace=0.35,
        wspace=0.25,
    )

    # Waveform section.
    ax0 = fig.add_subplot(gs[0, :])
    show_indices = np.linspace(
        0,
        r.size - 1,
        min(plot_config.waveform_trace_count, r.size),
        dtype=int,
    )

    for idx in show_indices:
        trace = cc_show[idx]
        trace = trace / (np.max(np.abs(trace)) + 1e-12)
        ax0.plot(
            lag,
            r[idx] + plot_config.waveform_scale * trace,
            color="k",
            linewidth=0.7,
        )

    ax0.set_xlim(
        -plot_config.waveform_time_window_s,
        plot_config.waveform_time_window_s,
    )
    ax0.set_ylim(r.min() * 0.5, r.max() * 1.1)
    ax0.set_xlabel("Lag time (s)")
    ax0.set_ylabel("Interstation distance (km)")
    ax0.set_title("Synthetic ambient-noise cross-correlations")

    # Phase-shift spectrum.
    ax1 = fig.add_subplot(gs[1, 0])
    mesh1 = ax1.pcolormesh(
        f_ps,
        c_grid,
        ps_img,
        shading="auto",
        cmap=plot_config.cmap,
        vmin=0.0,
        vmax=1.0,
    )

    ax1.plot(f_true, c0_true, color="w", linestyle="--", linewidth=1.3)
    ax1.plot(f_true, c1_true, color="w", linestyle=":", linewidth=1.5)
    ax1.set_xlim(imaging_config.fmin, imaging_config.fmax)
    ax1.set_ylim(c_grid.min(), c_grid.max())
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Phase velocity (km/s)")
    ax1.set_title("Phase-shift dispersion spectrum")

    cbar1 = fig.colorbar(mesh1, ax=ax1, pad=0.02)
    cbar1.set_label("Normalized energy")

    # F-J or MFJ spectrum.
    ax2 = fig.add_subplot(gs[1, 1])
    mesh2 = ax2.pcolormesh(
        f_right,
        c_grid,
        right_img,
        shading="auto",
        cmap=plot_config.cmap,
        vmin=0.0,
        vmax=1.0,
    )

    ax2.plot(f_true, c0_true, color="w", linestyle="--", linewidth=1.3)
    ax2.plot(f_true, c1_true, color="w", linestyle=":", linewidth=1.5)
    ax2.set_xlim(imaging_config.fmin, imaging_config.fmax)
    ax2.set_ylim(c_grid.min(), c_grid.max())
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Phase velocity (km/s)")
    ax2.set_title(right_title)

    cbar2 = fig.colorbar(mesh2, ax=ax2, pad=0.02)
    cbar2.set_label("Normalized energy")

    fig.savefig(plot_config.output_path, dpi=plot_config.dpi, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main workflow
# =============================================================================


def run_demo(
    synthetic_config: SyntheticConfig | None = None,
    imaging_config: ImagingConfig | None = None,
    plot_config: PlotConfig | None = None,
) -> None:
    """Run the full synthetic CCF and dispersion imaging demo."""
    synthetic_config = synthetic_config or SyntheticConfig()
    imaging_config = imaging_config or ImagingConfig()
    plot_config = plot_config or PlotConfig()

    r = np.linspace(
        synthetic_config.r_min,
        synthetic_config.r_max,
        synthetic_config.nr,
    )
    c_grid = np.linspace(
        imaging_config.c_min,
        imaging_config.c_max,
        imaging_config.nc,
    )

    cc_time, f_true, c0_true, c1_true = build_synthetic_ccf(
        r,
        config=synthetic_config,
    )

    f_ps, ps_img = phase_shift_transform(
        cc_time,
        r,
        synthetic_config.dt,
        c_grid,
        fmin=imaging_config.fmin,
        fmax=imaging_config.fmax,
        positive_branch=imaging_config.ps_positive_branch,
        phase_only=imaging_config.ps_phase_only,
        taper_frac=imaging_config.ps_taper_frac,
    )

    if imaging_config.use_mfj_for_right_panel:
        f_right, right_img = mfj_transform(
            cc_time,
            r,
            synthetic_config.dt,
            c_grid,
            fmin=imaging_config.fmin,
            fmax=imaging_config.fmax,
            use_positive_branch=imaging_config.mfj_use_positive_branch,
            normalize_each_frequency=imaging_config.mfj_normalize_each_frequency,
            use_real_part=imaging_config.mfj_use_real_part,
            taper_frac=imaging_config.mfj_taper_frac,
        )
        right_title = "Modified F-J dispersion spectrum"
    else:
        f_right, right_img = fj_transform(
            cc_time,
            r,
            synthetic_config.dt,
            c_grid,
            fmin=imaging_config.fmin,
            fmax=imaging_config.fmax,
            use_real_spectrum=imaging_config.fj_use_real_spectrum,
            positive_only=imaging_config.fj_positive_only,
            normalize_each_frequency=imaging_config.fj_normalize_each_frequency,
        )
        right_title = "F-J dispersion spectrum"

    print("Synthetic dispersion test")
    print(
        f"  offsets: {r.min():.2f}--{r.max():.2f} km, "
        f"nr={synthetic_config.nr}"
    )
    print(
        f"  frequency band: {imaging_config.fmin:.2f}--"
        f"{imaging_config.fmax:.2f} Hz"
    )
    print(f"  output figure: {plot_config.output_path}")

    plot_waveform_and_spectra(
        cc_time=cc_time,
        r=r,
        dt=synthetic_config.dt,
        c_grid=c_grid,
        f_true=f_true,
        c0_true=c0_true,
        c1_true=c1_true,
        f_ps=f_ps,
        ps_img=ps_img,
        f_right=f_right,
        right_img=right_img,
        right_title=right_title,
        imaging_config=imaging_config,
        plot_config=plot_config,
    )


def main() -> None:
    """Entry point."""
    synthetic_config = SyntheticConfig(
        # 这里可以集中修改合成数据参数。
        nr=50,
        r_min=0.1,
        r_max=5.0,
        dt=0.01,
        nt=4096,
        noise_level=0.1,
        a0_amp=1.00,
        a1_amp=0.50,
    )

    imaging_config = ImagingConfig(
        # 这里可以集中修改频率范围和速度扫描范围。
        c_min=0.6,
        c_max=2.6,
        nc=500,
        fmin=0.5,
        fmax=5.0,
        #use_mfj_for_right_panel=True,
    )

    plot_config = PlotConfig(
        output_path=Path("phase_shift_vs_fj_spectrum_clean.png"),
        dpi=220,
    )

    run_demo(
        synthetic_config=synthetic_config,
        imaging_config=imaging_config,
        plot_config=plot_config,
    )


if __name__ == "__main__":
    main()
