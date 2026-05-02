import numpy as np
import  matplotlib.pyplot as plt

def plot_particle_motion(st, t_start,t_end, freqmin=0.05, freqmax=0.1, normalize=True, show_time_color=True):
    
    st.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)

    tr_x = st.select(channel="*R")[0]
    tr_y = st.select(channel="*Z")[0]
    ref_time = min(tr_x.stats.starttime, tr_y.stats.starttime)
    abs_start = ref_time + t_start
    abs_end = ref_time + t_end

    tx = tr_x.copy().trim(abs_start, abs_end, pad=False)
    ty = tr_y.copy().trim(abs_start, abs_end, pad=False)

    if len(tx.data) == 0 or len(ty.data) == 0:
        raise ValueError("Selected time window contains no data.")

    common_start = max(tx.stats.starttime, ty.stats.starttime)
    common_end = min(tx.stats.endtime, ty.stats.endtime)

    tx.trim(common_start, common_end, pad=False)
    ty.trim(common_start, common_end, pad=False)

    n = min(len(tx.data), len(ty.data))
    x = tx.data[:n].astype(float)
    y = ty.data[:n].astype(float)

    if n < 2:
        raise ValueError("Not enough samples for particle motion plot.")

    x = x - np.mean(x)
    y = y - np.mean(y)

    if normalize:
        max_amp = max(np.max(np.abs(x)), np.max(np.abs(y)))
        if max_amp > 0:
            x = x / max_amp
            y = y / max_amp

    dt = tx.stats.delta
    rel_t = np.arange(n) * dt + (common_start - ref_time)

    fig, ax = plt.subplots(figsize=(6,4.5))

    if show_time_color:
        sc = ax.scatter(
            x,
            y,
            c=rel_t,
            s=8,
            cmap="viridis",
        )
        ax.plot(x, y, color="gray", linewidth=0.6, alpha=0.5)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Time after origin (s)")
    else:
        ax.plot(x, y, color="black", linewidth=1.0)

    ax.scatter(x[0], y[0], color="red", s=40, label="Start", zorder=3)
    ax.scatter(x[-1], y[-1], color="blue", s=40, label="End", zorder=3)

    x_label = tr_x.stats.channel[-1]
    y_label = tr_y.stats.channel[-1]


    if normalize:
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_xlabel(f"{x_label} (normalized)", fontsize=12)
        ax.set_ylabel(f"{y_label} (normalized)", fontsize=12)
    else:
        max_amp = max(np.max(np.abs(x)), np.max(np.abs(y)))
        ax.set_xlim(-max_amp*1.1, max_amp*1.1)
        ax.set_ylim(-max_amp*1.1, max_amp*1.1)
        ax.set_xlabel(f"{x_label} (m/s)", fontsize=12)
        ax.set_ylabel(f"{y_label} (m/s)", fontsize=12)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    plt.tight_layout()
    return fig, ax

def plot_waveforms_with_arrivals(st, p_arr=None, s_arr=None, dist_km=None, baz_deg=None, freqmin=0.05, freqmax=1., event_name = "S1222a"):
    
    
    st.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)

    fig = st.plot(size=(900, 500), title=event_name, show=False, type="relative")
    if fig is None:  
        fig = plt.gcf()

    fig.suptitle(f"Mars Event: {event_name}.  {freqmin}-{freqmax} Hz bandpass filtered", fontsize=14, y=0.92)
    axes = fig.get_axes()
    start_time = st[0].stats.starttime
    p_rel = p_arr - start_time if p_arr is not None else None
    s_rel = s_arr - start_time if s_arr is not None else None

    for ax in axes:
        if p_rel is not None:
            ax.axvline(x=p_rel, color='red', linestyle='-', linewidth=1.5, label='P arrival')

        if s_rel is not None:
            ax.axvline(x=s_rel, color='blue', linestyle='-', linewidth=1.5, label='S arrival')

        ax.set_ylabel("Velocity (m/s)", fontsize=10)

    if p_rel is not None or s_rel is not None:
        axes[-1].legend(loc='upper right', fontsize=10)

    info_text = f"Distance: {dist_km:.1f} km\nBack‑azimuth: {baz_deg:.1f}°"

    axes[0].text(0.98, 0.92, info_text, transform=axes[0].transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)

    axes[-1].set_xlabel("Time after origin (s)", fontsize=10)

    return fig, axes

from scipy.signal import spectrogram
def plot_spectrogram(st,channel,tP,tS,freqmin=0.01,freqmax=0.1,event_name=None):
    tr = st.select(channel=channel)[0].copy()
    tr.filter("bandpass", freqmin=freqmin, freqmax=freqmax, corners=4, zerophase=True)
    origin_time = tr.stats.starttime
    fs = tr.stats.sampling_rate 
    nperseg = int(200 * fs)
    noverlap = int(0.9 * nperseg)

    if nperseg > len(tr.data):
        nperseg = len(tr.data) // 4
        noverlap = int(0.9 * nperseg)

    f, t_spec, Sxx = spectrogram(
        tr.data,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="density",
        mode="psd"
    )

    Sxx_db = 10 * np.log10(Sxx + 1e-20)

    freq_min_plot = freqmin
    freq_max_plot = freqmax

    freq_mask = (f >= freq_min_plot) & (f <= freq_max_plot)
    f_plot = f[freq_mask]
    Sxx_db_plot = Sxx_db[freq_mask, :]

    vmin = np.percentile(Sxx_db_plot, 5)
    vmax = np.percentile(Sxx_db_plot, 99)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(9, 6),
        dpi=150,
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1.3]}
    )

    ax1 = axes[0]
    ax2 = axes[1]

    times = tr.times(reftime=origin_time)
    ax1.plot(times, tr.data, linewidth=0.8, label=tr.id, color="black")

    ax1.set_ylabel("Velocity (m/s)", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax1.set_title(f"{event_name} - {tr.stats.channel} (Bandpass {freqmin}-{freqmax} Hz)", fontsize=12)

    ymax = tr.data.max()
    ymin = tr.data.min()
    yrange = ymax - ymin
    ax1.set_ylim(ymin - 0.05 * yrange, ymax + 0.15 * yrange)

    # ----------------------------
    # 下图：时频谱
    # ----------------------------
    im = ax2.pcolormesh(
        t_spec,
        1./f_plot,
        Sxx_db_plot,
        shading="auto",
        vmin=vmin,
        vmax=vmax
    )

    ax2.set_xlabel("Time relative to Origin Time (s)", fontsize=12)
    ax2.set_ylabel("Period (s)", fontsize=12)
    ax2.set_ylim(1./freq_min_plot, 1./freq_max_plot)
    # ax2.set_ylim(30,10)  # 反频率坐标
    ax2.set_xlim(0+100, t_spec[-1]-100)
    ax2.grid(True, linestyle=":", alpha=0.3)

    # 标注 P、S、R
    for t_arr, name in [(tP, "P"), (tS, "S")]:
        if t_arr is None:
            continue
        t_rel = t_arr - origin_time
        for ax in axes:
            ax.axvline(t_rel, linestyle="--", linewidth=1.3, color="red")
        axes[0].text(t_rel + 20,tr.data.max() - 0.05 * (tr.data.max() - tr.data.min()),name,
            ha="center",
            va="bottom",
            fontsize=11,
            color="red"
        )


    return fig, axes

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftfreq, next_fast_len
from scipy.signal import hilbert
from obspy import Trace


def optimized_fft(data):
    nopt = next_fast_len(len(data) * 2)
    return fft(data, n=nopt)

def gaussian_bandpass_spectrum(sf, periods, delta, alpha=50):
    """
    Apply a bank of Gaussian bandpass filters in the frequency domain.
    Returns shape (nt, nperiods) real-valued filtered wavefields.
    """
    ns = sf.shape[0]
    domega = 2 * np.pi / (ns * delta)
    omega = domega * np.arange(ns)
    n_f = len(periods)
    filtered = np.zeros((ns, n_f), dtype=np.complex128)

    for i, per in enumerate(periods):
        omega0 = 2 * np.pi / per
        gauss = np.exp(-((omega - omega0) / omega0) ** 2 * alpha)
        spec = gauss * sf
        spec[ns // 2 + 1 :] = 0.0
        spec[0] /= 2.0
        spec[ns // 2] = np.real(spec[ns // 2])
        filtered[:, i] = ifft(spec)[: ns]

    return filtered

def narrowband_wavefield(wf,taxis, periods, alpha=50):
    """
    Given an ObsPy Trace and target periods, return:
    - taxis: time axis
    - freq: 1 / periods (Hz)
    - filtered_real: real part of narrow-band filtered wavefield
    - envelope: Hilbert envelope of the filtered wavefield
    """
    data = wf
    delta = taxis[1] - taxis[0]
    sf = optimized_fft(data)
    filtered_complex = gaussian_bandpass_spectrum(sf, periods, delta, alpha=alpha)
    filtered_real = np.real(filtered_complex[: len(data), :])
    envelope = np.abs(hilbert(filtered_real, axis=0))
    return periods, filtered_real, envelope

def extract_longest_continuous_branch(peak_periods, peak_times,
                                      monotonic_tol=0.0,
                                      jump_factor=3.0,
                                      min_points=4):
    """
    从拾取点中自动提取：
    1) 随周期增大，走时不变慢（time不增大）的点；
    2) 其中最长、最连续的一段。

    Parameters
    ----------
    peak_periods : 1D array
        每个拾取点对应的周期
    peak_times : 1D array
        每个拾取点对应的走时
    monotonic_tol : float
        允许的轻微反向波动容差（秒）。
        若严格要求“周期变大，走时绝不能变慢”，设为 0。
        若想允许一点点噪声波动，可设为 0.1~0.3。
    jump_factor : float
        连续性阈值的倍数。越小越严格。
    min_points : int
        至少保留多少个点才算有效连续段

    Returns
    -------
    seg_periods : 1D array
        提取出的连续段周期
    seg_times : 1D array
        提取出的连续段走时
    seg_idx : 1D array
        对应于原排序后数组的索引
    """
    peak_periods = np.asarray(peak_periods, dtype=float)
    peak_times = np.asarray(peak_times, dtype=float)

    # 去掉 NaN
    valid = np.isfinite(peak_periods) & np.isfinite(peak_times)
    peak_periods = peak_periods[valid]
    peak_times = peak_times[valid]

    if len(peak_periods) == 0:
        return np.array([]), np.array([]), np.array([], dtype=int)

    # 按周期从小到大排序
    order = np.argsort(peak_periods)
    peak_periods = peak_periods[order]
    peak_times = peak_times[order]

    if len(peak_periods) < 2:
        return peak_periods, peak_times, np.arange(len(peak_periods))

    dt = np.diff(peak_times)

    # 自动估计“连续性”允许的时间跳变阈值
    base_jump = np.nanmedian(np.abs(dt))
    if (not np.isfinite(base_jump)) or (base_jump == 0):
        base_jump = 0.5
    max_jump = jump_factor * base_jump

    # 条件1：周期变大时，走时不能明显变慢
    cond_monotonic = dt <= monotonic_tol

    # 条件2：相邻点之间不能跳得太厉害（保证连续）
    cond_continuous = np.abs(dt) <= max_jump

    # 相邻两点构成“有效连接”
    good_pair = cond_monotonic & cond_continuous

    # 在 good_pair 中找最长连续 True 段
    best_start = 0
    best_len = 1
    cur_start = 0

    for i, ok in enumerate(good_pair):
        if not ok:
            cur_len = i - cur_start + 1   # 点数 = pair数 + 1
            if cur_len > best_len:
                best_start = cur_start
                best_len = cur_len
            cur_start = i + 1

    # 处理最后一段
    cur_len = len(good_pair) - cur_start + 1
    if cur_len > best_len:
        best_start = cur_start
        best_len = cur_len

    if best_len < min_points:
        return np.array([]), np.array([]), np.array([], dtype=int)

    seg_idx = np.arange(best_start, best_start + best_len)
    return peak_periods[seg_idx], peak_times[seg_idx], seg_idx

def plot_nbp_wfs(tr,periods,dist_ij,vmin,vmax,alpha=20):
    taxis = tr.times(reftime=tr.stats.starttime)

    # Time window selection based on expected arrival times for a given velocity range
    tmin = dist_ij / vmax; tmax = dist_ij / vmin
    mask = (taxis >= tmin) & (taxis <= tmax)
    tr_tmp = Trace(data=tr.data[mask])
    tr_tmp.taper(max_percentage=0.05)
    wf = np.zeros_like(tr.data)
    wf[mask] = tr_tmp.data

    periods, filtered_real, envelope = narrowband_wavefield(wf, taxis, periods, alpha)

    pers = periods
    # normalize for better contrast
    filtered_real /= np.nanmax(np.abs(filtered_real), axis=0, keepdims=True)
    envelope /= np.nanmax(envelope, axis=0, keepdims=True)

    pers_grid, time_grid = np.meshgrid(pers, taxis)

    fig, axes = plt.subplots(1, 2, figsize=(12,8), sharey=True)

    im0 = axes[0].pcolormesh(pers_grid,time_grid,filtered_real,shading="auto",cmap="seismic",vmin=-1,vmax=1,)
    axes[0].set_title(f"Narrow-band filtered wavefield of {tr.stats.channel}",fontsize=14)
    axes[0].set_xlabel("Period (s)",fontsize=14)
    axes[0].set_ylabel("Time (s)",fontsize=14)
            
    im1 = axes[1].pcolormesh(pers_grid,time_grid,envelope,shading="auto",cmap="jet",vmin=0,vmax=1.)
    axes[1].set_title(f"Envelope of filtered wavefield of {tr.stats.channel}",fontsize=14)

    axes[1].set_xlabel("Period (s)",fontsize=14)
    #axes[1].set_ylabel("Time (s)")


    peak_idx = np.nanargmax(envelope, axis=0)
    peak_times = taxis[peak_idx]
    peak_periods = periods

    axes[1].scatter(
        peak_periods, peak_times,
        s=20, color="white", zorder=5
    )

    # 自动提取最长、最连续、且“随周期增大走时不变慢”的一段
    seg_periods, seg_times, seg_idx = extract_longest_continuous_branch(
        peak_periods,
        peak_times,
        monotonic_tol=0.0,   # 严格要求：周期变大，走时不能变大
        jump_factor=3.0,     # 连续性要求，越小越严格
        min_points=5         # 至少5个点才认为有效
    )

    # 高亮显示提取出的连续段
    if len(seg_periods) > 0:
        axes[1].plot(
            seg_periods, seg_times,
            color="black", linewidth=2.0, zorder=6, label="Selected"
        )
        axes[1].scatter(
            seg_periods, seg_times,
            s=28, color="black", linewidths=0.5, zorder=7
        )

    axes[1].legend()


    for ax in axes:
        ax.set_ylim(tmin, tmax)
        ax_twin = ax.twinx(); ax_twin.tick_params(labelsize=16,colors='r')
        ax_twin.set_ylim(ax.get_ylim())
        vshow = np.arange(vmin, vmax+0.01, 0.1)
        yshow = dist_ij/vshow


        ax_twin.set_yticks(yshow); ax_twin.set_yticklabels(vshow.round(1))
        ax.tick_params(labelsize=14)
        if ax == axes[-1]:
            ax_twin.set_ylabel("Group Velocity (km/s)", fontsize=14, color='r')

    return fig, axes
