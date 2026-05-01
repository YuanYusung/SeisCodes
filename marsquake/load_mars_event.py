"""
This module provides functions to load seismic data for specific Marsevents, including
metadata retrieval, data trimming, and rotation. It also includes an example usage that
plots the waveforms with P/S arrival annotations and saves the figure.

The Martian Seismic Dataset of InSight Data with basic Denoising processing
can be download from the Earth System Data Cloud (ESDC) or Science Data Bank (SciDB):
http://www.esdc.ac.cn/article/170 or https://www.scidb.cn/s/IVjuUz

Reference: Wang, S., Ning, S., Yao, Z., Li, J., Xiao, W., Yan, T., & Xu, F. (2025). 
Basic processing of the InSight seismic data from Mars for further seismological 
research. Earthquake Science, 38(5), 450～460. https://doi.org/10.1016/j.eqs.2025.06.006

Created by Yusong Yuan, CUG, 2026-05-01
"""

import os
import json
import matplotlib.pyplot as plt
from obspy import read, UTCDateTime
from mars_event_info import get_mars_event_info
import matplotlib.pyplot as plt


INSIGHT_data_dir = "/Users/yusong/workspace/Insight/data/Data_Step5_Rotate/"

event_info_dir = "./data/event_info/"
stream_dir = "./data/event_stream/"
# Ensure output directories exist
os.makedirs(event_info_dir, exist_ok=True)
os.makedirs(stream_dir, exist_ok=True)

# helper functions for time conversion between UTC and LMST
def utc2lmst(utc_time):
    """
    Convert a UTC time to InSight LMST (Local Mean Solar Time).
    Returns a dict with keys: sol, hour, minute, second.
    """
    # InSight Sol 0 reference UTC
    sol0_start = UTCDateTime("2018-11-26T05:10:50.336037Z")
    # A mean solar day length in Earth seconds, derived from two known Sol start times:
    sol1_start = UTCDateTime("2018-11-27T05:50:25.580014Z")
    sol2_start = UTCDateTime("2018-11-28T06:30:00.823990Z")
    sec_m = sol2_start - sol1_start - 0.000005  # Mars day in Earth seconds

    # Earth seconds since Sol 0
    dur_e = utc_time - sol0_start

    # Convert Earth seconds to Mars seconds (scaled by day length ratio)
    mars_seconds_total = dur_e * 86400.0 / sec_m

    # Extract sol number and time of day in Mars seconds
    sol = int(mars_seconds_total // 86400)
    tod = mars_seconds_total % 86400
    hour = int(tod // 3600)
    minute = int((tod % 3600) // 60)
    second = tod % 60

    return {"sol": sol, "hour": hour, "minute": minute, "second": second}

def sol_start_utc(sol):
    """
    Given a sol number, return the UTC time of the start of that sol.
    """
    sol0_start = UTCDateTime("2018-11-26T05:10:50.336037Z")
    sol1_start = UTCDateTime("2018-11-27T05:50:25.580014Z")
    sol2_start = UTCDateTime("2018-11-28T06:30:00.823990Z")
    sec_m = sol2_start - sol1_start - 0.000005  # Mars day in Earth seconds
    return sol0_start + sol * sec_m

def get_event_info(event_name):
    """
    Retrieve event metadata (origin time, P/S arrival times, distance, back-azimuth) for a given Marsevent.
    This function first checks if the event info is cached in a JSON file. If not, it calls
    get_mars_event_info() to retrieve the info from the event catalog, then caches it for future use.
    """
    info_file = os.path.join(event_info_dir, f"{event_name}_info.json")
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            event_info = json.load(f)
        event_info["origin_time"] = UTCDateTime(event_info["origin_time"])
        event_info["p_arrival_time"] = UTCDateTime(event_info["p_arrival_time"])
        event_info["s_arrival_time"] = UTCDateTime(event_info["s_arrival_time"])
        print(f"Loaded {event_name} info from cache.")
    else:
        # Retrieve metadata from the event catalogue and cache it
        event_info = get_mars_event_info(event_name, include_objects=False)
        # Convert UTCDateTime objects to ISO strings for JSON serialization
        cache_info = event_info.copy()
        cache_info["origin_time"] = event_info["origin_time"].isoformat()
        cache_info["p_arrival_time"] = event_info["p_arrival_time"].isoformat()
        cache_info["s_arrival_time"] = event_info["s_arrival_time"].isoformat()
        with open(info_file, 'w') as f:
            json.dump(cache_info, f, indent=2)
        print(f"Cached {event_name} info to {info_file}")

    # Extract event parameters
    origin_time = event_info["origin_time"]
    p_arrival_time = event_info["p_arrival_time"]
    s_arrival_time = event_info["s_arrival_time"]
    distance_km = event_info["distance_km"]
    back_azimuth_deg = event_info["back_azimuth_deg"]

    return origin_time, p_arrival_time, s_arrival_time, distance_km, back_azimuth_deg

def load_event(event_name):
    """
    Load the seismic stream for a given Marsevent.

    If a pre-processed (trimmed + rotated) MiniSEED file exists, it is loaded
    directly. Otherwise the raw daily files are loaded, trimmed between
    start_time and end_time, merged if the window spans two sols, rotated to
    the radial/transverse system, and saved for future reuse.
    """

    # Define the cut window
    origin_time, p_arrival_time, s_arrival_time, distance_km, back_azimuth_deg = get_event_info(event_name)
    start_time = origin_time
    ref_vel_km_s = 2.0
    end_time = origin_time + distance_km / ref_vel_km_s + 200

    trimmed_file = os.path.join(stream_dir, f"{event_name}_trimmed.mseed")
    if os.path.exists(trimmed_file):
        print(f"Trimmed data for event {event_name} already exists. Loading from file...")
        st = read(trimmed_file)
    else:
        start_sol = utc2lmst(start_time)["sol"]
        end_sol = utc2lmst(end_time)["sol"]

        if start_sol == end_sol:
            # Window lies entirely within a single sol
            st = read(f"{INSIGHT_data_dir}LMST_{start_sol:04d}.mseed")
            st.trim(starttime=start_time, endtime=end_time)
        else:
            # Window crosses a sol boundary – read both sols and merge
            print(f"Event {event_name} spans from Sol {start_sol} to Sol {end_sol}, loading and merging data...")
            st1 = read(f"{INSIGHT_data_dir}LMST_{start_sol:04d}.mseed")
            end_of_first = sol_start_utc(start_sol + 1) - 1e-6
            st1.trim(starttime=start_time, endtime=end_of_first)

            st2 = read(f"{INSIGHT_data_dir}LMST_{end_sol:04d}.mseed")
            start_of_second = sol_start_utc(end_sol)
            st2.trim(starttime=start_of_second, endtime=end_time)
            st = st1 + st2    
            st.merge(fill_value='interpolate',method=1) 

        # Rotate horizontal components from NE to RT
        st.rotate(method="NE->RT", back_azimuth=back_azimuth_deg)
        # Save processed stream for faster future loads
        st.write(trimmed_file, format="MSEED")
        print(f"Saved trimmed and rotated stream to {trimmed_file}")

    return st, p_arrival_time, s_arrival_time, distance_km, back_azimuth_deg


if __name__ == "__main__":
    """
    Example usage: Load a specific Marsevent, plot the waveforms with P/S arrival annotations,
    and save the figure.
    """

    event_name = "S1222a"
    st, p_arr, s_arr, dist_km, baz_deg = load_event(event_name)
    fig = st.plot(size=(900, 500), title=event_name, show=False)#outfile=f"./Figures/{event_name}_waveforms.png"
    if fig is None:  
        fig = plt.gcf()

    fig.suptitle(f"Mars Event: {event_name}", fontsize=14, y=0.92)
    axes = fig.get_axes()
    p_num = p_arr.matplotlib_date
    s_num = s_arr.matplotlib_date

    for ax in axes:
        ax.axvline(x=p_num, color='red', linestyle='-', linewidth=1.5, label='P arrival')
        ax.axvline(x=s_num, color='blue', linestyle='-', linewidth=1.5, label='S arrival')

    info_text = f"Distance: {dist_km:.1f} km\nBack‑azimuth: {baz_deg:.1f}°"

    axes[0].text(0.98, 0.92, info_text, transform=axes[0].transAxes,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=10)

    axes[-1].legend(loc='upper right',fontsize=10)
    
    fig.savefig(f"./Figures/{event_name}_waveforms.png", dpi=150)
    print(f"Saved annotated figure to ./Figures/{event_name}_waveforms.png")