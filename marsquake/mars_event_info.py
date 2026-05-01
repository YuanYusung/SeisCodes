"""
mars_event_utils.py

Utility functions for extracting basic marsquake event information from
an ObsPy QuakeML/XML catalog.

Main features
-------------
1. Find a marsquake event by event name, such as "S1222a".
2. Extract the preferred origin, P-wave pick, and S-wave pick.
3. Compute epicentral distance from the InSight lander.
4. Compute back azimuth from InSight to the event.
5. Return all results as a Python dictionary for use in other scripts.

Typical usage
-------------
from mars_event_utils import get_mars_event_info

info = get_mars_event_info("S1222a", print_result=True)

origin_time = info["origin_time"]
p_time = info["p_arrival_time"]
s_time = info["s_arrival_time"]
distance_deg = info["distance_deg"]
distance_km = info["distance_km"]
back_azimuth = info["back_azimuth_deg"]

Created by Yusong Yuan, CUG, 2026-05-01
"""

from obspy import read_events
import math

# Default event catalog path.
# This file can be downloaded from the Marsquake Service / MQS catalog.
DEFAULT_CATALOG_PATH = "./data/catalog/events_extended_multiorigin_v14_2023-01-01.xml"

# InSight lander coordinates
INSIGHT_LAT = 4.502384
INSIGHT_LON = 135.623447

# Mean radius of Mars, in kilometers
MARS_RADIUS_KM = 3389.5



def find_event_by_name(cat, name, desc_type="earthquake name"):
    """
    Find an event from an ObsPy catalog by its event name.

    Parameters
    ----------
    cat : obspy.core.event.Catalog
        ObsPy event catalog loaded by read_events().

    name : str
        Event name, for example "S1222a".

    desc_type : str, optional
        Type of event description used to identify the event.
        The default is "earthquake name".

    Returns
    -------
    ev : obspy.core.event.Event or None
        The matched ObsPy Event object. If no event is found, returns None.
    """

    for ev in cat:
        for desc in ev.event_descriptions:
            if desc.type == desc_type and desc.text == name:
                return ev

    return None

def get_preferred_origin_ps(ev):
    """
    Get the preferred origin and the first P and S picks associated with it.

    The function first tries to use ev.preferred_origin(). If no preferred
    origin is defined, it uses the first origin in ev.origins.

    Parameters
    ----------
    ev : obspy.core.event.Event
        ObsPy Event object.

    Returns
    -------
    origin : obspy.core.event.Origin or None
        Preferred origin, or the first available origin.

    p_pick : obspy.core.event.Pick or None
        First P-wave pick associated with the selected origin.

    s_pick : obspy.core.event.Pick or None
        First S-wave pick associated with the selected origin.
    """

    origin = ev.preferred_origin() or (ev.origins[0] if ev.origins else None)

    if origin is None:
        return None, None, None

    pick_map = {str(p.resource_id): p for p in ev.picks}

    p_pick = None
    s_pick = None

    for arr in origin.arrivals:
        pick = pick_map.get(str(arr.pick_id))

        if pick is None:
            continue

        phase = _get_phase_name(arr, pick)

        if phase == "P" and p_pick is None:
            p_pick = pick

        elif phase == "S" and s_pick is None:
            s_pick = pick

    return origin, p_pick, s_pick


def _get_phase_name(arrival, pick):
    """
    Get a normalized phase name from an Arrival or Pick object.

    Parameters
    ----------
    arrival : obspy.core.event.Arrival
        Arrival object associated with an origin.

    pick : obspy.core.event.Pick
        Pick object referenced by the arrival.

    Returns
    -------
    phase : str
        Upper-case phase name. If no phase name is available, returns "".
    """

    if arrival.phase is not None:
        return str(arrival.phase).strip().upper()

    if pick.phase_hint is not None:
        return str(pick.phase_hint).strip().upper()

    return ""

def mars_epicentral_distance(lat1,lon1,lat2,lon2,mars_radius_km=MARS_RADIUS_KM):
    """
    Compute great-circle distance between two points on Mars.

    Returns
    -------
    distance_deg : float
        Epicentral distance in degrees.
    distance_km : float
        Epicentral distance in kilometers.
    """

    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad)
        * math.cos(lat2_rad)
        * math.sin(dlon / 2) ** 2
    )

    central_angle_rad = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    distance_deg = math.degrees(central_angle_rad)
    distance_km = mars_radius_km * central_angle_rad

    return distance_deg, distance_km

def spherical_azimuth(lat1, lon1, lat2, lon2):
    """
    Compute the azimuth from point 1 to point 2.

    North is 0°, increasing clockwise, east is 90°.
    """

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)

    x = math.sin(dlon_rad) * math.cos(lat2_rad)

    y = (
        math.cos(lat1_rad) * math.sin(lat2_rad)
        - math.sin(lat1_rad)
        * math.cos(lat2_rad)
        * math.cos(dlon_rad)
    )

    az = math.degrees(math.atan2(x, y))
    az = (az + 360) % 360

    return az

def get_mars_event_info(
    event_name,
    catalog_path=DEFAULT_CATALOG_PATH,
    insight_lat=INSIGHT_LAT,
    insight_lon=INSIGHT_LON,
    mars_radius_km=MARS_RADIUS_KM,
    desc_type="earthquake name",
    include_objects=True,
    print_result=False
):
    """
    Read a catalog file and return basic information for one marsquake event.

    This is the main interface function for use in other scripts.

    Parameters
    ----------
    event_name : str
        Event name, for example "S1222a".

    catalog_path : str, optional
        Path to the QuakeML/XML catalog file.
        The default is DEFAULT_CATALOG_PATH.

    insight_lat, insight_lon : float, optional
        Latitude and longitude of the InSight lander.

    mars_radius_km : float, optional
        Mean radius of Mars in kilometers.

    desc_type : str, optional
        Event description type used for event-name matching.
        The default is "earthquake name".

    include_objects : bool, optional
        If True, the returned dictionary also includes raw ObsPy objects:
        event, origin, p_pick, and s_pick.

    print_result : bool, optional
        If True, print a formatted summary of the event information.

    Returns
    -------
    info : dict
        Dictionary containing event time, P/S arrival times, event location,
        epicentral distance, back azimuth, and optional ObsPy objects.

    Raises
    ------
    ValueError
        If the event is not found or has no valid origin/location.
    """

    cat = read_events(catalog_path)

    return get_mars_event_info_from_catalog(
        cat=cat,
        event_name=event_name,
        insight_lat=insight_lat,
        insight_lon=insight_lon,
        mars_radius_km=mars_radius_km,
        desc_type=desc_type,
        include_objects=include_objects,
        print_result=print_result
    )


def get_mars_event_info_from_catalog(
    cat,
    event_name,
    insight_lat=INSIGHT_LAT,
    insight_lon=INSIGHT_LON,
    mars_radius_km=MARS_RADIUS_KM,
    desc_type="earthquake name",
    include_objects=True,
    print_result=False
):
    """
    Return basic information for one event from an already loaded catalog.

    This function is useful when processing many events, because the catalog
    only needs to be read once.

    Parameters
    ----------
    cat : obspy.core.event.Catalog
        ObsPy event catalog.

    event_name : str
        Event name, for example "S1222a".

    insight_lat, insight_lon : float, optional
        Latitude and longitude of the InSight lander.

    mars_radius_km : float, optional
        Mean radius of Mars in kilometers.

    desc_type : str, optional
        Event description type used for event-name matching.

    include_objects : bool, optional
        If True, include raw ObsPy objects in the returned dictionary.

    print_result : bool, optional
        If True, print a formatted summary.

    Returns
    -------
    info : dict
        Event information dictionary.

    Raises
    ------
    ValueError
        If the event is not found or has no valid origin/location.
    """

    ev = find_event_by_name(
        cat=cat,
        name=event_name,
        desc_type=desc_type
    )

    if ev is None:
        raise ValueError(f"Event not found: {event_name}")

    origin, p_pick, s_pick = get_preferred_origin_ps(ev)

    if origin is None:
        raise ValueError(f"Event {event_name} has no available origin.")

    if origin.latitude is None or origin.longitude is None:
        raise ValueError(f"Event {event_name} has no latitude/longitude information.")

    event_lat = origin.latitude
    event_lon = origin.longitude

    distance_deg, distance_km = mars_epicentral_distance(
        insight_lat,
        insight_lon,
        event_lat,
        event_lon,
        mars_radius_km=mars_radius_km
    )

    # Back azimuth: direction from InSight to the event
    back_azimuth_deg = spherical_azimuth(
        insight_lat,
        insight_lon,
        event_lat,
        event_lon
    )

    # Azimuth: direction from the event to InSight
    azimuth_deg = spherical_azimuth(
        event_lat,
        event_lon,
        insight_lat,
        insight_lon
    )

    info = {
        "event_name": event_name,

        "origin_time": origin.time,
        "p_arrival_time": p_pick.time if p_pick else None,
        "s_arrival_time": s_pick.time if s_pick else None,

        "event_latitude": event_lat,
        "event_longitude": event_lon,

        "insight_latitude": insight_lat,
        "insight_longitude": insight_lon,

        "distance_deg": distance_deg,
        "distance_km": distance_km,

        "back_azimuth_deg": back_azimuth_deg,
        "azimuth_deg": azimuth_deg,
    }

    if include_objects:
        info.update(
            {
                "event": ev,
                "origin": origin,
                "p_pick": p_pick,
                "s_pick": s_pick,
            }
        )

    if print_result:
        print_mars_event_info(info)

    return info


# ============================================================
# Output helper
# ============================================================

def print_mars_event_info(info):
    """
    Print a formatted summary of a marsquake event information dictionary.

    Parameters
    ----------
    info : dict
        Dictionary returned by get_mars_event_info() or
        get_mars_event_info_from_catalog().
    """

    print("\n========================================")
    print(f"Event name: {info['event_name']}")
    print("========================================")

    print("\nTime information:")
    print("Origin time:", info["origin_time"])
    print("P arrival time:", info["p_arrival_time"])
    print("S arrival time:", info["s_arrival_time"])

    print("\nEvent location:")
    print("Latitude:", info["event_latitude"])
    print("Longitude:", info["event_longitude"])

    print("\nInSight location:")
    print("Latitude:", info["insight_latitude"])
    print("Longitude:", info["insight_longitude"])

    print("\nEpicentral distance:")
    print(f"{info['distance_deg']:.2f} degrees")
    print(f"{info['distance_km']:.1f} km")

    print("\nAzimuth information:")
    print(f"Back azimuth, InSight -> Event: {info['back_azimuth_deg']:.2f} degrees")
    print(f"Azimuth, Event -> InSight: {info['azimuth_deg']:.2f} degrees")

    print("\n========================================")


# Optional command-line test
if __name__ == "__main__":
    event_info = get_mars_event_info(
        event_name="S1222a",
        print_result=True
    )