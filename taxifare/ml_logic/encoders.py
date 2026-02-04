import math
import numpy as np
import pandas as pd
import pygeohash as gh


def transform_time_features(X: pd.DataFrame) -> np.ndarray:
    # Convert pickup_datetime to timezone-aware NYC time
    pickup_dt = X["pickup_datetime"].dt.tz_convert("America/New_York")

    dow = pickup_dt.dt.weekday
    hour = pickup_dt.dt.hour
    month = pickup_dt.dt.month

    hour_sin = np.sin(2 * math.pi * hour / 24)
    hour_cos = np.cos(2 * math.pi * hour / 24)

    # Days since a fixed origin (same idea as notebook)
    origin = pd.Timestamp("2009-01-01T00:00:00", tz="UTC")
    delta_days = (X["pickup_datetime"] - origin) / pd.Timedelta(1, "D")

    return np.stack([hour_sin, hour_cos, dow, month, delta_days], axis=1)


def transform_lonlat_features(X: pd.DataFrame) -> pd.DataFrame:
    earth_radius = 6371  # km

    lat_1 = np.radians(X["pickup_latitude"])
    lon_1 = np.radians(X["pickup_longitude"])
    lat_2 = np.radians(X["dropoff_latitude"])
    lon_2 = np.radians(X["dropoff_longitude"])

    dlat = lat_2 - lat_1
    dlon = lon_2 - lon_1

    # Manhattan distance approximation on sphere
    manhattan_km = (np.abs(dlat) + np.abs(dlon)) * earth_radius

    # Haversine distance
    a = (np.sin(dlat / 2) ** 2 +
         np.cos(lat_1) * np.cos(lat_2) * np.sin(dlon / 2) ** 2)
    haversine_km = 2 * earth_radius * np.arcsin(np.sqrt(a))

    return pd.DataFrame({"haversine": haversine_km, "manhattan": manhattan_km})

def compute_geohash(X: pd.DataFrame, precision: int = 5) -> np.ndarray:
    """
    Add a geohash (ex: "dr5rx") of len "precision" = 5 by default
    corresponding to each (lon, lat) tuple, for pick-up, and drop-off
    """
    pickup_geohash = X.apply(
        lambda row: gh.encode(row["pickup_latitude"], row["pickup_longitude"], precision=precision),
        axis=1
    )
    dropoff_geohash = X.apply(
        lambda row: gh.encode(row["dropoff_latitude"], row["dropoff_longitude"], precision=precision),
        axis=1
    )

    return np.stack([pickup_geohash, dropoff_geohash], axis=1)
