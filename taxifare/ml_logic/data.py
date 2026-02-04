import pandas as pd

from google.cloud import bigquery
from colorama import Fore, Style
from pathlib import Path

from taxifare.params import *

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw data by
    - assigning correct dtypes to each column
    - removing buggy or irrelevant transactions
    """
    # Compress raw_data by setting types to DTYPES_RAW
    df = df.astype(DTYPES_RAW)

    # Remove buggy transactions
    df = df.drop_duplicates()
    df = df.dropna()

    df = df[df["fare_amount"] > 0]
    df = df[df["passenger_count"] > 0]

    df = df[df["fare_amount"] < 400]
    df = df[df["passenger_count"] < 8]

    # Remove geographically irrelevant transactions (rows)
    lon_min, lon_max, lat_min, lat_max = BOUNDING_BOXES

    df = df[df["pickup_longitude"].between(lon_min, lon_max)]
    df = df[df["dropoff_longitude"].between(lon_min, lon_max)]
    df = df[df["pickup_latitude"].between(lat_min, lat_max)]
    df = df[df["dropoff_latitude"].between(lat_min, lat_max)]

    print("✅ data cleaned")

    return df
