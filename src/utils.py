import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path("/Users/rafalbuczynski/Git/bellinge")
DOWNLOADED = PROJECT_ROOT / "downloaded"
DATA_RAW = PROJECT_ROOT / "data/raw"
DATA_INTERIM = PROJECT_ROOT / "data/interim"
DATA_PROCESSED = PROJECT_ROOT / "data/processed"
OUTPUT = PROJECT_ROOT / "output"


def load_sensor_csv(filepath):
    cols = ['time', 'depth_s', 'level']
    df = pd.read_csv(filepath, usecols=cols, parse_dates=['time'])
    df = df.dropna(subset=['depth_s'])
    return df


def load_rain_gauge(filepath):
    df = pd.read_csv(filepath, sep=';', skiprows=2, parse_dates=[0])
    df.columns = ['time', 'intensity']
    return df


def load_radar(filepath):
    df = pd.read_csv(filepath, sep=';', skiprows=3, parse_dates=[0])
    df.columns = ['time', 'intensity']
    return df


def merge_sensor_parts(sensor_files):
    dfs = [load_sensor_csv(f) for f in sensor_files]
    merged = pd.concat(dfs, ignore_index=False)
    merged = merged.sort_values('time').reset_index(drop=True)
    return merged


def print_summary(df, name):
    print(f"\n{name}:")
    print(f"  Shape: {df.shape}")
    print(f"  Time range: {df['time'].min()} to {df['time'].max()}")
    print(f"  Columns: {list(df.columns)}")
