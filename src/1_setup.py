import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
import pickle
from utils import *

print("="*60)
print("STEP 1: SETUP AND DATA LOADING")
print("="*60)


# 1. LOAD SENSOR DATA
print("\n1. Loading sensor data from 2_cleaned_data...")

sensor_dir = DOWNLOADED / "2_cleaned_data"
sensor_files = list(sensor_dir.glob("*.csv"))

sensors = {}
for f in sensor_files:
    sensor_id = f.stem.replace("_proc_v6", "")
    df = load_sensor_csv(f)

    base_name = "_".join(sensor_id.split("_")[:-1])
    if base_name not in sensors:
        sensors[base_name] = []
    sensors[base_name].append((sensor_id, df))

merged_sensors = {}
for base_name, parts in sensors.items():
    if len(parts) == 1:
        merged_sensors[base_name] = parts[0][1]
    else:
        dfs = [p[1] for p in parts]
        merged = pd.concat(dfs, ignore_index=False)
        merged = merged.sort_values('time').drop_duplicates(subset=['time'])
        merged_sensors[base_name] = merged

print(f"  Loaded {len(merged_sensors)} unique sensors")
for name, df in list(merged_sensors.items())[:3]:
    print_summary(df, name)


# 2. LOAD RAIN GAUGE DATA
print("\n2. Loading rain gauge data from 3a_Raingauges...")

rain_gauges = {}
for gauge_id in ['5425', '5427', 'Aabakken']:
    if gauge_id == 'Aabakken':
        filepath = DOWNLOADED / "3a_Raingauges" / "Aabakken_bellinge_vandvaerk_v2_ts.txt"
    else:
        filepath = DOWNLOADED / "3a_Raingauges" / f"{gauge_id}_ts.txt"

    df = load_rain_gauge(filepath)
    rain_gauges[gauge_id] = df
    print(f"  {gauge_id}: {len(df)} records, {df['time'].min()} to {df['time'].max()}")


# 3. LOAD RADAR DATA
print("\n3. Loading radar data...")

xband_dir = DOWNLOADED / "Local_X-band"
xband_files = list(xband_dir.glob("*.txt"))
xband_data = {}
for f in xband_files[:5]:
    catch_id = f.stem.split("_")[-1]
    df = load_radar(f)
    xband_data[catch_id] = df

print(f"  X-band: {len(xband_data)} catchments loaded")

cband_dir = DOWNLOADED / "DMI_C-band"
cband_files = list(cband_dir.glob("*.txt"))
cband_data = {}
for f in cband_files[:5]:
    catch_id = f.stem.split("_")[-1]
    df = load_radar(f)
    cband_data[catch_id] = df

print(f"  C-band: {len(cband_data)} catchments loaded")


# 4. LOAD METEOROLOGICAL DATA
print("\n4. Loading meteorological station data...")

met_dir = DOWNLOADED / "3b_Meterologicalstation"
met_data = {}
for var in ['precip_past10min', 'temp_dry', 'humidity', 'wind_speed']:
    files = list(met_dir.glob(f"dmi_{var}_*.p"))
    dfs = [pd.read_pickle(f) for f in files]
    met_data[var] = pd.concat(dfs).sort_index()
    print(f"  {var}: {len(met_data[var])} records")


# 5. LOAD ASSET DATA
print("\n5. Loading asset data from 1_Assetdata...")

asset_dir = DOWNLOADED / "1_Assetdata"
links = gpd.read_file(asset_dir / "Links.shp")
nodes = gpd.read_file(asset_dir / "Manholes.shp")

print(f"  Links (pipes): {len(links)} segments")
print(f"  Nodes: {len(nodes)} computational nodes")


# 6. COMBINE SENSOR DATA INTO SINGLE DATAFRAME
print("\n6. Creating unified sensor dataframe...")

all_data = []
for sensor_name, sensor_df in merged_sensors.items():
    temp_df = sensor_df[['time', 'depth_s']].copy()
    temp_df.columns = ['time', sensor_name]
    temp_df = temp_df.dropna(subset=[sensor_name])
    all_data.append(temp_df)
    del sensor_df

sensor_unified = all_data[0]
for df in all_data[1:]:
    sensor_unified = sensor_unified.merge(df, on='time', how='outer')
    del df

sensor_unified = sensor_unified.sort_values('time').reset_index(drop=True)

print(f"  Unified sensor data: {sensor_unified.shape}")
print(f"  Time range: {sensor_unified['time'].min()} to {sensor_unified['time'].max()}")


# 7. COMBINE RAIN GAUGE DATA
print("\n7. Creating unified rain gauge dataframe...")

rain_5425 = rain_gauges['5425'].rename(columns={'intensity': 'rain_5425'})
rain_5427 = rain_gauges['5427'].rename(columns={'intensity': 'rain_5427'})
rain_aabakken = rain_gauges['Aabakken'].rename(columns={'intensity': 'rain_aabakken'})

rain_unified = rain_5425.merge(rain_5427, on='time', how='outer')
rain_unified = rain_unified.merge(rain_aabakken, on='time', how='outer')
rain_unified = rain_unified.sort_values('time').reset_index(drop=True)

print(f"  Unified rain data: {rain_unified.shape}")


# 8. SAVE INTERIM DATA
print("\n8. Saving interim data...")

with open(DATA_INTERIM / "sensor_unified.pkl", 'wb') as f:
    pickle.dump(sensor_unified, f)

with open(DATA_INTERIM / "rain_unified.pkl", 'wb') as f:
    pickle.dump(rain_unified, f)

with open(DATA_INTERIM / "merged_sensors.pkl", 'wb') as f:
    pickle.dump(merged_sensors, f)

with open(DATA_INTERIM / "rain_gauges.pkl", 'wb') as f:
    pickle.dump(rain_gauges, f)

with open(DATA_INTERIM / "xband_data.pkl", 'wb') as f:
    pickle.dump(xband_data, f)

with open(DATA_INTERIM / "cband_data.pkl", 'wb') as f:
    pickle.dump(cband_data, f)

with open(DATA_INTERIM / "met_data.pkl", 'wb') as f:
    pickle.dump(met_data, f)

links.to_file(DATA_INTERIM / "links.shp")
nodes.to_file(DATA_INTERIM / "nodes.shp")

print("\nSTEP 1 COMPLETE!")
print(f"  Saved sensor_unified.pkl: {sensor_unified.shape}")
print(f"  Saved rain_unified.pkl: {rain_unified.shape}")
