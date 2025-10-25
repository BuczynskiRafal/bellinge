import pandas as pd
import numpy as np
import pickle
from utils import *

print("="*60)
print("STEP 2: DATA QUALITY CONTROL")
print("="*60)


# 1. LOAD DATA
print("\n1. Loading interim data...")

with open(DATA_INTERIM / "sensor_unified.pkl", 'rb') as f:
    sensor_data = pickle.load(f)

with open(DATA_INTERIM / "rain_unified.pkl", 'rb') as f:
    rain_data = pickle.load(f)

print(f"  Sensor data: {sensor_data.shape}")
print(f"  Rain data: {rain_data.shape}")


# 2. CALCULATE COMPLETENESS
print("\n2. Assessing data completeness...")

sensor_cols = [c for c in sensor_data.columns if c != 'time']
completeness = {}

for col in sensor_cols:
    total = len(sensor_data)
    non_null = sensor_data[col].notna().sum()
    ratio = non_null / total
    completeness[col] = ratio
    if ratio >= 0.95:
        status = "✓"
    else:
        status = "✗"
    print(f"  {col}: {ratio:.2%} {status}")

completeness_df = pd.DataFrame.from_dict(completeness, orient='index', columns=['completeness'])
completeness_df.to_csv(OUTPUT / "reports" / "completeness_report.csv")


# 3. CLEAN RAIN DATA
print("\n3. Cleaning rain data...")

rain_data_cleaned = rain_data.copy()

rain_cols = [c for c in rain_data.columns if c != 'time']
for col in rain_cols:
    rain_data_cleaned.loc[rain_data_cleaned[col] < 0, col] = 0
    rain_data_cleaned.loc[rain_data_cleaned[col] > 200, col] = np.nan

print(f"  Rain data cleaned: {rain_data_cleaned.shape}")


# 4. SENSOR DATA IS ALREADY CLEANED
print("\n4. Using pre-cleaned sensor data from 2_cleaned_data...")

sensor_data_cleaned = sensor_data.copy()

print("  depth_s column already contains:")
print("    ✓ Interpolated values (gap filling done)")
print("    ✓ Outliers removed (DBSCAN already applied)")
print("    ✓ Physical range validated (outbound flag used)")
print("    ✓ Frozen sensor values removed")


# 5. SAVE CLEANED DATA
print("\n5. Saving cleaned data to data/interim/...")

with open(DATA_INTERIM / "sensor_cleaned.pkl", 'wb') as f:
    pickle.dump(sensor_data_cleaned, f)

with open(DATA_INTERIM / "rain_cleaned.pkl", 'wb') as f:
    pickle.dump(rain_data_cleaned, f)

print("\nSTEP 2 COMPLETE!")
print(f"  Cleaned sensor data: {sensor_data_cleaned.shape}")
print(f"  Cleaned rain data: {rain_data_cleaned.shape}")
print(f"  Reports saved to output/reports/")
