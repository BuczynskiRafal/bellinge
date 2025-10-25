import pandas as pd
import numpy as np
import pickle
from utils import *

print("="*60)
print("STEP 3: FEATURE ENGINEERING")
print("="*60)


# 1. LOAD CLEANED DATA
print("\n1. Loading cleaned data...")

with open(DATA_INTERIM / "sensor_cleaned.pkl", 'rb') as f:
    sensor_data = pickle.load(f)

with open(DATA_INTERIM / "rain_cleaned.pkl", 'rb') as f:
    rain_data = pickle.load(f)

print(f"  Sensor data: {sensor_data.shape}")
print(f"  Rain data: {rain_data.shape}")


# 2. MERGE SENSOR AND RAIN DATA
print("\n2. Merging sensor and rain data on 1-minute resolution...")

data = sensor_data.merge(rain_data, on='time', how='left')
data = data.sort_values('time').reset_index(drop=True)

print(f"  Merged data: {data.shape}")


# 3. RAINFALL FEATURES
print("\n3. Creating rainfall features...")

rain_cols = ['rain_5425', 'rain_5427', 'rain_aabakken']

data['rain_avg'] = data[rain_cols].mean(axis=1)

data['I_t'] = data['rain_avg'].rolling(window=5, min_periods=1).mean()

data['P_t'] = data['rain_avg'].rolling(window=60, min_periods=1).sum()

print("  Created: I(t) - instantaneous intensity (5-min window)")
print("  Created: P(t) - cumulative precipitation (60-min sum)")


# 4. FLOW FEATURES
print("\n4. Creating flow features...")

flowmeter_col = [c for c in data.columns if 'G71F68Yp1' in c and 'power' not in c]

if len(flowmeter_col) > 0:
    data['Q_in'] = data[flowmeter_col[0]]
else:
    data['Q_in'] = 0

print(f"  Created: Q_in(t) - inflow discharge")


# 5. NORMALIZED FILLING DEPTH
print("\n5. Creating normalized filling depth h(t)/D...")

level_cols = [c for c in data.columns if c not in ['time', 'rain_5425', 'rain_5427', 'rain_aabakken', 'rain_avg', 'I_t', 'P_t', 'Q_in']]

pipe_diameters = {
    'G71F04R': 0.8,
    'G71F05R': 1.2,
    'G71F06R': 0.6,
    'G71F68Y': 1.0,
    'G73F010': 0.8,
    'G80F11B': 1.5,
    'G80F13P': 1.2,
    'G80F66Y': 0.9
}

for col in level_cols:
    sensor_id = col.split('_')[0]
    diameter = pipe_diameters.get(sensor_id, 1.0)
    normalized_col = f"{col}_norm"
    data[normalized_col] = data[col] / diameter

print(f"  Created {len(level_cols)} normalized depth features")


# 6. FLOW VELOCITY
print("\n6. Creating flow velocity v(t)...")

for col in level_cols:
    sensor_id = col.split('_')[0]
    diameter = pipe_diameters.get(sensor_id, 1.0)
    area = np.pi * (diameter / 2) ** 2

    velocity_col = f"{col}_velocity"

    if 'Q_in' in data.columns:
        data[velocity_col] = data['Q_in'] / area
    else:
        data[velocity_col] = 0

print(f"  Created {len(level_cols)} velocity features")


# 7. HYDRAULIC GRADIENT
print("\n7. Creating hydraulic gradient ∂h/∂x...")

for i, col in enumerate(level_cols[:-1]):
    gradient_col = f"gradient_{i}"
    data[gradient_col] = data[level_cols[i+1]] - data[level_cols[i]]

print(f"  Created {len(level_cols)-1} gradient features")


# 8. FLOW ACCELERATION
print("\n8. Creating flow acceleration dv/dt...")

velocity_cols = [c for c in data.columns if 'velocity' in c]

for col in velocity_cols:
    accel_col = col.replace('velocity', 'acceleration')
    data[accel_col] = data[col].diff()

print(f"  Created {len(velocity_cols)} acceleration features")


# 9. ANTECEDENT PRECIPITATION INDEX (API)
print("\n9. Creating Antecedent Precipitation Index (API)...")

k = 0.85
data['API'] = 0.0

for i in range(1, len(data)):
    data.loc[i, 'API'] = k * data.loc[i-1, 'API'] + data.loc[i, 'rain_avg']

print("  Created: API - antecedent precipitation index")


# 10. ELAPSED TIME SINCE EVENT START
print("\n10. Creating elapsed time since event τ_event...")

data['event_active'] = (data['rain_avg'] > 0.1).astype(int)
data['tau_event'] = 0

current_tau = 0
for i in range(len(data)):
    if data.loc[i, 'event_active'] == 1:
        current_tau += 1
        data.loc[i, 'tau_event'] = current_tau
    else:
        current_tau = 0

print("  Created: τ_event - elapsed time since event start")


# 11. TEMPORAL ENCODING
print("\n11. Creating temporal features...")

data['hour'] = data['time'].dt.hour
data['hour_sin'] = np.sin(2 * np.pi * data['hour'] / 24)
data['hour_cos'] = np.cos(2 * np.pi * data['hour'] / 24)

data['day_of_week'] = data['time'].dt.dayofweek
for i in range(7):
    data[f'dow_{i}'] = (data['day_of_week'] == i).astype(int)

data['month'] = data['time'].dt.month
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

print("  Created: hour_sin, hour_cos")
print("  Created: dow_0 to dow_6")
print("  Created: month_sin, month_cos")


# 12. SAVE FEATURE MATRIX
print("\n12. Saving feature matrix...")

data.to_csv(DATA_INTERIM / "features.csv", index=False)

with open(DATA_INTERIM / "features.pkl", 'wb') as f:
    pickle.dump(data, f)

feature_names = [c for c in data.columns if c != 'time']
with open(DATA_INTERIM / "feature_names.txt", 'w') as f:
    f.write('\n'.join(feature_names))

print("\nSTEP 3 COMPLETE!")
print(f"  Feature matrix: {data.shape}")
print(f"  Total features: {len(feature_names)}")
print(f"  Saved to data/interim/features.pkl")
