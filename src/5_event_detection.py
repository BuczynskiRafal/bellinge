import pandas as pd
import numpy as np
import pickle
from utils import *

print("="*60)
print("STEP 5: EVENT DETECTION AND TEMPORAL SEGMENTATION")
print("="*60)


# 1. LOAD FEATURE DATA (BEFORE NORMALIZATION)
print("\n1. Loading feature data...")

with open(DATA_INTERIM / "features.pkl", 'rb') as f:
    data = pickle.load(f)

data = data.sort_values('time').reset_index(drop=True)

print(f"  Feature data: {data.shape}")


# 2. DUAL-THRESHOLD RAINFALL EVENT DETECTION
print("\n2. Detecting rainfall events...")

data['rain_intensity'] = data['I_t']

data['is_rain'] = (data['rain_intensity'] > 0.1).astype(int)

data['rain_change'] = data['is_rain'].diff()

event_starts = data[data['rain_change'] == 1].index.tolist()
event_ends = []

for start in event_starts:
    dry_period = data.loc[start:, 'is_rain'].rolling(window=360).sum()
    end_candidates = dry_period[dry_period == 0].index

    if len(end_candidates) > 0:
        event_ends.append(end_candidates[0])
    else:
        event_ends.append(len(data) - 1)

data['rainfall_event'] = 0
for start, end in zip(event_starts, event_ends):
    data.loc[start:end, 'rainfall_event'] = 1

n_events = len(event_starts)
print(f"  Detected {n_events} rainfall events")


# 3. IDENTIFY FLASH FLOOD EVENTS
print("\n3. Identifying flash flood events...")

print("\nDiagnostics - Rain intensity statistics:")
print(f"  Max: {data['rain_intensity'].max():.2f}")
print(f"  Mean: {data['rain_intensity'].mean():.2f}")
print(f"  95th percentile: {data['rain_intensity'].quantile(0.95):.2f}")
print(f"  99th percentile: {data['rain_intensity'].quantile(0.99):.2f}")
print(f"  99.9th percentile: {data['rain_intensity'].quantile(0.999):.2f}")
print(f"  Times > 20: {(data['rain_intensity'] > 20).sum()}")
print(f"  Times > 10: {(data['rain_intensity'] > 10).sum()}")
print(f"  Times > 5: {(data['rain_intensity'] > 5).sum()}")
print(f"  Times > 1: {(data['rain_intensity'] > 1).sum()}")

flash_flood_intensity = 6
flash_flood_threshold = 15
required_minutes = 11

data['high_intensity'] = (data['rain_intensity'] > flash_flood_intensity).astype(int)

data['flash_flood'] = 0

for i in range(len(data) - flash_flood_threshold):
    window = data.loc[i:i+flash_flood_threshold-1, 'high_intensity']

    if window.sum() >= required_minutes:
        data.loc[i:i+flash_flood_threshold-1, 'flash_flood'] = 1

n_flash_floods = (data['flash_flood'] == 1).sum()
flash_flood_ratio = n_flash_floods / len(data)

total_minutes = len(data)
total_years = (data['time'].max() - data['time'].min()).days / 365.25
events_per_year = n_flash_floods / 60 / total_years

print(f"\nFlash flood detection (>{flash_flood_intensity}mm/h, {required_minutes}/{flash_flood_threshold} min):")
print(f"  Flash flood timesteps: {n_flash_floods}")
print(f"  Flash flood ratio: {flash_flood_ratio:.2%}")
print(f"  Estimated events per year: {events_per_year:.1f}")
print(f"  Target: 8-12 events/year")


# 4. CREATE BINARY CLASSIFICATION TARGET
print("\n4. Creating binary classification target...")

data['target'] = data['flash_flood']

class_counts = data['target'].value_counts()
print(f"  Class 0 (normal): {class_counts.get(0, 0)}")
print(f"  Class 1 (flash flood): {class_counts.get(1, 0)}")


# 5. SPLIT INTO TRAIN/VAL/TEST
print("\n5. Splitting into train/val/test...")

data['year'] = data['time'].dt.year

train_labeled = data[data['year'] <= 2017].copy()
val_labeled = data[data['year'] == 2018].copy()
test_labeled = data[data['year'] == 2019].copy()

print(f"  Train: {train_labeled.shape}, flash floods: {train_labeled['target'].sum()}")
print(f"  Val: {val_labeled.shape}, flash floods: {val_labeled['target'].sum()}")
print(f"  Test: {test_labeled.shape}, flash floods: {test_labeled['target'].sum()}")


# 6. SAVE LABELED DATA
print("\n6. Saving labeled data...")

with open(DATA_INTERIM / "train_labeled.pkl", 'wb') as f:
    pickle.dump(train_labeled, f)

with open(DATA_INTERIM / "val_labeled.pkl", 'wb') as f:
    pickle.dump(val_labeled, f)

with open(DATA_INTERIM / "test_labeled.pkl", 'wb') as f:
    pickle.dump(test_labeled, f)

event_summary = pd.DataFrame({
    'n_rainfall_events': [n_events],
    'n_flash_flood_timesteps': [n_flash_floods],
    'flash_flood_ratio': [flash_flood_ratio]
})
event_summary.to_csv(OUTPUT / "reports" / "event_summary.csv", index=False)

print("\nSTEP 5 COMPLETE!")
print(f"  Saved train_labeled.pkl: {train_labeled.shape}")
print(f"  Saved val_labeled.pkl: {val_labeled.shape}")
print(f"  Saved test_labeled.pkl: {test_labeled.shape}")
