import pandas as pd
import numpy as np
import pickle
from utils import *

print("="*60)
print("STEP 4: NORMALIZATION")
print("="*60)


# 1. LOAD FEATURE DATA
print("\n1. Loading feature matrix...")

with open(DATA_INTERIM / "features.pkl", 'rb') as f:
    data = pickle.load(f)

print(f"  Feature matrix: {data.shape}")


# 2. CHRONOLOGICAL SPLIT
print("\n2. Splitting data chronologically...")

data['year'] = data['time'].dt.year

train = data[data['year'] <= 2017].copy()
val = data[data['year'] == 2018].copy()
test = data[data['year'] == 2019].copy()

print(f"  Train (2010-2017): {train.shape}")
print(f"  Validation (2018): {val.shape}")
print(f"  Test (2019): {test.shape}")


# 3. IDENTIFY FEATURE TYPES
print("\n3. Identifying feature types...")

feature_cols = [c for c in data.columns if c not in ['time', 'year', 'event_active', 'event_change']]

temporal_features = [c for c in feature_cols if any(x in c for x in ['sin', 'cos', 'dow_'])]
bounded_features = [c for c in feature_cols if 'norm' in c]
rain_features = [c for c in feature_cols if 'rain' in c or c in ['I_t', 'P_t', 'API']]
continuous_features = [c for c in feature_cols if c not in temporal_features + bounded_features + rain_features]

print(f"  Temporal features: {len(temporal_features)}")
print(f"  Bounded features: {len(bounded_features)}")
print(f"  Rain features: {len(rain_features)}")
print(f"  Continuous features: {len(continuous_features)}")


# 4. CALCULATE NORMALIZATION PARAMETERS FROM TRAINING SET
print("\n4. Calculating normalization parameters from training set...")

norm_params = {}

for col in continuous_features:
    values = train[col].dropna()
    if len(values) > 0:
        mu = values.mean()
        sigma = values.std()
        norm_params[col] = {'type': 'zscore', 'mu': mu, 'sigma': sigma}

for col in bounded_features:
    values = train[col].dropna()
    if len(values) > 0:
        min_val = values.min()
        max_val = values.max()
        norm_params[col] = {'type': 'minmax', 'min': min_val, 'max': max_val}

for col in rain_features:
    values = train[col].dropna()
    if len(values) > 0:
        epsilon = 1e-6
        norm_params[col] = {'type': 'log', 'epsilon': epsilon}

for col in temporal_features:
    norm_params[col] = {'type': 'none'}

print(f"  Calculated parameters for {len(norm_params)} features")


# 5. APPLY NORMALIZATION
print("\n5. Applying normalization...")

def normalize_data(df, params):
    df_norm = df.copy()

    for col, param in params.items():
        if col not in df_norm.columns:
            continue

        if param['type'] == 'zscore':
            df_norm[col] = (df_norm[col] - param['mu']) / param['sigma']

        elif param['type'] == 'minmax':
            df_norm[col] = (df_norm[col] - param['min']) / (param['max'] - param['min'])

        elif param['type'] == 'log':
            df_norm[col] = np.log(df_norm[col] + param['epsilon'])

    return df_norm

train_norm = normalize_data(train, norm_params)
val_norm = normalize_data(val, norm_params)
test_norm = normalize_data(test, norm_params)

print(f"  Normalized train: {train_norm.shape}")
print(f"  Normalized val: {val_norm.shape}")
print(f"  Normalized test: {test_norm.shape}")


# 6. VERIFY NO INFORMATION LEAKAGE
print("\n6. Verifying no information leakage...")

for col in continuous_features[:5]:
    if col in train_norm.columns:
        train_mean = train_norm[col].mean()
        val_mean = val_norm[col].mean()
        test_mean = test_norm[col].mean()
        print(f"  {col}: train_mean={train_mean:.3f}, val_mean={val_mean:.3f}, test_mean={test_mean:.3f}")

print("  ✓ Normalization uses only training parameters")


# 7. SAVE NORMALIZED DATA
print("\n7. Saving normalized data...")

with open(DATA_INTERIM / "train_normalized.pkl", 'wb') as f:
    pickle.dump(train_norm, f)

with open(DATA_INTERIM / "val_normalized.pkl", 'wb') as f:
    pickle.dump(val_norm, f)

with open(DATA_INTERIM / "test_normalized.pkl", 'wb') as f:
    pickle.dump(test_norm, f)

with open(DATA_INTERIM / "norm_params.pkl", 'wb') as f:
    pickle.dump(norm_params, f)

norm_params_df = pd.DataFrame.from_dict(norm_params, orient='index')
norm_params_df.to_csv(OUTPUT / "reports" / "normalization_params.csv")

print("\nSTEP 4 COMPLETE!")
print(f"  Saved train_normalized.pkl: {train_norm.shape}")
print(f"  Saved val_normalized.pkl: {val_norm.shape}")
print(f"  Saved test_normalized.pkl: {test_norm.shape}")
print(f"  Saved norm_params.pkl")
