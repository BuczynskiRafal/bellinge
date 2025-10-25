import pandas as pd
import numpy as np
import pickle
from imblearn.over_sampling import RandomOverSampler
from utils import *

print("="*60)
print("STEP 6: CLASS BALANCING AND DATA AUGMENTATION")
print("="*60)


# 1. LOAD LABELED DATA
print("\n1. Loading labeled data...")

with open(DATA_INTERIM / "train_labeled.pkl", 'rb') as f:
    train = pickle.load(f)

with open(DATA_INTERIM / "val_labeled.pkl", 'rb') as f:
    val = pickle.load(f)

with open(DATA_INTERIM / "test_labeled.pkl", 'rb') as f:
    test = pickle.load(f)

print(f"  Train: {train.shape}")
print(f"  Val: {val.shape}")
print(f"  Test: {test.shape}")


# 2. ANALYZE CLASS IMBALANCE
print("\n2. Analyzing class imbalance...")

train_class_counts = train['target'].value_counts()
minority_ratio = train_class_counts.get(1, 0) / len(train)

print(f"  Class 0: {train_class_counts.get(0, 0)}")
print(f"  Class 1: {train_class_counts.get(1, 0)}")
print(f"  Minority ratio: {minority_ratio:.2%}")


# 3. PREPARE DATA FOR OVERSAMPLING
print("\n3. Applying RandomOverSampler...")

feature_cols = [c for c in train.columns if c not in ['time', 'year', 'target', 'rainfall_event', 'flash_flood', 'is_rain', 'rain_change', 'high_intensity', 'event_active']]

X_train = train[feature_cols].fillna(0).values
y_train = train['target'].values

ros = RandomOverSampler(sampling_strategy=0.75, random_state=42)

X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

print(f"  Original: {X_train.shape}")
print(f"  Resampled: {X_resampled.shape}")
print(f"  New class 1 count: {y_resampled.sum()}")


# 4. CREATE BALANCED DATAFRAME
print("\n4. Creating balanced training dataframe...")

train_balanced = pd.DataFrame(X_resampled, columns=feature_cols)
train_balanced['target'] = y_resampled

n_original = len(X_train)
n_synthetic = len(X_resampled) - n_original

times = list(train['time'].values[:n_original])
years = list(train['year'].values[:n_original])

if n_synthetic > 0:
    random_indices = np.random.choice(n_original, n_synthetic)
    synthetic_times = [train['time'].iloc[idx] + pd.Timedelta(minutes=np.random.randint(-10, 10)) for idx in random_indices]
    synthetic_years = [train['year'].iloc[idx] for idx in random_indices]

    times.extend(synthetic_times)
    years.extend(synthetic_years)

train_balanced['time'] = times
train_balanced['year'] = years

print(f"  Balanced train: {train_balanced.shape}")


# 5. GAUSSIAN NOISE AUGMENTATION
print("\n5. Applying Gaussian noise augmentation...")

noise_sigma = 0.05

for col in feature_cols:
    signal_std = train[col].std()
    noise = np.random.normal(0, noise_sigma * signal_std, len(train_balanced))
    train_balanced[col] = train_balanced[col] + noise

print(f"  Added Gaussian noise (σ = 0.05 × σ_signal)")


# 6. VERIFY BALANCED DISTRIBUTION
print("\n6. Verifying balanced distribution...")

balanced_counts = train_balanced['target'].value_counts()
print(f"  Class 0: {balanced_counts.get(0, 0)}")
print(f"  Class 1: {balanced_counts.get(1, 0)}")
print(f"  New ratio: {balanced_counts.get(1, 0) / len(train_balanced):.2%}")


# 7. SAVE BALANCED DATA
print("\n7. Saving balanced data...")

with open(DATA_INTERIM / "train_balanced.pkl", 'wb') as f:
    pickle.dump(train_balanced, f)

with open(DATA_INTERIM / "val_final.pkl", 'wb') as f:
    pickle.dump(val, f)

with open(DATA_INTERIM / "test_final.pkl", 'wb') as f:
    pickle.dump(test, f)

balance_summary = pd.DataFrame({
    'split': ['train_original', 'train_balanced', 'val', 'test'],
    'class_0': [train_class_counts.get(0, 0), balanced_counts.get(0, 0),
                val['target'].value_counts().get(0, 0), test['target'].value_counts().get(0, 0)],
    'class_1': [train_class_counts.get(1, 0), balanced_counts.get(1, 0),
                val['target'].value_counts().get(1, 0), test['target'].value_counts().get(1, 0)]
})
balance_summary.to_csv(OUTPUT / "reports" / "class_balance.csv", index=False)

print("\nSTEP 6 COMPLETE!")
print(f"  Saved train_balanced.pkl: {train_balanced.shape}")
print(f"  Saved val_final.pkl: {val.shape}")
print(f"  Saved test_final.pkl: {test.shape}")
