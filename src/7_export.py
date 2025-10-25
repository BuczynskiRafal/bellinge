import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from utils import *

print("="*60)
print("STEP 7: FINAL DATA PREPARATION AND EXPORT")
print("="*60)


# 1. LOAD BALANCED DATA
print("\n1. Loading balanced data...")

with open(DATA_INTERIM / "train_balanced.pkl", 'rb') as f:
    train = pickle.load(f)

with open(DATA_INTERIM / "val_final.pkl", 'rb') as f:
    val = pickle.load(f)

with open(DATA_INTERIM / "test_final.pkl", 'rb') as f:
    test = pickle.load(f)

print(f"  Train: {train.shape}")
print(f"  Val: {val.shape}")
print(f"  Test: {test.shape}")


# 2. CREATE SLIDING WINDOW SEQUENCES
print("\n2. Creating sliding window sequences...")

T_in = 60
T_out = 15
stride = 5

feature_cols = [c for c in train.columns if c not in ['time', 'year', 'target']]


def create_and_save_sequences(df, T_in, T_out, stride, output_prefix):
    df_sorted = df.sort_values('time').reset_index(drop=True)
    n_samples = len(df_sorted)
    n_features = len(feature_cols)

    chunk_files = []
    chunk_id = 0
    chunk_size = 5000

    for start_idx in range(0, n_samples - T_in - T_out, stride * chunk_size):
        end_idx = min(start_idx + stride * chunk_size, n_samples - T_in - T_out)

        chunk_indices = list(range(start_idx, end_idx, stride))
        n_chunk = len(chunk_indices)

        X_chunk = np.zeros((n_chunk, T_in, n_features), dtype=np.float32)
        y_chunk = np.zeros(n_chunk, dtype=np.int32)

        for j, idx in enumerate(chunk_indices):
            window_df = df_sorted.iloc[idx:idx+T_in]
            X_chunk[j] = window_df[feature_cols].fillna(0).values.astype(np.float32)
            y_chunk[j] = df_sorted.iloc[idx + T_in + T_out]['target']

        chunk_file = DATA_PROCESSED / f"{output_prefix}_chunk_{chunk_id}.pkl"
        with open(chunk_file, 'wb') as f:
            pickle.dump({'X': X_chunk, 'y': y_chunk}, f)

        chunk_files.append(chunk_file)
        chunk_id += 1

        print(f"    Chunk {chunk_id}: {n_chunk} sequences")

        del X_chunk, y_chunk

    print(f"    Calculating total sequences...")
    total_sequences = 0
    for chunk_file in chunk_files:
        with open(chunk_file, 'rb') as f:
            chunk_data = pickle.load(f)
            total_sequences += len(chunk_data['y'])

    print(f"    Creating memory-mapped arrays for {total_sequences} sequences...")
    X_mmap = np.lib.format.open_memmap(
        DATA_PROCESSED / f"{output_prefix}_X.npy",
        mode='w+',
        dtype=np.float32,
        shape=(total_sequences, T_in, n_features)
    )
    y_mmap = np.lib.format.open_memmap(
        DATA_PROCESSED / f"{output_prefix}_y.npy",
        mode='w+',
        dtype=np.int32,
        shape=(total_sequences,)
    )

    print(f"    Writing chunks to memory-mapped files...")
    offset = 0
    for i, chunk_file in enumerate(chunk_files):
        with open(chunk_file, 'rb') as f:
            chunk_data = pickle.load(f)
            n_chunk = len(chunk_data['y'])
            X_mmap[offset:offset+n_chunk] = chunk_data['X']
            y_mmap[offset:offset+n_chunk] = chunk_data['y']
            offset += n_chunk
        chunk_file.unlink()

        if (i + 1) % 50 == 0 or i == len(chunk_files) - 1:
            print(f"      Written {i+1}/{len(chunk_files)} chunks")

    del X_mmap, y_mmap

    return total_sequences


print("  Creating and saving train sequences...")
n_train = create_and_save_sequences(train, T_in, T_out, stride, "train")

print("  Creating and saving val sequences...")
n_val = create_and_save_sequences(val, T_in, T_out, stride, "val")

print("  Creating and saving test sequences...")
n_test = create_and_save_sequences(test, T_in, T_out, stride, "test")

print(f"  Train sequences: {n_train}")
print(f"  Val sequences: {n_val}")
print(f"  Test sequences: {n_test}")

# 3. LOAD ONLY SMALL DATASETS FOR VISUALIZATION
print("\n3. Loading val/test sequences for visualization...")

y_train = np.load(DATA_PROCESSED / "train_y.npy")
X_val = np.load(DATA_PROCESSED / "val_X.npy")
y_val = np.load(DATA_PROCESSED / "val_y.npy")
X_test = np.load(DATA_PROCESSED / "test_X.npy")
y_test = np.load(DATA_PROCESSED / "test_y.npy")

with open(DATA_PROCESSED / "feature_names.pkl", 'wb') as f:
    pickle.dump(feature_cols, f)

print("  Saved X_train.pkl, y_train.pkl")
print("  Saved X_val.pkl, y_val.pkl")
print("  Saved X_test.pkl, y_test.pkl")
print("  Saved feature_names.pkl")


# 4. GENERATE DATA SUMMARY REPORT
print("\n4. Generating data summary report...")

summary = {
    'train_sequences': n_train,
    'val_sequences': n_val,
    'test_sequences': n_test,
    'input_window': T_in,
    'forecast_horizon': T_out,
    'stride': stride,
    'n_features': len(feature_cols),
    'train_flash_floods': int(y_train.sum()),
    'val_flash_floods': int(y_val.sum()),
    'test_flash_floods': int(y_test.sum())
}

summary_df = pd.DataFrame([summary])
summary_df.to_csv(OUTPUT / "reports" / "final_summary.csv", index=False)

print("\n  SUMMARY:")
for key, value in summary.items():
    print(f"    {key}: {value}")


# 5. DATASET STATISTICS
print("\n5. Generating dataset statistics...")

stats = {}
for col in feature_cols[:10]:
    stats[col] = {
        'mean': train[col].mean(),
        'std': train[col].std(),
        'min': train[col].min(),
        'max': train[col].max()
    }

stats_df = pd.DataFrame(stats).T
stats_df.to_csv(OUTPUT / "reports" / "feature_statistics.csv")
print(f"  Saved feature statistics for {len(feature_cols)} features")


# 6. CREATE VISUALIZATIONS
print("\n6. Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].bar(['Train', 'Val', 'Test'], [n_train, len(X_val), len(X_test)])
axes[0, 0].set_title('Sequence Counts')
axes[0, 0].set_ylabel('Number of Sequences')

axes[0, 1].bar(['Train', 'Val', 'Test'],
               [y_train.sum(), y_val.sum(), y_test.sum()])
axes[0, 1].set_title('Flash Flood Events')
axes[0, 1].set_ylabel('Number of Flash Floods')

class_dist = [len(y_train) - y_train.sum(), y_train.sum()]
axes[1, 0].pie(class_dist, labels=['Normal', 'Flash Flood'], autopct='%1.1f%%')
axes[1, 0].set_title('Training Class Distribution')

if 'I_t' in train.columns:
    axes[1, 1].hist(train['I_t'].dropna(), bins=50, alpha=0.7)
    axes[1, 1].set_title('Rainfall Intensity Distribution')
    axes[1, 1].set_xlabel('Intensity (mm/h)')
    axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig(OUTPUT / "visualizations" / "data_summary.png", dpi=150)
print("  Saved data_summary.png")


# 7. CREATE CORRELATION HEATMAP
print("\n7. Creating feature correlation heatmap...")

correlation_features = feature_cols[:20]
corr_matrix = train[correlation_features].corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap (Top 20 Features)')
plt.tight_layout()
plt.savefig(OUTPUT / "visualizations" / "correlation_heatmap.png", dpi=150)
print("  Saved correlation_heatmap.png")


# 8. DOCUMENT PREPROCESSING PIPELINE
print("\n8. Documenting preprocessing pipeline...")

pipeline_config = {
    'data_sources': {
        'sensors': '2_cleaned_data/*.csv',
        'rain_gauges': '3a_Raingauges/*_ts.txt',
        'radar_xband': 'Local_X-band/*.txt',
        'radar_cband': 'DMI_C-band/*.txt',
        'meteorological': '3b_Meterologicalstation/*.p',
        'assets': '1_Assetdata/*.shp'
    },
    'quality_control': {
        'completeness_threshold': 0.95,
        'dbscan_eps_factor': 0.3,
        'dbscan_min_samples': 5,
        'gap_fill_max_minutes': 15,
        'interpolation_method': 'cubic_spline'
    },
    'features': {
        'rainfall_window': '5_minutes',
        'cumulative_window': '60_minutes',
        'flash_flood_threshold': '20_mm_h',
        'flash_flood_duration': '15_minutes'
    },
    'normalization': {
        'train_years': '2010-2017',
        'val_year': '2018',
        'test_year': '2019',
        'continuous': 'zscore',
        'bounded': 'minmax',
        'rain': 'log'
    },
    'class_balancing': {
        'method': 'ADASYN',
        'sampling_strategy': 0.75,
        'noise_sigma': 0.05,
        'temporal_jitter': '±10_minutes'
    },
    'sequences': {
        'input_window': T_in,
        'forecast_horizon': T_out,
        'stride': stride
    }
}

with open(DATA_PROCESSED / "pipeline_config.pkl", 'wb') as f:
    pickle.dump(pipeline_config, f)

print("  Saved pipeline_config.pkl")


print("\n" + "="*60)
print("STEP 7 COMPLETE!")
print("="*60)
print("\nFINAL DELIVERABLES:")
print(f"  ✓ Sequence tensors: data/processed/X_*.pkl, y_*.pkl")
print(f"  ✓ Feature names: data/processed/feature_names.pkl")
print(f"  ✓ Pipeline config: data/processed/pipeline_config.pkl")
print(f"  ✓ Reports: output/reports/*.csv")
print(f"  ✓ Visualizations: output/visualizations/*.png")
print("\n" + "="*60)
print("PREPROCESSING PIPELINE COMPLETE!")
print("="*60)
