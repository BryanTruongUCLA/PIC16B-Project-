"""
Kalshi Prediction Market Model for Tesla Stock Price
Uses Kalshi market data as features to predict Tesla price movements
Tests multiple time horizons with proper feature engineering
"""

import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Paths
STOCK_DATA_PATH = r"C:\Users\komal\PIC16B-Project-\normalized_tesla_stock_1min.csv"
MARKET_DATA_DIR = r"C:\Users\komal\PIC16B-Project-\market_data"

# Load Tesla stock data
tesla_stock_data = pd.read_csv(STOCK_DATA_PATH)
tesla_stock_data['timestamp'] = pd.to_datetime(tesla_stock_data['datetime'])

# Load Kalshi market data
# Get all CSV files in market_data directory
csv_files = [f for f in os.listdir(MARKET_DATA_DIR) if f.endswith('_historical_1min.csv')]

kalshi_data_list = []
for csv_file in csv_files:
    file_path = os.path.join(MARKET_DATA_DIR, csv_file)
    try:
        df = pd.read_csv(file_path)

        # Handle datetime column names
        df['datetime'] = pd.to_datetime(df['datetime'])
        # Parse close_time
        df['close_time'] = pd.to_datetime(df['close_time'], errors='coerce')

        kalshi_data_list.append(df)
    except Exception as e:
        print(f"Error reading {csv_file}: {e}")

# Combine all market data (each file is a separate market)
kalshi_combined = pd.concat(kalshi_data_list, ignore_index=True)

# Create per-market features
# Each market gets its own columns: market1_yes_bid, market1_yes_ask, market1_volume, etc.
# Select core features per market
feature_cols = ['yes_bid_close', 'yes_ask_close', 'volume', 'open_interest']
available_features = [col for col in feature_cols if col in kalshi_combined.columns]

# Create a long-format dataframe for pivoting
data_for_pivot = kalshi_combined[['datetime', 'market_ticker'] + available_features].copy()

# Pivot each feature separately
pivoted_dfs = []

for feature in available_features:
    # Pivot rows=datetime, columns=market_ticker, values=feature
    pivoted = data_for_pivot.pivot_table(
        index='datetime',
        columns='market_ticker',
        values=feature,
        aggfunc='first'
    )

    # Rename columns to market_feature format
    pivoted.columns = [f"{col}_{feature}" for col in pivoted.columns]
    pivoted_dfs.append(pivoted)

# Combine all pivoted features
kalshi_features = pd.concat(pivoted_dfs, axis=1).reset_index()
kalshi_features = kalshi_features.rename(columns={'datetime': 'timestamp'})

# Calculate per-market time-to-expiry features
if 'close_time' in kalshi_combined.columns:
    # Remove timezone info to avoid conflicts
    kalshi_combined['datetime_naive'] = pd.to_datetime(kalshi_combined['datetime']).dt.tz_localize(None)
    kalshi_combined['close_time_naive'] = pd.to_datetime(kalshi_combined['close_time'], errors='coerce').dt.tz_localize(None)

    # Filter valid data
    valid_data = kalshi_combined[
        kalshi_combined['datetime_naive'].notna() &
        kalshi_combined['close_time_naive'].notna()
    ].copy()

    # Calculate time to expiry in hours for each market
    valid_data['ttm_hours'] = (
        (valid_data['close_time_naive'] - valid_data['datetime_naive']).dt.total_seconds() / 3600
    ).clip(lower=0.1)

    # Create per-market TTM and velocity features
    # Add derived features per market
    derived_features = []

    if 'yes_bid_close' in valid_data.columns:
        valid_data['bid_velocity'] = valid_data['yes_bid_close'] / valid_data['ttm_hours']
        derived_features.append('bid_velocity')

    if 'yes_ask_close' in valid_data.columns:
        valid_data['ask_velocity'] = valid_data['yes_ask_close'] / valid_data['ttm_hours']
        derived_features.append('ask_velocity')

    if 'open_interest' in valid_data.columns:
        valid_data['interest_velocity'] = valid_data['open_interest'] / valid_data['ttm_hours']
        derived_features.append('interest_velocity')

    if 'volume' in valid_data.columns:
        valid_data['volume_velocity'] = valid_data['volume'] / valid_data['ttm_hours']
        derived_features.append('volume_velocity')

    # Add TTM itself as a feature
    derived_features.append('ttm_hours')

    # Pivot the derived features by market
    ttm_pivoted_dfs = []

    for feature in derived_features:
        pivoted = valid_data.pivot_table(
            index='datetime_naive',
            columns='market_ticker',
            values=feature,
            aggfunc='first'
        )
        pivoted.columns = [f"{col}_{feature}" for col in pivoted.columns]
        ttm_pivoted_dfs.append(pivoted)

    # Concatenate and merge with main features
    if ttm_pivoted_dfs:
        ttm_features = pd.concat(ttm_pivoted_dfs, axis=1).reset_index()
        ttm_features = ttm_features.rename(columns={'datetime_naive': 'timestamp'})

        print(f"    Created {len(ttm_features.columns)-1} TTM/velocity features")
        kalshi_features = kalshi_features.merge(ttm_features, on='timestamp', how='left')
    else:
        print("No Time Till Market expiration features created")
else:
    print("No close_time column found, no Time Till Market expiration features created")

# Merge stock and Kalshi data
tesla_stock_data['timestamp'] = tesla_stock_data['timestamp'].dt.tz_localize(None)
merged_data = tesla_stock_data.merge(
    kalshi_features,
    on='timestamp',
    how='inner'
)

# Sort by timestamp (CRITICAL for time series)
merged_data = merged_data.sort_values('timestamp').reset_index(drop=True)

# Create continuous timeline and reindex
min_time = merged_data['timestamp'].min()
max_time = merged_data['timestamp'].max()
continuous_timeline = pd.date_range(start=min_time, end=max_time, freq='1min')

merged_data = merged_data.set_index('timestamp').reindex(continuous_timeline)

# Separate stock and Kalshi columns for forward fill
tsla_spy_price_cols = [col for col in merged_data.columns if 'tsla_' in col.lower() or 'spy_' in col.lower()]
kalshi_cols = [col for col in merged_data.columns if col not in tsla_spy_price_cols]

# Forward fill Kalshi features with limit of 120 minutes, then fill remaining NaN with 0
merged_data[kalshi_cols] = merged_data[kalshi_cols].ffill(limit=120).fillna(0)

# Forward fill stock prices (no limit)
merged_data[tsla_spy_price_cols] = merged_data[tsla_spy_price_cols].ffill()

# Create normalized Tesla/SPY ratio
merged_data['tsla_spy_ratio'] = merged_data['tsla_close'] / merged_data['spy_close']

# Drop rows where stock prices are NaN
rows_before = len(merged_data)
merged_data = merged_data.dropna(subset=['tsla_close', 'spy_close', 'tsla_spy_ratio'])
rows_after = len(merged_data)

# Reset index to have timestamp as column again
merged_data = merged_data.reset_index()
merged_data = merged_data.rename(columns={'index': 'timestamp'})

# Create future datetime for 1hr ahead
merged_data['future_datetime_1hr'] = merged_data['timestamp'] + pd.Timedelta(minutes=60)

# Merge with future TSLA/SPY ratios
future_ratios_1hr = merged_data[['timestamp', 'tsla_spy_ratio']].rename(
    columns={'timestamp': 'future_datetime', 'tsla_spy_ratio': 'future_ratio'}
)

temp_merge_1hr = pd.merge_asof(
    merged_data[['timestamp', 'future_datetime_1hr', 'tsla_spy_ratio']].sort_values('future_datetime_1hr'),
    future_ratios_1hr.sort_values('future_datetime'),
    left_on='future_datetime_1hr',
    right_on='future_datetime',
    direction='forward',
    tolerance=pd.Timedelta(minutes=10)
)

temp_merge_1hr = temp_merge_1hr.sort_values('timestamp').reset_index(drop=True)
merged_data['temp_return_1hr'] = (
    (temp_merge_1hr['future_ratio'] - merged_data['tsla_spy_ratio']) / merged_data['tsla_spy_ratio']
)

# Clean up temporary column
merged_data = merged_data.drop(columns=['future_datetime_1hr'])

# Get all Kalshi feature columns
kalshi_feature_cols = [col for col in merged_data.columns
                       if col not in ['timestamp', 'datetime', 'tsla_close', 'tsla_volume',
                                      'tsla_open', 'tsla_high', 'tsla_low',
                                      'spy_close', 'spy_volume', 'spy_open', 'spy_high', 'spy_low',
                                      'tsla_spy_ratio', 'temp_return_1hr']]

# Calculate correlation with 1hr returns for each feature
correlations = {}
valid_return_data = merged_data['temp_return_1hr'].dropna()

for col in kalshi_feature_cols:
    valid_data = merged_data[[col, 'temp_return_1hr']].dropna()
    if len(valid_data) > 100:  # Need sufficient data
        corr = valid_data[col].corr(valid_data['temp_return_1hr'])
        correlations[col] = abs(corr)  # Use absolute correlation
    else:
        correlations[col] = 0.0

# Keep features with correlation greater than the low minimum or velocity/price features
correlation_threshold = 0.01
keep_keywords = ['velocity', 'price', 'bid', 'ask', 'confidence', 'open_interest', 'volume']

features_to_keep = []
for col in kalshi_feature_cols:
    has_useful_keyword = any(keyword in col.lower() for keyword in keep_keywords)
    is_correlated = correlations.get(col, 0) > correlation_threshold

    if has_useful_keyword or is_correlated:
        features_to_keep.append(col)

# Show top correlated features
top_correlated = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:10]
print(f"\nTop 10 most correlated Kalshi features:")
for feat, corr in top_correlated:
    print(f"{feat[:50]:50s}: {corr:.6f}")

# Keep only relevant features
columns_to_keep = ['timestamp', 'datetime', 'tsla_close', 'tsla_volume',
                   'tsla_open', 'tsla_high', 'tsla_low',
                   'spy_close', 'spy_volume', 'spy_open', 'spy_high', 'spy_low',
                   'tsla_spy_ratio'] + features_to_keep

merged_data = merged_data[columns_to_keep]

# Remove temporary return column
if 'temp_return_1hr' in merged_data.columns:
    merged_data = merged_data.drop(columns=['temp_return_1hr'])

# Time horizons in minutes (ONLY ≥1 day predictions)
time_horizons = {
    '1day': 1440,
    '2day': 2880,
    '3day': 4320,
    '4day': 5760,
    '5day': 7200,
    '6day': 8640,
    '7day': 10080
}

for name, horizon in time_horizons.items():
    # Create future datetime column for this horizon
    merged_data[f'future_datetime_{name}'] = merged_data['timestamp'] + pd.Timedelta(minutes=horizon)

    # Merge with future TSLA/SPY ratio
    future_ratios = merged_data[['timestamp', 'tsla_spy_ratio']].copy()
    future_ratios = future_ratios.rename(columns={'timestamp': 'future_datetime', 'tsla_spy_ratio': 'future_ratio'})

    tolerance_minutes = min(30, max(5, horizon // 10))
    temp_merge = pd.merge_asof(
        merged_data[['timestamp', f'future_datetime_{name}', 'tsla_spy_ratio']].sort_values(f'future_datetime_{name}'),
        future_ratios.sort_values('future_datetime'),
        left_on=f'future_datetime_{name}',
        right_on='future_datetime',
        direction='forward',
        tolerance=pd.Timedelta(minutes=tolerance_minutes)
    )

    # Sort back to original order
    temp_merge = temp_merge.sort_values('timestamp').reset_index(drop=True)
    merged_data[f'target_return_{name}'] = (
        (temp_merge['future_ratio'] - merged_data['tsla_spy_ratio']) / merged_data['tsla_spy_ratio']
    )

    # CLASSIFICATION: Only defined if movement greater than 0.5%
    NOISE_THRESHOLD = 0.005

    # Create binary classification where 1 = UP  0 = DOWN
    merged_data[f'target_direction_{name}'] = np.nan
    merged_data.loc[merged_data[f'target_return_{name}'] > NOISE_THRESHOLD, f'target_direction_{name}'] = 1.0
    merged_data.loc[merged_data[f'target_return_{name}'] < -NOISE_THRESHOLD, f'target_direction_{name}'] = 0.0

    # Clean up temporary column
    merged_data = merged_data.drop(columns=[f'future_datetime_{name}'])


feature_cols = features_to_keep.copy() if 'features_to_keep' in locals() and features_to_keep else []

# Ensure stock features are included
stock_features = ['tsla_close', 'tsla_volume', 'spy_close', 'spy_volume']
for sf in stock_features:
    if sf in merged_data.columns and sf not in feature_cols:
        feature_cols.append(sf)

# Handle missing Kalshi features
# Separate stock features from Kalshi features
stock_features_in_cols = [col for col in feature_cols if col in ['tsla_close', 'tsla_volume', 'spy_close', 'spy_volume']]
kalshi_features_in_cols = [col for col in feature_cols if col not in stock_features_in_cols]

# Stock features must be valid (we can't predict without stock data)
stock_valid_mask = merged_data[stock_features_in_cols].notna().all(axis=1)
clean_data = merged_data[stock_valid_mask].copy()

rows_removed = len(merged_data) - len(clean_data)

# Fill missing Kalshi features with 0 (market doesn't exist or no activity)
missing_before = clean_data[kalshi_features_in_cols].isna().sum().sum()
clean_data[kalshi_features_in_cols] = clean_data[kalshi_features_in_cols].fillna(0)

# Extract features
X = clean_data[feature_cols]

# Get time-based split with gaps
max_horizon = max(time_horizons.values())
max_horizon_td = pd.Timedelta(minutes=max_horizon)

min_time = clean_data['timestamp'].min()
max_time = clean_data['timestamp'].max()
total_duration = max_time - min_time

# Split by time: 70% train, 15% val, 15% test with time gaps
train_end_time = min_time + (total_duration * 0.70) - max_horizon_td
val_start_time = train_end_time + max_horizon_td
val_end_time = min_time + (total_duration * 0.85) - max_horizon_td
test_start_time = val_end_time + max_horizon_td

# Create masks based on timestamps
train_mask = clean_data['timestamp'] <= train_end_time
val_mask = (clean_data['timestamp'] >= val_start_time) & (clean_data['timestamp'] <= val_end_time)
test_mask = clean_data['timestamp'] >= test_start_time

X_train_full = X[train_mask]
X_val = X[val_mask]
X_test = X[test_mask]

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import RandomizedSearchCV
import sys

results = {}
best_models = {}  # Store best model for each horizon
horizon_names = list(time_horizons.keys())

for horizon_name in horizon_names:
    print()
    print(f"\n{'='*70}")
    print(f"TIME HORIZON: {horizon_name}")
    print(f"{'='*70}")

    # Get targets for all splits using time-based masks
    # Use direction as target, filter out NaN
    y_all = clean_data[f'target_direction_{horizon_name}'].values

    y_train_full = y_all[train_mask]
    y_val = y_all[val_mask]
    y_test = y_all[test_mask]

    # Remove NaN values from each split
    train_valid_mask = ~np.isnan(y_train_full)
    val_valid_mask = ~np.isnan(y_val)
    test_valid_mask = ~np.isnan(y_test)

    # Apply masks to get valid samples
    train_indices = X_train_full.index[train_valid_mask]
    val_indices = X_val.index[val_valid_mask]
    test_indices = X_test.index[test_valid_mask]

    X_train_valid = X_train_full.loc[train_indices].reset_index(drop=True)
    y_train_valid = y_train_full[train_valid_mask]

    X_val_valid = X_val.loc[val_indices].reset_index(drop=True)
    y_val_valid = y_val[val_valid_mask]

    X_test_valid = X_test.loc[test_indices].reset_index(drop=True)
    y_test_valid = y_test[test_valid_mask]

    # Check class distributions
    train_class_dist = np.bincount(y_train_valid.astype(int))
    val_class_dist = np.bincount(y_val_valid.astype(int)) if len(y_val_valid) > 0 else np.array([0, 0])
    test_class_dist = np.bincount(y_test_valid.astype(int)) if len(y_test_valid) > 0 else np.array([0, 0])

    print(f"\nClass distribution (0=DOWN, 1=UP):")
    print(f"Train: DOWN={train_class_dist[0]:,} ({train_class_dist[0]/len(y_train_valid)*100:.1f}%), UP={train_class_dist[1]:,} ({train_class_dist[1]/len(y_train_valid)*100:.1f}%)")
    if len(y_val_valid) > 0:
        print(f"Val:   DOWN={val_class_dist[0]:,} ({val_class_dist[0]/len(y_val_valid)*100:.1f}%), UP={val_class_dist[1]:,} ({val_class_dist[1]/len(y_val_valid)*100:.1f}%)")
    if len(y_test_valid) > 0:
        print(f"Test:  DOWN={test_class_dist[0]:,} ({test_class_dist[0]/len(y_test_valid)*100:.1f}%), UP={test_class_dist[1]:,} ({test_class_dist[1]/len(y_test_valid)*100:.1f}%)")

    # Check if we have enough samples and both classes in test set
    min_samples_required = 100
    if len(y_test_valid) < min_samples_required:
        print(f"\nNot enough test samples")
        print()
        continue

    if len(y_val_valid) < 50:
        print(f"\nNot enough validation samples")
        print()
        continue

    # Check for class imbalance issues
    if len(test_class_dist) < 2 or test_class_dist.min() < 10:
        print(f"\nTest set has insufficient samples in one class")
        print()
        continue

    print()

    # Use time series split on training data only
    tscv = TimeSeriesSplit(n_splits=5)


    fold_scores = []
    best_params = None
    best_score = -np.inf

    # Cross-validation with PCA inside each fold
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train_valid), 1):
        X_train_fold = X_train_valid.iloc[train_idx]
        X_test_fold = X_train_valid.iloc[test_idx]
        y_train_fold = y_train_valid[train_idx]
        y_test_fold = y_train_valid[test_idx]

        # fit scaler and PCA only on training fold
        scaler_fold = StandardScaler()
        X_train_scaled = scaler_fold.fit_transform(X_train_fold)
        X_test_scaled = scaler_fold.transform(X_test_fold)

        n_components = min(10, len(feature_cols))
        pca_fold = PCA(n_components=n_components)
        X_train_pca = pca_fold.fit_transform(X_train_scaled)
        X_test_pca = pca_fold.transform(X_test_scaled)

        if fold == 1:
            print(f"    PCA: {len(feature_cols)} features -> {n_components} components")
            print(f"    Explained variance: {pca_fold.explained_variance_ratio_.sum():.4f}")

        # Train classifier with default params for this fold
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=0
        )

        model.fit(X_train_pca, y_train_fold)
        y_pred = model.predict(X_test_pca)

        # Calculate classification metrics
        acc = accuracy_score(y_test_fold, y_pred)
        prec = precision_score(y_test_fold, y_pred, zero_division=0)
        rec = recall_score(y_test_fold, y_pred, zero_division=0)
        f1 = f1_score(y_test_fold, y_pred, zero_division=0)

        print(f"Acc: {acc*100:.2f}% Prec: {prec*100:.2f}% Rec: {rec*100:.2f}% F1: {f1:.4f}")

        fold_scores.append({
            'acc': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })

    # Calculate standard errors for error bars
    fold_acc_std = np.std([s['acc'] for s in fold_scores])
    fold_f1_std = np.std([s['f1'] for s in fold_scores])

    # Average metrics across folds
    avg_metrics = {
        'acc': np.mean([s['acc'] for s in fold_scores]),
        'precision': np.mean([s['precision'] for s in fold_scores]),
        'recall': np.mean([s['recall'] for s in fold_scores]),
        'f1': np.mean([s['f1'] for s in fold_scores])
    }

    print(f"\nCross-validation results:")
    print(f"Accuracy:  {avg_metrics['acc']*100:.2f}%")
    print(f"Precision: {avg_metrics['precision']*100:.2f}%")
    print(f"Recall:    {avg_metrics['recall']*100:.2f}%")
    print(f"F1 Score:  {avg_metrics['f1']:.4f}")


    # Fit scaler and PCA on full training set
    scaler_final = StandardScaler()
    X_train_scaled = scaler_final.fit_transform(X_train_valid)
    X_val_scaled = scaler_final.transform(X_val_valid)
    X_test_scaled = scaler_final.transform(X_test_valid)

    n_components = min(10, len(feature_cols))
    pca_final = PCA(n_components=n_components)
    X_train_pca = pca_final.fit_transform(X_train_scaled)
    X_val_pca = pca_final.transform(X_val_scaled)
    X_test_pca = pca_final.transform(X_test_scaled)

    print(f"  PCA explained variance: {pca_final.explained_variance_ratio_.sum():.4f}")

    # Train final classifier with class balancing
    model_final = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    model_final.fit(X_train_pca, y_train_valid)

    # Evaluate on validation set
    y_val_pred = model_final.predict(X_val_pca)

    val_acc = accuracy_score(y_val_valid, y_val_pred)
    val_prec = precision_score(y_val_valid, y_val_pred, zero_division=0)
    val_rec = recall_score(y_val_valid, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val_valid, y_val_pred, zero_division=0)

    print(f"\nValidation set results:")
    print(f"Acc: {val_acc*100:.2f}% Prec: {val_prec*100:.2f}% Rec: {val_rec*100:.2f}% F1: {val_f1:.4f}")

    # Store results including standard errors
    results[horizon_name] = {
        'cv_acc': avg_metrics['acc'],
        'cv_acc_std': fold_acc_std,
        'cv_precision': avg_metrics['precision'],
        'cv_recall': avg_metrics['recall'],
        'cv_f1': avg_metrics['f1'],
        'cv_f1_std': fold_f1_std,
        'val_acc': val_acc,
        'val_precision': val_prec,
        'val_recall': val_rec,
        'val_f1': val_f1
    }

    # Store best model for final evaluation
    best_models[horizon_name] = {
        'model': model_final,
        'scaler': scaler_final,
        'pca': pca_final
    }


for horizon_name in horizon_names:

    # Use direction as target, filter out NaN
    y_test = clean_data[f'target_direction_{horizon_name}'].values[test_mask]

    test_valid_mask = ~np.isnan(y_test)
    test_indices = X_test.index[test_valid_mask]

    X_test_valid = X_test.loc[test_indices].reset_index(drop=True)
    y_test_valid = y_test[test_valid_mask]

    # Get saved model components
    model_final = best_models[horizon_name]['model']
    scaler_final = best_models[horizon_name]['scaler']
    pca_final = best_models[horizon_name]['pca']

    # Transform test set
    X_test_scaled = scaler_final.transform(X_test_valid)
    X_test_pca = pca_final.transform(X_test_scaled)

    # Predict
    y_test_pred = model_final.predict(X_test_pca)

    # Calculate classification metrics
    test_acc = accuracy_score(y_test_valid, y_test_pred)
    test_prec = precision_score(y_test_valid, y_test_pred, zero_division=0)
    test_rec = recall_score(y_test_valid, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test_valid, y_test_pred, zero_division=0)

    # Report sample size and metrics
    n_test = len(y_test_valid)

    print(f"{horizon_name:10s} - n={n_test:<5} Acc: {test_acc*100:5.2f}% Prec: {test_prec*100:5.2f}% Rec: {test_rec*100:5.2f}% F1: {test_f1:.4f}")

    # Update results with test metrics
    results[horizon_name]['n_test'] = n_test
    results[horizon_name]['test_acc'] = test_acc
    results[horizon_name]['test_precision'] = test_prec
    results[horizon_name]['test_recall'] = test_rec
    results[horizon_name]['test_f1'] = test_f1


confidence_results = {}

for horizon_name in horizon_names:
    if horizon_name not in best_models:
        continue

    print(f"\n{horizon_name} Horizon - Confidence Bins:")
    print("-" * 70)

    # Get model components
    model_final = best_models[horizon_name]['model']
    scaler_final = best_models[horizon_name]['scaler']
    pca_final = best_models[horizon_name]['pca']

    y_val = clean_data[f'target_direction_{horizon_name}'].values[val_mask]
    val_valid_mask = ~np.isnan(y_val)
    val_indices = X_val.index[val_valid_mask]

    X_val_valid = X_val.loc[val_indices].reset_index(drop=True)
    y_val_valid = y_val[val_valid_mask]

    X_val_scaled = scaler_final.transform(X_val_valid)
    X_val_pca = pca_final.transform(X_val_scaled)
    # For classification, use predict_proba to get confidence
    y_val_pred_proba = model_final.predict_proba(X_val_pca)
    # Confidence = max probability
    val_prediction_confidence = np.max(y_val_pred_proba, axis=1)

    # Make confidence bins based on validation set percentiles
    n_bins = 5
    confidence_percentiles = np.percentile(val_prediction_confidence, np.linspace(0, 100, n_bins + 1))

    # Apply fixed thresholds to the test set
    y_test = clean_data[f'target_direction_{horizon_name}'].values[test_mask]
    test_valid_mask = ~np.isnan(y_test)
    test_indices = X_test.index[test_valid_mask]

    X_test_valid = X_test.loc[test_indices].reset_index(drop=True)
    y_test_valid = y_test[test_valid_mask]

    X_test_scaled = scaler_final.transform(X_test_valid)
    X_test_pca = pca_final.transform(X_test_scaled)
    y_test_pred_proba = model_final.predict_proba(X_test_pca)
    y_test_pred = model_final.predict(X_test_pca)
    prediction_confidence = np.max(y_test_pred_proba, axis=1)

    # Classification accuracy and correctness
    correct = (y_test_pred == y_test_valid).astype(int)

    print(f"{'Bin':<5} {'Confidence Range':<25} {'N':<8} {'Accuracy':<12}")

    bin_results = []
    for i in range(n_bins):
        lower = confidence_percentiles[i]
        upper = confidence_percentiles[i + 1]

        if i == n_bins - 1:
            mask = (prediction_confidence >= lower) & (prediction_confidence <= upper)
        else:
            mask = (prediction_confidence >= lower) & (prediction_confidence < upper)

        if mask.sum() == 0:
            continue

        bin_accuracy = correct[mask].mean()

        conf_range = f"{lower:.6f} - {upper:.6f}"
        print(f"{i+1:<5} {conf_range:<25} {mask.sum():<8} {bin_accuracy*100:<12.2f}%")

        bin_results.append({
            'bin': i + 1,
            'n_samples': mask.sum(),
            'accuracy': bin_accuracy
        })

    # Check if confidence correlates with accuracy
    accuracies = [b['accuracy'] for b in bin_results]
    if len(accuracies) >= 3:
        improvement = accuracies[-1] - accuracies[0]

    confidence_results[horizon_name] = bin_results


# Plot classification metrics: Accuracy and F1 Score with error bars
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

horizons = list(results.keys())
test_accs = [results[h]['test_acc'] * 100 for h in horizons]
cv_accs = [results[h]['cv_acc'] * 100 for h in horizons]
cv_acc_stds = [results[h]['cv_acc_std'] * 100 for h in horizons]
test_f1s = [results[h]['test_f1'] for h in horizons]
cv_f1s = [results[h]['cv_f1'] for h in horizons]
cv_f1_stds = [results[h]['cv_f1_std'] for h in horizons]

# Convert horizon names to x positions for plotting
x_pos = np.arange(len(horizons))

# Accuracy (Test vs CV) with error bars
axes[0].errorbar(x_pos, cv_accs, yerr=cv_acc_stds, fmt='s--', linewidth=2, markersize=8,
                 color='lightgreen', alpha=0.7, label='Cross-Validation', capsize=5, capthick=2)
axes[0].plot(x_pos, test_accs, 'o-', linewidth=2.5, markersize=10, color='green', label='Test Set')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(horizons, rotation=45, ha='right')
axes[0].set_xlabel('Time Horizon', fontsize=12)
axes[0].set_ylabel('Accuracy (%)', fontsize=12)
axes[0].set_title('Accuracy: Classification Performance', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Baseline (50%)')
axes[0].legend(fontsize=10)

# F1 Score (Test vs CV) with error bars
axes[1].errorbar(x_pos, cv_f1s, yerr=cv_f1_stds, fmt='s--', linewidth=2, markersize=8,
                 color='plum', alpha=0.7, label='Cross-Validation', capsize=5, capthick=2)
axes[1].plot(x_pos, test_f1s, 'o-', linewidth=2.5, markersize=10, color='purple', label='Test Set')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(horizons, rotation=45, ha='right')
axes[1].set_xlabel('Time Horizon', fontsize=12)
axes[1].set_ylabel('F1 Score', fontsize=12)
axes[1].set_title('F1 Score: Precision-Recall Balance', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Baseline (0.5)')
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig('kalshi_prediction_CLASSIFICATION_results.png', dpi=150, bbox_inches='tight')
print("[OK] Saved classification visualization (Accuracy + F1 Score): kalshi_prediction_CLASSIFICATION_results.png")

# Plot confidence analysis
if confidence_results:

    for horizon_name, bins in confidence_results.items():
        # Create individual figure for this horizon
        fig_conf, ax = plt.subplots(1, 1, figsize=(8, 6))

        bin_nums = [b['bin'] for b in bins]
        accuracies = [b['accuracy'] * 100 for b in bins]
        n_samples = [b['n_samples'] for b in bins]

        # Plot bars
        bars = ax.bar(bin_nums, accuracies, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Color by accuracy
        for bar, acc in zip(bars, accuracies):
            if acc > 60:
                bar.set_color('darkgreen')
            elif acc > 55:
                bar.set_color('green')
            elif acc > 50:
                bar.set_color('lightgreen')
            elif acc > 45:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        # Add 50% reference line
        ax.axhline(y=50, color='black', linestyle='--', linewidth=2, alpha=0.6, label='Random (50%)')

        # Add sample counts on bars
        for i, (bar, n) in enumerate(zip(bars, n_samples)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'n={n}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xlabel('Confidence Bin (1=Low, 5=High)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
        ax.set_title(f'{horizon_name.upper()} - Confidence Analysis\nDoes Higher Confidence = Better Accuracy?',
                     fontsize=15, fontweight='bold', pad=15)
        ax.set_xticks(bin_nums)
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=11)

        plt.tight_layout()
        filename = f'kalshi_confidence_{horizon_name}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"  [OK] Saved: {filename}")
        plt.close(fig_conf)

    print(f"[OK] Created {len(confidence_results)} separate confidence analysis graphs")

# Use PCA from one of the final models
sample_horizon = horizon_names[0]
pca_final = best_models[sample_horizon]['pca']
n_components_final = pca_final.n_components_

# Aggregate feature importance across all components weighted by explained variance
overall_feature_importance = np.zeros(len(feature_cols))
for i in range(n_components_final):
    loadings = pca_final.components_[i]
    variance_weight = pca_final.explained_variance_ratio_[i]
    overall_feature_importance += np.abs(loadings) * variance_weight

# Find most predictive horizon (highest test accuracy)
best_predictive_horizon = max(results.keys(), key=lambda h: results[h]['test_acc'])
print(f"Most Predictive Horizon: {best_predictive_horizon} (Test Accuracy: {results[best_predictive_horizon]['test_acc']*100:.2f}%)")
print()

# Use PCA from the most predictive horizon
pca_best = best_models[best_predictive_horizon]['pca']
n_components_best = pca_best.n_components_

best_feature_importance = np.zeros(len(feature_cols))
for i in range(n_components_best):
    loadings = pca_best.components_[i]
    variance_weight = pca_best.explained_variance_ratio_[i]
    best_feature_importance += np.abs(loadings) * variance_weight

# Show overall most important original features
print(f"Overall Feature Importance for {best_predictive_horizon}")
feature_importance_pairs = [(feature_cols[i], best_feature_importance[i]) for i in range(len(feature_cols))]
feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)

for rank, (feature, importance) in enumerate(feature_importance_pairs[:10], 1):
    print(f"  {rank:2d}. {feature:30s}: {importance:.4f}")

# Create feature importance bar chart for most predictive horizon
fig_importance, ax_importance = plt.subplots(figsize=(12, 8))
top_20_features = feature_importance_pairs[:20]
feature_names = [f[0] for f in top_20_features]
feature_values = [f[1] for f in top_20_features]

ax_importance.barh(range(len(top_20_features)), feature_values, color='steelblue')
ax_importance.set_yticks(range(len(top_20_features)))
ax_importance.set_yticklabels(feature_names, fontsize=10)
ax_importance.set_xlabel('Feature Importance (PCA-weighted)', fontsize=12, fontweight='bold')
ax_importance.set_title(f'Top 20 Features - {best_predictive_horizon} Horizon',
                       fontsize=14, fontweight='bold')
ax_importance.invert_yaxis()
ax_importance.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
importance_filename = f'feature_importance_{best_predictive_horizon}_pca.png'
plt.savefig(importance_filename, dpi=300, bbox_inches='tight')
print(f"\n[OK] Feature importance chart saved to {importance_filename}")
print()

top_3_features = [f[0] for f in feature_importance_pairs[:3]]
top_3_importance = sum([f[1] for f in feature_importance_pairs[:3]])
total_importance = sum([f[1] for f in feature_importance_pairs])
concentration = (top_3_importance / total_importance * 100) if total_importance > 0 else 0

print()
print("Top contributing features for each principal component:")
print()

for i in range(min(3, n_components_final)):  # Show first 3 PCs
    print(f"Principal Component {i+1} which explains {pca_final.explained_variance_ratio_[i]*100:.2f}% variance:")

    loadings = pca_final.components_[i]
    feature_importance = [(feature_cols[j], abs(loadings[j])) for j in range(len(feature_cols))]
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for feature, importance in feature_importance[:5]:
        print(f"{feature:30s}: {importance:.4f}")
    print()


print("Test Set Performance:")
print()
print(f"{'Horizon':<10} {'n_test':<8} {'Test Acc':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
print("-" * 80)
for horizon in horizons:
    r = results[horizon]
    print(f"{horizon:<10} {r['n_test']:<8} {r['test_acc']*100:<11.2f}% {r['test_precision']*100:<11.2f}% {r['test_recall']*100:<11.2f}% {r['test_f1']:<12.4f}")

print()
print("\nCross-Validation Performance (Training Set):")
print()
print(f"{'Horizon':<10} {'CV Acc':<12} {'CV Prec':<12} {'CV Recall':<12} {'CV F1':<12}")
print("-" * 60)
for horizon in horizons:
    r = results[horizon]
    print(f"{horizon:<10} {r['cv_acc']*100:<11.2f}% {r['cv_precision']*100:<11.2f}% {r['cv_recall']*100:<11.2f}% {r['cv_f1']:<12.4f}")