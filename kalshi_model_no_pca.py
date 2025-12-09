"""
Kalshi Predictive Model - No PCA Version
Tests if raw Kalshi market features can predict Tesla stock movements
Compares multiple algorithms that handle feature interactions well:
- XGBoost: Gradient boosting with built-in feature interaction detection
- LightGBM: Fast gradient boosting optimized for high-dimensional data
- CatBoost: Handles categorical features and interactions naturally
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import catboost as cb


# Paths
STOCK_DATA_PATH = r"C:\Users\komal\PIC16B-Project-\normalized_tesla_stock_1min.csv"
MARKET_DATA_DIR = r"C:\Users\komal\PIC16B-Project-\market_data"

# Load Tesla stock data
stock_data = pd.read_csv(STOCK_DATA_PATH)

# Handle both 'timestamp' and 'datetime' column names
time_col = 'timestamp' if 'timestamp' in stock_data.columns else 'datetime'
stock_data['datetime'] = pd.to_datetime(stock_data[time_col])

# Load all Kalshi market data
market_files = [f for f in os.listdir(MARKET_DATA_DIR) if f.endswith('_historical_1min.csv')]

kalshi_dfs = []
for i, file in enumerate(market_files, 1):
    df = pd.read_csv(os.path.join(MARKET_DATA_DIR, file))
    kalshi_dfs.append(df)

kalshi_combined = pd.concat(kalshi_dfs, ignore_index=True)

# Prepare Kalshi data
kalshi_combined['datetime'] = pd.to_datetime(kalshi_combined['datetime'])

# Calculate time-to-expiry
kalshi_combined['datetime_naive'] = pd.to_datetime(kalshi_combined['datetime']).dt.tz_localize(None)

# Handle both close_time and expiration_time
time_col = 'close_time' if 'close_time' in kalshi_combined.columns else 'expiration_time'
kalshi_combined['close_time_naive'] = pd.to_datetime(kalshi_combined[time_col], errors='coerce').dt.tz_localize(None)

valid_data = kalshi_combined[kalshi_combined['close_time_naive'].notna()].copy()

# Time to expiry calculation
valid_data['ttm_hours'] = (
    (valid_data['close_time_naive'] - valid_data['datetime_naive']).dt.total_seconds() / 3600
).clip(lower=0.1)

# Price velocity features which is price signal strength per hour remaining
if 'yes_bid_close' in valid_data.columns:
    valid_data['bid_price_velocity'] = valid_data['yes_bid_close'] / valid_data['ttm_hours']
if 'yes_ask_close' in valid_data.columns:
    valid_data['ask_price_velocity'] = valid_data['yes_ask_close'] / valid_data['ttm_hours']

# Mid-price confidence
if 'yes_bid_close' in valid_data.columns and 'yes_ask_close' in valid_data.columns:
    valid_data['price_confidence'] = ((valid_data['yes_bid_close'] + valid_data['yes_ask_close']) / 2) / valid_data['ttm_hours']

# Market activity velocity
if 'open_interest' in valid_data.columns:
    valid_data['interest_velocity'] = valid_data['open_interest'] / valid_data['ttm_hours']
if 'volume' in valid_data.columns:
    valid_data['volume_velocity'] = valid_data['volume'] / valid_data['ttm_hours']

# Select features to keep per market
possible_features = [
    'yes_bid_close', 'yes_ask_close', 'volume', 'open_interest',
    'bid_price_velocity', 'ask_price_velocity', 'price_confidence',
    'interest_velocity', 'volume_velocity'
]
feature_cols_per_market = [col for col in possible_features if col in valid_data.columns]
feature_cols_per_market.append('ttm_hours')

# Pivot data: each market becomes separate columns
# Use market_ticker to identify each market uniquely
kalshi_wide = valid_data.pivot_table(
    index='datetime_naive',
    columns='market_ticker',
    values=feature_cols_per_market,
    aggfunc='first'
)

# Flatten multi-level columns
kalshi_wide.columns = [f'{ticker}_{feat}' for feat, ticker in kalshi_wide.columns]
kalshi_wide = kalshi_wide.reset_index()
kalshi_wide.rename(columns={'datetime_naive': 'datetime'}, inplace=True)

# Max 120 rows to avoid using outdated Kalshi prices
kalshi_features = kalshi_wide.ffill(limit=120).fillna(0)

# Merge with stock data
stock_data['datetime'] = stock_data['datetime'].dt.tz_localize(None)
merged_data = pd.merge(stock_data, kalshi_features, on='datetime', how='inner')

# Sort by datetime and reset index to ensure chronological order
merged_data = merged_data.sort_values('datetime').reset_index(drop=True)

min_time = merged_data['datetime'].min()
max_time = merged_data['datetime'].max()
continuous_timeline = pd.date_range(start=min_time, end=max_time, freq='1min')

# Reindex to continuous timeline
merged_data = merged_data.set_index('datetime').reindex(continuous_timeline)

tsla_spy_price_cols = [col for col in merged_data.columns if 'tsla_' in col.lower() or 'spy_' in col.lower()]
kalshi_cols = [col for col in merged_data.columns if col not in tsla_spy_price_cols and col != 'datetime']

merged_data[kalshi_cols] = merged_data[kalshi_cols].ffill(limit=120)
merged_data[kalshi_cols] = merged_data[kalshi_cols].fillna(0)

merged_data[tsla_spy_price_cols] = merged_data[tsla_spy_price_cols].ffill()
merged_data = merged_data.copy()

merged_data['tsla_spy_ratio'] = merged_data['tsla_close'] / merged_data['spy_close']

# Drop rows where stock prices are NaN
rows_before = len(merged_data)
merged_data = merged_data.dropna(subset=['tsla_close', 'spy_close', 'tsla_spy_ratio'])
rows_after = len(merged_data)

# Reset index to have datetime as column again
merged_data = merged_data.reset_index()
merged_data = merged_data.rename(columns={'index': 'datetime'})

# Create future datetime for 1hr ahead
merged_data['future_datetime_1hr'] = merged_data['datetime'] + pd.Timedelta(minutes=60)

# Merge with future TSLA/SPY ratios
future_ratios_1hr = merged_data[['datetime', 'tsla_spy_ratio']].rename(
    columns={'datetime': 'future_datetime', 'tsla_spy_ratio': 'future_ratio'}
)

temp_merge_1hr = pd.merge_asof(
    merged_data[['datetime', 'future_datetime_1hr', 'tsla_spy_ratio']].sort_values('future_datetime_1hr'),
    future_ratios_1hr.sort_values('future_datetime'),
    left_on='future_datetime_1hr',
    right_on='future_datetime',
    direction='forward',
    tolerance=pd.Timedelta(minutes=10)
)

temp_merge_1hr = temp_merge_1hr.sort_values('datetime').reset_index(drop=True)
merged_data['temp_return_1hr'] = (
    (temp_merge_1hr['future_ratio'] - merged_data['tsla_spy_ratio']) / merged_data['tsla_spy_ratio']
)

# Clean up temporary column
merged_data = merged_data.drop(columns=['future_datetime_1hr'])

# Get all Kalshi feature columns excluding stock data columns
all_feature_cols = [col for col in kalshi_features.columns if col != 'datetime']
stock_cols = [col for col in merged_data.columns if 'tsla_' in col or 'spy_' in col or col in ['datetime', 'temp_return_1hr']]
kalshi_feature_cols = [col for col in all_feature_cols if col not in stock_cols]

print(f"  Total Kalshi features before filtering: {len(kalshi_feature_cols)}")

# Calculate correlation with 1hr returns for each feature
import warnings
correlations = {}
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', 'invalid value encountered', RuntimeWarning)

    for col in kalshi_feature_cols:
        valid_data = merged_data[[col, 'temp_return_1hr']].dropna()
        if len(valid_data) > 100:
            corr = valid_data[col].corr(valid_data['temp_return_1hr'])
            correlations[col] = abs(corr)
        else:
            correlations[col] = 0.0

# Keep features with correlation over the threshold
correlation_threshold = 0.001
keep_keywords = ['velocity', 'price', 'bid', 'ask', 'confidence', 'open_interest', 'volume']

features_to_keep = []
for col in kalshi_feature_cols:
    has_useful_keyword = any(keyword in col.lower() for keyword in keep_keywords)
    is_correlated = correlations.get(col, 0) > correlation_threshold

    if has_useful_keyword or is_correlated:
        features_to_keep.append(col)

# Show top correlated features
top_correlated = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:20]
print(f"  Top 20 most correlated per-market features:")
for feat, corr in top_correlated:
    parts = feat.split('_', 1)
    market_ticker = parts[0] if len(parts) > 0 else 'unknown'
    feature_name = parts[1] if len(parts) > 1 else feat
    print(f"    {feat[:60]:60s}: {corr:.6f}  [{market_ticker}]")

print(f"\n  Per-market features kept: {len(features_to_keep)} out of {len(kalshi_feature_cols)}")
print(f"  Per-market features removed: {len(kalshi_feature_cols) - len(features_to_keep)}")

# Count unique markets in kept features
unique_markets_kept = set()
for feat in features_to_keep:
    market_ticker = feat.split('_', 1)[0]
    unique_markets_kept.add(market_ticker)
print(f"  Unique markets contributing: {len(unique_markets_kept)} out of 61")

# Filter feature columns to match kept features
feature_cols = features_to_keep

# Remove temporary return column
merged_data = merged_data.drop(columns=['temp_return_1hr'])

# Create targets for multiple time horizons
time_horizons = {
    '5min': 5,
    '10min': 10,
    '15min': 15,
    '30min': 30,
    '1hr': 60,
    '6hr': 360,
    '12hr': 720,
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
    merged_data[f'future_datetime_{name}'] = merged_data['datetime'] + pd.Timedelta(minutes=horizon)

    # Merge with future TSLA/SPY ratio
    future_ratios = merged_data[['datetime', 'tsla_spy_ratio']].copy()
    future_ratios = future_ratios.rename(columns={'datetime': 'future_datetime', 'tsla_spy_ratio': 'future_ratio'})

    tolerance_minutes = min(30, max(5, horizon // 10))
    temp_merge = pd.merge_asof(
        merged_data[['datetime', f'future_datetime_{name}', 'tsla_spy_ratio']].sort_values(f'future_datetime_{name}'),
        future_ratios.sort_values('future_datetime'),
        left_on=f'future_datetime_{name}',
        right_on='future_datetime',
        direction='forward',  # ONLY look forward in time
        tolerance=pd.Timedelta(minutes=tolerance_minutes)
    )

    # Sort back to original order
    temp_merge = temp_merge.sort_values('datetime').reset_index(drop=True)
    merged_data[f'target_return_{name}'] = (
        (temp_merge['future_ratio'] - merged_data['tsla_spy_ratio']) / merged_data['tsla_spy_ratio']
    )

    # Only classify if |return| > 0.5%
    NOISE_THRESHOLD = 0.005

    # Create binary classification: 1 = UP (return > +0.5%), 0 = DOWN (return < -0.5%)
    merged_data[f'target_direction_{name}'] = np.nan  # Default to NaN
    merged_data.loc[merged_data[f'target_return_{name}'] > NOISE_THRESHOLD, f'target_direction_{name}'] = 1.0 
    merged_data.loc[merged_data[f'target_return_{name}'] < -NOISE_THRESHOLD, f'target_direction_{name}'] = 0.0

    # Clean up temporary column
    merged_data = merged_data.drop(columns=[f'future_datetime_{name}'])


# Prepare features (X) and targets (y)
X = merged_data[feature_cols]
print(f"Using {len(feature_cols)} filtered features for modeling")
max_horizon = max(time_horizons.values())
max_horizon_td = pd.Timedelta(minutes=max_horizon)

# Get time range
min_time = merged_data['datetime'].min()
max_time = merged_data['datetime'].max()
total_duration = max_time - min_time

# Split by time: 70% train, 15% val, 15% test with time gaps
train_end_time = min_time + (total_duration * 0.70) - max_horizon_td
val_start_time = train_end_time + max_horizon_td
val_end_time = min_time + (total_duration * 0.85) - max_horizon_td
test_start_time = val_end_time + max_horizon_td

# Create masks based on timestamps
train_mask = merged_data['datetime'] <= train_end_time
val_mask = (merged_data['datetime'] >= val_start_time) & (merged_data['datetime'] <= val_end_time)
test_mask = merged_data['datetime'] >= test_start_time

X_train_full = X[train_mask]
X_val = X[val_mask]
X_test = X[test_mask]

print(f"Total timeline: {total_duration}")
print(f"Train: {len(X_train_full):,} samples ({train_end_time})")
print(f"  Gap: {max_horizon} minutes")
print(f"Val:   {len(X_val):,} samples ({val_start_time} to {val_end_time})")
print(f"  Gap: {max_horizon} minutes")
print(f"Test:  {len(X_test):,} samples ({test_start_time} onwards)")

# Define models to test with better regularization
models_to_test = {}


models_to_test['XGBoost'] = lambda: xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbosity=0,
    enable_categorical=False,
    device='cuda:0'
)


models_to_test['LightGBM'] = lambda: lgb.LGBMClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    min_child_samples=20,  # Prevent splits with too few samples
    min_split_gain=0.0,
    random_state=42,
    verbosity=-1,
    device='cpu',
    n_jobs=-1
)


models_to_test['CatBoost'] = lambda: cb.CatBoostClassifier(
    iterations=200,
    depth=5,
    learning_rate=0.05,
    l2_leaf_reg=3,
    random_state=42,
    verbose=0,
    allow_writing_files=False,
    task_type='GPU',
    devices='0'
)

# Helper function to scale features while preserving DataFrame structure
def scale_features(scaler, X, fit=False):
    """Scale features and return as DataFrame with preserved column names and index"""
    data = scaler.fit_transform(X) if fit else scaler.transform(X)
    return pd.DataFrame(data, columns=X.columns, index=X.index)

# Time series cross-validation on training data
tscv = TimeSeriesSplit(n_splits=5)

# Store results for each model
all_results = {model_name: [] for model_name in models_to_test.keys()}
all_val_results = {model_name: [] for model_name in models_to_test.keys()}
all_test_results = {model_name: [] for model_name in models_to_test.keys()}
best_models = {model_name: {} for model_name in models_to_test.keys()}

for name, horizon in time_horizons.items():
    print()
    print(f"Training for {name} horizon")
    print()

    # Get targets for all splits
    y_direction_all = merged_data[f'target_direction_{name}'].values

    # Split targets
    y_train = y_direction_all[train_mask]
    y_val = y_direction_all[val_mask]
    y_test = y_direction_all[test_mask]

    # Remove NaN targets from each set
    train_valid_mask = ~np.isnan(y_train)
    val_valid_mask = ~np.isnan(y_val)
    test_valid_mask = ~np.isnan(y_test)

    train_indices = X_train_full.index[train_valid_mask]
    val_indices = X_val.index[val_valid_mask]
    test_indices = X_test.index[test_valid_mask]

    # Reset index to ensure X and y alignment
    X_train_valid = X_train_full.loc[train_indices].reset_index(drop=True)
    y_train_valid = y_train[train_valid_mask]

    X_val_valid = X_val.loc[val_indices].reset_index(drop=True)
    y_val_valid = y_val[val_valid_mask]

    X_test_valid = X_test.loc[test_indices].reset_index(drop=True)
    y_test_valid = y_test[test_valid_mask]

    print(f"Train samples: {len(X_train_valid):,}")
    print(f"Val samples: {len(X_val_valid):,}")
    print(f"Test samples: {len(X_test_valid):,}")

    # Check if we have enough samples in test set
    if len(X_test_valid) < 10:
        print(f"  [SKIP] Not enough test samples ({len(X_test_valid)}) - horizon too long for available data")
        print()
        continue

    print()

    # Test each model
    for model_name, model_func in models_to_test.items():
        print(f"  Testing {model_name}...")

        # Step 1: Cross-validation on training set
        fold_acc = []
        fold_precision = []
        fold_recall = []
        fold_f1 = []
        feature_importances = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_train_valid), 1):
            gap_rows = horizon
            if len(train_idx) > gap_rows:
                train_idx_trimmed = train_idx[:-gap_rows]
            else:
                # Fold too small, skip
                continue

            X_fold_train = X_train_valid.iloc[train_idx_trimmed]
            X_fold_test = X_train_valid.iloc[test_idx]
            y_fold_train = y_train_valid[train_idx_trimmed]
            y_fold_test = y_train_valid[test_idx]

            # Scale features in fold
            scaler = StandardScaler()
            X_fold_train_scaled = scale_features(scaler, X_fold_train, fit=True)
            X_fold_test_scaled = scale_features(scaler, X_fold_test)

            # Train classifier
            try:
                model = model_func()
                model.fit(X_fold_train_scaled, y_fold_train)

                # Predict
                y_pred = model.predict(X_fold_test_scaled)

                # Evaluate classification metrics
                acc = accuracy_score(y_fold_test, y_pred)
                prec = precision_score(y_fold_test, y_pred, zero_division=0)
                rec = recall_score(y_fold_test, y_pred, zero_division=0)
                f1 = f1_score(y_fold_test, y_pred, zero_division=0)

                fold_acc.append(acc)
                fold_precision.append(prec)
                fold_recall.append(rec)
                fold_f1.append(f1)

                if hasattr(model, 'feature_importances_'):
                    feature_importances.append(model.feature_importances_)
            except Exception as e:
                print(f"    [WARN] Fold {fold} failed: {str(e)[:100]}... Skipping fold.")
                continue

        # Average CV results (with guard for empty folds)
        if len(fold_acc) == 0:
            # No folds ran - horizon too long for available training data
            print(f"    [SKIP] No CV folds completed (horizon {horizon} min too long for training data)")
            print(f"           Skipping this horizon - not enough data after gap trimming")
            continue

        cv_acc = np.mean(fold_acc)
        cv_acc_std = np.std(fold_acc)
        cv_precision = np.mean(fold_precision)
        cv_precision_std = np.std(fold_precision)
        cv_recall = np.mean(fold_recall)
        cv_recall_std = np.std(fold_recall)
        cv_f1 = np.mean(fold_f1)
        cv_f1_std = np.std(fold_f1)
        avg_importance = np.mean(feature_importances, axis=0) if feature_importances else None

        print(f"    CV - Acc: {cv_acc*100:.2f}% ±{cv_acc_std*100:.2f}%, Prec: {cv_precision*100:.2f}% ±{cv_precision_std*100:.2f}%, Rec: {cv_recall*100:.2f}% ±{cv_recall_std*100:.2f}%, F1: {cv_f1:.4f} ±{cv_f1_std:.4f} ({len(fold_acc)} folds)")

        # Show top 5 features for this model/horizon
        if avg_importance is not None:
            top_5_idx = np.argsort(avg_importance)[-5:][::-1]
            top_5_features = [f"{feature_cols[i]} ({avg_importance[i]:.1f})" for i in top_5_idx]
            print(f"    Top 5: {', '.join(top_5_features)}")

        # Step 2: Train final classifier on full training set, evaluate on validation
        scaler_final = StandardScaler()
        X_train_scaled = scale_features(scaler_final, X_train_valid, fit=True)
        X_val_scaled = scale_features(scaler_final, X_val_valid)
        X_test_scaled = scale_features(scaler_final, X_test_valid)

        # Train final model with error handling
    
        model_final = model_func()
        model_final.fit(X_train_scaled, y_train_valid)

        # Validation set evaluation
        y_val_pred = model_final.predict(X_val_scaled)
        val_acc = accuracy_score(y_val_valid, y_val_pred)
        val_precision = precision_score(y_val_valid, y_val_pred, zero_division=0)
        val_recall = recall_score(y_val_valid, y_val_pred, zero_division=0)
        val_f1 = f1_score(y_val_valid, y_val_pred, zero_division=0)

        print(f"    Val - Acc: {val_acc*100:.2f}%, Prec: {val_precision*100:.2f}%, Rec: {val_recall*100:.2f}%, F1: {val_f1:.4f}")

        y_test_pred = model_final.predict(X_test_scaled)
        y_test_pred_proba = model_final.predict_proba(X_test_scaled)

        test_acc = accuracy_score(y_test_valid, y_test_pred)
        test_precision = precision_score(y_test_valid, y_test_pred, zero_division=0)
        test_recall = recall_score(y_test_valid, y_test_pred, zero_division=0)
        test_f1 = f1_score(y_test_valid, y_test_pred, zero_division=0)

        # 1. Confusion Matrix
        TP = np.sum((y_test_pred == 1) & (y_test_valid == 1))
        FP = np.sum((y_test_pred == 1) & (y_test_valid == 0))
        TN = np.sum((y_test_pred == 0) & (y_test_valid == 0))
        FN = np.sum((y_test_pred == 0) & (y_test_valid == 1))

        # 2. Class Distribution
        n_positive = np.sum(y_test_valid == 1)
        n_negative = np.sum(y_test_valid == 0)

        # 3. Prediction Confidence
        prob_positive_class = y_test_pred_proba[:, 1]  # Probability of class 1
        prob_negative_class = y_test_pred_proba[:, 0]  # Probability of class 0

        # Mean confidence for each predicted class
        pos_pred_mask = y_test_pred == 1
        neg_pred_mask = y_test_pred == 0

        mean_conf_pos = prob_positive_class[pos_pred_mask].mean() if pos_pred_mask.sum() > 0 else 0.0
        mean_conf_neg = prob_negative_class[neg_pred_mask].mean() if neg_pred_mask.sum() > 0 else 0.0
        std_conf = prob_positive_class.std()

        # Store all results
        all_results[model_name].append({
            'horizon': name,
            'minutes': horizon,
            'cv_acc': cv_acc,
            'cv_acc_std': cv_acc_std,
            'cv_precision': cv_precision,
            'cv_precision_std': cv_precision_std,
            'cv_recall': cv_recall,
            'cv_recall_std': cv_recall_std,
            'cv_f1': cv_f1,
            'cv_f1_std': cv_f1_std,
            'feature_importance': avg_importance
        })

        all_val_results[model_name].append({
            'horizon': name,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1
        })

        all_test_results[model_name].append({
            'horizon': name,
            'test_acc': test_acc,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1
        })

        # Output diagnostics
        print(f"    Horizon: {name}")
        print(f"    True Positive={TP}, False Poitive={FP}, True negative={TN}, False negative={FN}")
        print(f"    n_positive={n_positive}, n_negative={n_negative}")
        print(f"    test_acc={test_acc:.4f}, test_precision={test_precision:.4f}, test_recall={test_recall:.4f}, test_f1={test_f1:.4f}")
        print(f"    mean_conf_pos={mean_conf_pos:.4f}, mean_conf_neg={mean_conf_neg:.4f}, std_conf={std_conf:.4f}")

        # Store best model for this horizon
        if name not in best_models[model_name]:
            best_models[model_name][name] = {
                'model': model_final,
                'scaler': scaler_final
            }

# Helper function to parse feature names
def parse_feature_name(feat):
    """Parse feature name into market ticker and feature type"""
    parts = feat.split('_', 1)
    if len(parts) == 2:
        market_ticker = parts[0]
        feature_name = parts[1]
        return market_ticker, feature_name
    return feat, ''

# Generate feature importance charts for all models at their best time ranges
for model_name in models_to_test.keys():
    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")

    # Find best time range for this model
    test_results = all_test_results[model_name]
    best_predictive_idx = np.argmax([r['test_acc'] for r in test_results])
    cv_results = all_results[model_name]
    best_predictive_result = cv_results[best_predictive_idx]

    if best_predictive_result['feature_importance'] is not None:
        print(f"Top 20 Features for {best_predictive_result['horizon']} horizon ({model_name}) - MOST PREDICTIVE:")
        print()

        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': best_predictive_result['feature_importance']
        }).sort_values('importance', ascending=False)

        importance_df['market'] = importance_df['feature'].apply(lambda x: parse_feature_name(x)[0])
        importance_df['feature_type'] = importance_df['feature'].apply(lambda x: parse_feature_name(x)[1])

        print(f"{'Rank':<5} {'Market':<25} {'Feature':<25} {'Importance':<12}")
        for rank, (i, row) in enumerate(importance_df.head(20).iterrows(), 1):
            market = row['market'][:24]
            feat_type = row['feature_type'][:24]
            print(f"{rank:<5} {market:<25} {feat_type:<25} {row['importance']:<12.6f}")
        print()

        # Create feature importance bar chart with market names
        fig_importance, ax_importance = plt.subplots(figsize=(14, 10))
        top_20_features = importance_df.head(20)

        # Create labels that show both market and feature
        labels = []
        for _, row in top_20_features.iterrows():
            market = row['market'][:20]
            feat = row['feature_type'][:25]
            labels.append(f"{market}: {feat}")

        ax_importance.barh(range(len(top_20_features)), top_20_features['importance'].values, color='steelblue')
        ax_importance.set_yticks(range(len(top_20_features)))
        ax_importance.set_yticklabels(labels, fontsize=9)
        ax_importance.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        ax_importance.set_title(f'Top 20 Per-Market Features - {best_predictive_result["horizon"]} Horizon ({model_name})\nMost Predictive Time Period',
                               fontsize=14, fontweight='bold')
        ax_importance.invert_yaxis()
        ax_importance.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        importance_filename = f'feature_importance_{best_predictive_result["horizon"]}_{model_name.lower()}_no_pca.png'
        plt.savefig(importance_filename, dpi=300, bbox_inches='tight')
        
        print()
        print("Market Contribution Summary:")
        print()
        market_importance = importance_df.groupby('market')['importance'].sum().sort_values(ascending=False)
        print(f"{'Market':<30} {'Total Importance':<20} {'% of Total':<12}")
        total_importance = market_importance.sum()
        for market, importance in market_importance.head(10).items():
            pct = (importance / total_importance * 100) if total_importance > 0 else 0
            print(f"{market[:29]:<30} {importance:<20.6f} {pct:<12.2f}%")
        print()

print("Final test set results:")
print()

# Determine best model based on average test accuracy across all horizons
avg_test_accs = {
    model_name: np.mean([r['test_acc'] for r in all_test_results[model_name]])
    for model_name in models_to_test.keys()
}
best_model_name = max(avg_test_accs, key=avg_test_accs.get)

# Display test set results for best model
test_results = all_test_results[best_model_name]
print(f"Best Model: {best_model_name} (avg test accuracy: {avg_test_accs[best_model_name]*100:.2f}%)")
print()
print(f"{'Horizon':<10} {'Test Acc':<10} {'Precision':<12} {'Recall':<10} {'F1 Score':<12}")
for result in test_results:
    print(f"{result['horizon']:<10} {result['test_acc']*100:<9.2f}% {result['test_precision']*100:<11.2f}% {result['test_recall']*100:<9.2f}% {result['test_f1']:<12.4f}")

print()
print("\nCross-Validation Results on Training Set:")
print(f"{'Horizon':<10} {'CV Acc':<10} {'CV Prec':<12} {'CV Recall':<12} {'CV F1':<12}")
cv_results = all_results[best_model_name]
for result in cv_results:
    print(f"{result['horizon']:<10} {result['cv_acc']*100:<9.2f}% {result['cv_precision']*100:<11.2f}% {result['cv_recall']*100:<11.2f}% {result['cv_f1']:<12.4f}")

# Save results for all models
print()
for model_name in models_to_test.keys():
    # Combine all results for this model
    combined_results = []
    for i, result in enumerate(all_results[model_name]):
        combined_results.append({
            'horizon': result['horizon'],
            'cv_acc': result['cv_acc'],
            'cv_acc_std': result['cv_acc_std'],
            'cv_precision': result['cv_precision'],
            'cv_precision_std': result['cv_precision_std'],
            'cv_recall': result['cv_recall'],
            'cv_recall_std': result['cv_recall_std'],
            'cv_f1': result['cv_f1'],
            'cv_f1_std': result['cv_f1_std'],
            'val_acc': all_val_results[model_name][i]['val_acc'],
            'val_precision': all_val_results[model_name][i]['val_precision'],
            'val_recall': all_val_results[model_name][i]['val_recall'],
            'val_f1': all_val_results[model_name][i]['val_f1'],
            'test_acc': all_test_results[model_name][i]['test_acc'],
            'test_precision': all_test_results[model_name][i]['test_precision'],
            'test_recall': all_test_results[model_name][i]['test_recall'],
            'test_f1': all_test_results[model_name][i]['test_f1']
        })

    model_df = pd.DataFrame(combined_results)
    filename = f'kalshi_model_no_pca_results_{model_name.lower()}.csv'
    model_df.to_csv(filename, index=False)

for model_name in models_to_test.keys():
    print(f"{model_name}:")

    for result in all_results[model_name]:
        horizon = result['horizon']
        importance = result['feature_importance']

        if importance is not None and len(importance) > 0:
            # Get top 3 features
            top_3_idx = np.argsort(importance)[-3:][::-1]
            top_3_names = [feature_cols[i] for i in top_3_idx]
            top_3_values = importance[top_3_idx]

            # Calculate % in top 3
            total_importance = importance.sum()
            if total_importance > 0:
                concentration = top_3_values.sum() / total_importance * 100
            else:
                concentration = 0

            # Print feature importance summary
            print(f"  {horizon:6s}: {top_3_names[0]:25s} ({top_3_values[0]:6.1f}) | "
                  f"{top_3_names[1]:25s} ({top_3_values[1]:6.1f}) | "
                  f"{top_3_names[2]:25s} ({top_3_values[2]:6.1f}) | "
                  f"Top-3: {concentration:.1f}%")

            # Add warnings for problematic patterns
            warnings = []

            # Check for time-to-market dominance
            if any('ttm' in name.lower() for name in top_3_names[:1]):
                warnings.append("Relying on time-to-market features")

            # Check for over-concentration
            if concentration > 80:
                warnings.append(f"Over-reliance on few features ({concentration:.0f}% in top 3)")

            # Check for very low importance
            if top_3_values[0] < 1.0:
                warnings.append(f"Very low feature importance")

            # Print warnings
            for warning in warnings:
                print(f"{warning}")

    print()

print()
print("Confidence analysis")
print()

all_confidence_results = {}

for model_name in models_to_test.keys():
    print()
    print(f"MODEL: {model_name}")
    print()

    model_confidence_results = {}

    # Get horizons for this model
    horizons_for_model = [r['horizon'] for r in all_test_results[model_name]]

    for horizon_name in horizons_for_model:
        if horizon_name not in best_models[model_name]:
            continue

        print(f"\n{horizon_name} Horizon - Confidence Bins:")

        # Get validation data to establish confidence thresholds using TIME-BASED masks
        y_direction_all = merged_data[f'target_direction_{horizon_name}'].values

        y_val = y_direction_all[val_mask]
        y_test = y_direction_all[test_mask]

        # Get model and scaler
        model_final = best_models[model_name][horizon_name]['model']
        scaler_final = best_models[model_name][horizon_name]['scaler']

        # Calculate confidence thresholds on validation set using predict_proba
        val_valid_mask = ~np.isnan(y_val)
        val_indices = X_val.index[val_valid_mask]
        X_val_valid = X_val.loc[val_indices].reset_index(drop=True)

        X_val_scaled = scale_features(scaler_final, X_val_valid)
        y_val_pred_proba = model_final.predict_proba(X_val_scaled)
        # Confidence = max probability
        val_prediction_confidence = np.max(y_val_pred_proba, axis=1)

        # Define confidence bins based on VALIDATION SET percentiles
        n_bins = 5
        confidence_percentiles = np.percentile(val_prediction_confidence, np.linspace(0, 100, n_bins + 1))

        # Apply fixed thresholds to TEST SET
        test_valid_mask = ~np.isnan(y_test)
        test_indices = X_test.index[test_valid_mask]
        X_test_valid = X_test.loc[test_indices].reset_index(drop=True)
        y_test_valid = y_test[test_valid_mask]

        X_test_scaled = scale_features(scaler_final, X_test_valid)
        y_test_pred_proba = model_final.predict_proba(X_test_scaled)
        y_test_pred = model_final.predict(X_test_scaled)
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
        model_confidence_results[horizon_name] = bin_results

    all_confidence_results[model_name] = model_confidence_results

print()

# Get horizons from any model
horizons = [r['horizon'] for r in all_results[list(models_to_test.keys())[0]]]

# Create comprehensive comparison plot
n_models = len(all_results)
fig = plt.figure(figsize=(16, 10))

# Test Accuracy comparison
ax1 = plt.subplot(2, 2, 1)
for model_name in models_to_test.keys():
    test_results = all_test_results[model_name]
    test_accs = [r['test_acc'] * 100 for r in test_results]
    ax1.plot(range(len(test_accs)), test_accs, marker='o', linewidth=2,
             markersize=6, label=model_name, alpha=0.8)
ax1.axhline(y=50, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='Baseline (50%)')
ax1.set_xlabel('Time Horizon', fontsize=11)
ax1.set_ylabel('Test Accuracy (%)', fontsize=11)
ax1.set_title('Test Accuracy Comparison Across Models\n', fontsize=12, fontweight='bold')
ax1.set_xticks(range(len(horizons)))
ax1.set_xticklabels(horizons, rotation=45, ha='right', fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9)

# Test F1 Score comparison
ax2 = plt.subplot(2, 2, 2)
for model_name in models_to_test.keys():
    test_results = all_test_results[model_name]
    test_f1s = [r['test_f1'] for r in test_results]
    ax2.plot(range(len(test_f1s)), test_f1s, marker='s',
             linewidth=2, markersize=6, label=model_name, alpha=0.8)
ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='Baseline (0.5)')
ax2.set_xlabel('Time Horizon', fontsize=11)
ax2.set_ylabel('Test F1 Score', fontsize=11)
ax2.set_title('Test F1 Score Comparison Across Models\n', fontsize=12, fontweight='bold')
ax2.set_xticks(range(len(horizons)))
ax2.set_xticklabels(horizons, rotation=45, ha='right', fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9)


plt.tight_layout()
plot_filename = 'kalshi_model_no_pca_comparison.png'
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')

for model_name in models_to_test.keys():
    fig_model = plt.figure(figsize=(16, 10))

    # CV vs Test Accuracy with error bars
    ax1 = plt.subplot(2, 2, 1)
    cv_accs = [r['cv_acc'] * 100 for r in all_results[model_name]]
    cv_acc_stds = [r['cv_acc_std'] * 100 for r in all_results[model_name]]
    test_accs = [r['test_acc'] * 100 for r in all_test_results[model_name]]
    x_pos = range(len(cv_accs))
    ax1.errorbar(x_pos, cv_accs, yerr=cv_acc_stds, marker='s', linewidth=2.5,
                 markersize=8, color='lightblue', alpha=0.8, label='CV Mean ± Std', capsize=5, capthick=2)
    ax1.plot(x_pos, test_accs, marker='o', linewidth=2.5,
             markersize=8, color='blue', alpha=0.8, label='Test')
    ax1.axhline(y=50, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='Baseline (50%)')
    ax1.set_xlabel('Time Horizon', fontsize=11)
    ax1.set_ylabel('Accuracy (%)', fontsize=11)
    ax1.set_title(f'{model_name} - CV vs Test Accuracy\n(Classification, No PCA)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(horizons)))
    ax1.set_xticklabels(horizons, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    # Precision vs Recall with CV error bars
    ax2 = plt.subplot(2, 2, 2)
    cv_precision = [r['cv_precision'] * 100 for r in all_results[model_name]]
    cv_precision_std = [r['cv_precision_std'] * 100 for r in all_results[model_name]]
    cv_recall = [r['cv_recall'] * 100 for r in all_results[model_name]]
    cv_recall_std = [r['cv_recall_std'] * 100 for r in all_results[model_name]]
    x_pos = range(len(cv_precision))
    ax2.errorbar(x_pos, cv_precision, yerr=cv_precision_std, marker='o', linewidth=2,
                 markersize=6, label='CV Precision ± Std', color='green', alpha=0.8, capsize=4, capthick=1.5)
    ax2.errorbar(x_pos, cv_recall, yerr=cv_recall_std, marker='s', linewidth=2,
                 markersize=6, label='CV Recall ± Std', color='orange', alpha=0.8, capsize=4, capthick=1.5)
    ax2.set_xlabel('Time Horizon', fontsize=11)
    ax2.set_ylabel('Score (%)', fontsize=11)
    ax2.set_title(f'{model_name} - CV Precision vs Recall\n(with Variability)', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(horizons)))
    ax2.set_xticklabels(horizons, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    # Test Precision vs Recall
    ax3 = plt.subplot(2, 2, 3)
    test_precision = [r['test_precision'] * 100 for r in all_test_results[model_name]]
    test_recall = [r['test_recall'] * 100 for r in all_test_results[model_name]]
    x_pos = range(len(test_precision))
    ax3.plot(x_pos, test_precision, marker='o', linewidth=2,
             markersize=6, label='Test Precision', color='darkgreen', alpha=0.8)
    ax3.plot(x_pos, test_recall, marker='s', linewidth=2,
             markersize=6, label='Test Recall', color='darkorange', alpha=0.8)
    ax3.set_xlabel('Time Horizon', fontsize=11)
    ax3.set_ylabel('Score (%)', fontsize=11)
    ax3.set_title(f'{model_name} - Test Precision vs Recall', fontsize=12, fontweight='bold')
    ax3.set_xticks(range(len(horizons)))
    ax3.set_xticklabels(horizons, rotation=45, ha='right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)

    # Plot 4: Test F1 Score
    ax4 = plt.subplot(2, 2, 4)
    test_f1s = [r['test_f1'] for r in all_test_results[model_name]]
    x_pos = range(len(test_f1s))
    ax4.plot(x_pos, test_f1s, marker='D', linewidth=2.5,
             markersize=8, color='purple', alpha=0.8, label='Test F1')
    ax4.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='Baseline (0.5)')
    ax4.set_xlabel('Time Horizon', fontsize=11)
    ax4.set_ylabel('F1 Score', fontsize=11)
    ax4.set_title(f'{model_name} - Test F1 Score', fontsize=12, fontweight='bold')
    ax4.set_xticks(range(len(horizons)))
    ax4.set_xticklabels(horizons, rotation=45, ha='right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9)

    plt.tight_layout()
    individual_plot_filename = f'kalshi_model_no_pca_{model_name.lower()}_detailed.png'
    plt.savefig(individual_plot_filename, dpi=300, bbox_inches='tight')

print()

# Create confidence analysis plots for all models
if all_confidence_results:
    for model_name, confidence_data in all_confidence_results.items():
        if not confidence_data:
            continue

        n_horizons = len(confidence_data)

        # Create grid layout
        nrows, ncols = 2, 7
        fig_conf, axes_conf = plt.subplots(nrows, ncols, figsize=(28, 8))
        axes_conf = axes_conf.flatten()

        # Plot all horizons
        for idx, (horizon_name, bins) in enumerate(confidence_data.items()):
            ax = axes_conf[idx]

            bin_nums = [b['bin'] for b in bins]
            accuracies = [b['accuracy'] * 100 for b in bins]
            n_samples = [b['n_samples'] for b in bins]

            # Plot bars
            bars = ax.bar(bin_nums, accuracies, alpha=0.7, edgecolor='black')

            # Color by accuracy
            for bar, acc in zip(bars, accuracies):
                if acc > 55:
                    bar.set_color('darkgreen')
                elif acc > 50:
                    bar.set_color('green')
                else:
                    bar.set_color('red')

            # Add 50% reference line
            ax.axhline(y=50, color='black', linestyle='--', linewidth=1.5, alpha=0.5)


            # Add sample counts on bars
            for i, (bar, n) in enumerate(zip(bars, n_samples)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'n={n}', ha='center', va='bottom', fontsize=8)

            ax.set_xlabel('Confidence Bin (1=Low, 5=High)', fontsize=9)
            ax.set_ylabel('Direction Accuracy (%)', fontsize=9)
            ax.set_title(f'{horizon_name}\nAccuracy by Confidence', fontsize=10, fontweight='bold')
            ax.set_xticks(bin_nums)
            ax.set_ylim([0, 100])
            ax.grid(True, alpha=0.3, axis='y')

        plt.suptitle(f'Confidence Analysis: {model_name}', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        conf_filename = f'kalshi_confidence_{model_name.lower()}_all_horizons.png'
        plt.savefig(conf_filename, dpi=300, bbox_inches='tight')


print("Key Findings (Test Set):")
test_results_best = all_test_results[best_model_name]
best_test_acc_idx = np.argmax([r['test_acc'] for r in test_results_best])
best_test_f1_idx = np.argmax([r['test_f1'] for r in test_results_best])
print(f"  - Best Test Accuracy: {test_results_best[best_test_acc_idx]['test_acc']*100:.2f}% at {test_results_best[best_test_acc_idx]['horizon']} horizon")
print(f"  - Best Test F1 Score: {test_results_best[best_test_f1_idx]['test_f1']:.4f} at {test_results_best[best_test_f1_idx]['horizon']} horizon")
