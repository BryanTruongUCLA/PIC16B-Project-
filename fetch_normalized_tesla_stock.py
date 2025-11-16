"""
Fetch Normalized Tesla Stock Data
Downloads 1-minute Tesla and S&P 500 data from EODHD API, normalized for beta calculation
Only fetches data for the time range that exists in the downloaded market data
"""

import pandas as pd
import os
from datetime import datetime, timedelta
import requests
import matplotlib.pyplot as plt

# Configuration
EODHD_API_KEY = "API-Key-Here"
MARKET_DATA_DIR = "market_data"
OUTPUT_FILE = "normalized_tesla_stock_1min.csv"
VISUALIZATION_FILE = "normalized_tesla_stock_1min.png"


def get_market_data_timerange() -> tuple:
    """
    Scan all market data CSVs to find the min and max datetime range

    Returns:
        (start_datetime, end_datetime) tuple
    """
    print("Scanning market data to determine time range...")

    if not os.path.exists(MARKET_DATA_DIR):
        print(f"  [ERROR] {MARKET_DATA_DIR} directory not found")
        return None, None

    all_datetimes = []
    csv_files = [f for f in os.listdir(MARKET_DATA_DIR) if f.endswith('_historical_1min.csv')]

    if len(csv_files) == 0:
        print(f"  [ERROR] No 1-minute market data files found in {MARKET_DATA_DIR}")
        return None, None

    for csv_file in csv_files:
        csv_path = os.path.join(MARKET_DATA_DIR, csv_file)
        try:
            df = pd.read_csv(csv_path)
            if 'datetime' in df.columns and len(df) > 0:
                df['datetime'] = pd.to_datetime(df['datetime'])
                all_datetimes.extend(df['datetime'].tolist())
        except Exception as e:
            print(f"  [WARN] Error reading {csv_file}: {e}")
            continue

    if len(all_datetimes) == 0:
        print("  [ERROR] No datetime data found in market files")
        return None, None

    start_dt = min(all_datetimes)
    end_dt = max(all_datetimes)

    print(f"  [OK] Market data range: {start_dt} to {end_dt}")
    print(f"       Duration: {(end_dt - start_dt).total_seconds() / 3600:.1f} hours")

    return start_dt, end_dt


def fetch_stock_data_1min(symbol: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """
    Fetch 1-minute stock data from EODHD API

    Args:
        symbol: Stock ticker (e.g., 'TSLA', 'SPY')
        start_dt: Start datetime
        end_dt: End datetime

    Returns:
        DataFrame with columns: timestamp, close, volume
    """
    print(f"  Fetching {symbol} 1-minute data from EODHD...")

    try:
        # EODHD uses format: TICKER.EXCHANGE (e.g., TSLA.US)
        ticker_symbol = f"{symbol}.US"

        # EODHD intraday endpoint - using 1m interval
        url = f"https://eodhd.com/api/intraday/{ticker_symbol}"
        params = {
            'api_token': EODHD_API_KEY,
            'fmt': 'json',
            'interval': '1m',
            'from': int(start_dt.timestamp()),
            'to': int(end_dt.timestamp())
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if not data:
            print(f"    [WARN] No data returned for {symbol}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # EODHD returns: timestamp, open, high, low, close, volume
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df[['timestamp', 'close', 'volume']].copy()

        print(f"    [OK] Fetched {len(df)} bars of {symbol} data")
        return df

    except Exception as e:
        print(f"    [ERROR] Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


def normalize_tesla_with_spy(tsla_df: pd.DataFrame, spy_df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Tesla prices with S&P 500

    Creates a normalized price series: TSLA_normalized = TSLA / SPY

    Args:
        tsla_df: Tesla data with 'timestamp' and 'close' columns
        spy_df: S&P 500 data with 'timestamp' and 'close' columns

    Returns:
        DataFrame with columns: timestamp, tsla_close, spy_close, normalized_close
    """
    print("  Normalizing Tesla with S&P 500...")

    # Merge on timestamp
    merged = tsla_df.merge(spy_df, on='timestamp', suffixes=('_tsla', '_spy'), how='inner')

    # Calculate normalized price
    merged['normalized_close'] = merged['close_tsla'] / merged['close_spy']

    # Rename columns for clarity
    merged = merged.rename(columns={
        'close_tsla': 'tsla_close',
        'close_spy': 'spy_close',
        'volume_tsla': 'tsla_volume',
        'volume_spy': 'spy_volume'
    })

    print(f"    [OK] Created {len(merged)} normalized data points")

    return merged[['timestamp', 'tsla_close', 'spy_close', 'normalized_close',
                   'tsla_volume', 'spy_volume']]


if __name__ == '__main__':
    print("="*70)
    print("FETCHING NORMALIZED TESLA STOCK DATA (1-MINUTE)")
    print("="*70)
    print()

    # Check if output file already exists
    if os.path.exists(OUTPUT_FILE):
        print(f"[INFO] {OUTPUT_FILE} already exists, loading from file...")
        normalized_df = pd.read_csv(OUTPUT_FILE)
        normalized_df['timestamp'] = pd.to_datetime(normalized_df['timestamp'])
        print(f"[OK] Loaded {len(normalized_df)} data points from existing file")
        print(f"     Time range: {normalized_df['timestamp'].min()} to {normalized_df['timestamp'].max()}")
    else:
        # Step 1: Determine time range from market data
        start_dt, end_dt = get_market_data_timerange()

        if start_dt is None or end_dt is None:
            print("\n[ERROR] Could not determine time range from market data")
            print("Please ensure market data files exist in the market_data/ directory")
            exit(1)

        print()

        # Step 2: Fetch Tesla and S&P 500 data
        print("Fetching stock data from EODHD...")
        tsla_df = fetch_stock_data_1min('TSLA', start_dt, end_dt)
        spy_df = fetch_stock_data_1min('SPY', start_dt, end_dt)

        if tsla_df.empty or spy_df.empty:
            print("\n[ERROR] Could not fetch stock data from EODHD")
            print("Please check your API key and network connection")
            exit(1)

        print()

        # Step 3: Normalize Tesla with S&P 500
        print("Normalizing data...")
        normalized_df = normalize_tesla_with_spy(tsla_df, spy_df)

        print()

        # Step 4: Export to CSV
        normalized_df.to_csv(OUTPUT_FILE, index=False)
        print(f"[OK] Exported normalized Tesla stock data to: {OUTPUT_FILE}")
        print(f"     {len(normalized_df)} data points")
        print(f"     Time range: {normalized_df['timestamp'].min()} to {normalized_df['timestamp'].max()}")

    print()

    # Step 5: Create visualization
    print("Creating visualization...")
    plt.figure(figsize=(16, 10))

    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12))

    # Plot 1: Tesla price
    ax1.plot(normalized_df['timestamp'], normalized_df['tsla_close'],
             linewidth=1, color='#E31937', label='TSLA Price')
    ax1.set_title('Tesla Stock Price', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # Plot 2: S&P 500 price
    ax2.plot(normalized_df['timestamp'], normalized_df['spy_close'],
             linewidth=1, color='#2E86AB', label='SPY Price')
    ax2.set_title('S&P 500 (SPY) Price', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Price (USD)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    # Plot 3: Normalized price (TSLA/SPY)
    ax3.plot(normalized_df['timestamp'], normalized_df['normalized_close'],
             linewidth=1, color='#06A77D', label='TSLA/SPY Normalized')
    ax3.set_title('Normalized Tesla Stock (TSLA/SPY Ratio)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_ylabel('Normalized Price Ratio', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(VISUALIZATION_FILE, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"[OK] Saved visualization to: {VISUALIZATION_FILE}")

    print()
    print("="*70)
    print("FETCH COMPLETE")
    print("="*70)
    print()
    print("Generated files:")
    print(f"  1. {OUTPUT_FILE} - Normalized stock data CSV")
    print(f"  2. {VISUALIZATION_FILE} - Price visualization")
    print()
    print("Columns in CSV file:")
    print("  - timestamp: Datetime of the bar")
    print("  - tsla_close: Tesla closing price")
    print("  - spy_close: S&P 500 closing price")
    print("  - normalized_close: TSLA / SPY (normalized price)")
    print("  - tsla_volume: Tesla trading volume")
    print("  - spy_volume: S&P 500 trading volume")
    print()
