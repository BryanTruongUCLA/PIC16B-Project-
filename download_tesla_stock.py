"""
Download 6 Months of Tesla Stock Data (1-minute candles)
Downloads TSLA and SPY historical data from EODHD
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# Configuration
EODHD_API_KEY = "6918f440b08f73.14158490"
OUTPUT_FILE = "normalized_tesla_stock_1min.csv"

# Calculate date range 6 months back from time of running
end_date = datetime.now()
start_date = end_date - timedelta(days=180)

print(f"Date range: {start_date.date()} to {end_date.date()}")
print()

def fetch_stock_data_1min(symbol, start_dt, end_dt):
    """Fetch 1-minute stock data from EODHD API"""
    print(f"Fetching {symbol} data...")

    all_data = []
    current_date = start_dt.date()
    end_date_only = end_dt.date()

    while current_date <= end_date_only:
        ticker_symbol = f"{symbol}.US"

        # Convert to Unix timestamps
        from_timestamp = int(datetime.combine(current_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(current_date, datetime.max.time()).timestamp())

        url = f"https://eodhd.com/api/intraday/{ticker_symbol}"
        params = {
            'api_token': EODHD_API_KEY,
            'interval': '1m',
            'from': from_timestamp,
            'to': to_timestamp,
            'fmt': 'json'
        }

        date_str = current_date.strftime('%Y-%m-%d')

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data and isinstance(data, list):
                all_data.extend(data)
            else:
                print(f"No data")

        except Exception as e:
            print(f"Error - {e}")

        current_date += timedelta(days=1)

    if all_data:
        df = pd.DataFrame(all_data)
        df['timestamp'] = pd.to_datetime(df['datetime'])
        df = df.rename(columns={
            'close': f'{symbol.lower()}_close',
            'volume': f'{symbol.lower()}_volume',
            'open': f'{symbol.lower()}_open',
            'high': f'{symbol.lower()}_high',
            'low': f'{symbol.lower()}_low'
        })
        return df[['timestamp'] + [col for col in df.columns if symbol.lower() in col]]

    return pd.DataFrame()

# Download TSLA and SPY data
tsla_data = fetch_stock_data_1min('TSLA', start_date, end_date)
spy_data = fetch_stock_data_1min('SPY', start_date, end_date)

if tsla_data.empty or spy_data.empty:
    print("Failed to download stock data")
    exit(1)

# Merge TSLA and SPY data
merged_stock_data = pd.merge(tsla_data, spy_data, on='timestamp', how='outer')
merged_stock_data = merged_stock_data.sort_values('timestamp').reset_index(drop=True)

# Forward fill missing values
merged_stock_data = merged_stock_data.ffill()


# Save stock data
merged_stock_data['datetime'] = merged_stock_data['timestamp']
merged_stock_data.to_csv(OUTPUT_FILE, index=False)
print(f"Saved stock data to {OUTPUT_FILE}")
