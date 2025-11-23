"""
Download Tesla Market Data
Downloads 1-minute historical data for all Tesla markets
"""

import pandas as pd
import os
from datetime import datetime

try:
    from kalshi_analysis import KalshiDataCollector, export_market_historical_data
except ImportError:
    print("ERROR: kalshi_analysis.py not found in current directory")
    exit(1)


MARKET_DATA_DIR = "market_data"

print("="*70)
print("LOADING TESLA MARKET DATA")
print("="*70)
print()

if not os.path.exists('all_stock_related_markets.csv'):
    print("[ERROR] all_stock_related_markets.csv not found!")
    print("Please run fetch_stock_markets.py first to generate verified market data.")
    exit(1)

print("Loading verified stock markets from fetch_stock_markets.py output...")
stock_df = pd.read_csv('all_stock_related_markets.csv')
tesla_markets = stock_df[stock_df['company'] == 'Tesla'].copy()

if len(tesla_markets) == 0:
    print("[ERROR] No Tesla markets found in CSV!")
    print("Please run fetch_stock_markets.py to fetch Tesla market data.")
    exit(1)

print(f"[OK] Found {len(tesla_markets)} verified Tesla markets")
print(f"Total Tesla volume: ${tesla_markets['volume_usd'].sum():,.2f}")
print()

print("="*70)
print("DOWNLOADING MISSING MARKET DATA")
print("="*70)
print()

markets_needing_data = []
for _, row in tesla_markets.iterrows():
    ticker = row['ticker']
    csv_path = os.path.join(MARKET_DATA_DIR, f"{ticker}_historical_1min.csv")
    if not os.path.exists(csv_path):
        markets_needing_data.append(row)

markets_needing_data_df = pd.DataFrame(markets_needing_data)


def parse_iso(ts):
    if isinstance(ts, str) and ts:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except:
            return None
    return None

if not markets_needing_data_df.empty:
    markets_needing_data_df["open_time"] = markets_needing_data_df["open_time"].apply(parse_iso)
    markets_needing_data_df["close_time"] = markets_needing_data_df["close_time"].apply(parse_iso)


if len(markets_needing_data) > 0:
    print(f"Need to download 1-minute data for {len(markets_needing_data)} markets...")
    try:
        user_input = input("Download now? (y/n, default=y): ").strip().lower()
    except EOFError:
        user_input = 'y'
        print("y  (auto-confirmed in non-interactive mode)")

    if user_input != 'n':
        print("Downloading market data...\n")
        collector = KalshiDataCollector()

        export_market_historical_data(
            collector=collector,
            df=markets_needing_data_df,
            output_dir=MARKET_DATA_DIR,
            months_back=3,
            interval='1min'
        )

        print(f"\n[OK] Downloaded data for {len(markets_needing_data)} markets")
    else:
        print("Skipping download. Will only use markets with existing data.")

else:
    print("[OK] All Tesla markets already have 1-minute data")

print()
print("="*70)
print("DOWNLOAD COMPLETE")
print("="*70)
print()
print("="*70)
print("EXPORTING COMBINED TESLA 1-MINUTE DATA")
print("="*70)

combined_rows = []

for _, row in tesla_markets.iterrows():
    ticker = row["ticker"]
    csv_path = os.path.join(MARKET_DATA_DIR, f"{ticker}_historical_1min.csv")

    if not os.path.exists(csv_path):
        print(f"[WARN] Missing historical data for {ticker}, skipping.")
        continue

    df = pd.read_csv(csv_path)

    df["market_ticker"] = ticker
    df["event_ticker"] = row.get("event_ticker", None)
    df["title"] = row.get("title", None)

    combined_rows.append(df)

if len(combined_rows) == 0:
    print("[ERROR] No market data found. Nothing to export.")
    exit(1)

final_df = pd.concat(combined_rows, ignore_index=True)

if "timestamp" in final_df.columns:
    final_df["timestamp"] = pd.to_datetime(final_df["timestamp"], errors="coerce")
    final_df = final_df.sort_values("timestamp")

output_path = "tesla_markets_1min_combined.csv"
final_df.to_csv(output_path, index=False)

print(f"[OK] Exported combined Tesla 1-minute dataset → {output_path}")
print(f"Rows: {len(final_df):,}")
print("="*70)
