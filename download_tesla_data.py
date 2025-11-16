"""
Download Tesla Market Data
Downloads 1-minute historical data for all Tesla markets
"""

import pandas as pd
import os

try:
    from kalshi_analysis import KalshiDataCollector, export_market_historical_data
except ImportError:
    print("ERROR: kalshi_analysis.py not found in current directory")
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

MARKET_DATA_DIR = "market_data"


# Step 0: Load Tesla markets from verified CSV
print("="*70)
print("LOADING TESLA MARKET DATA")
print("="*70)
print()

# Check if verified CSV exists
if not os.path.exists('all_stock_related_markets.csv'):
    print("[ERROR] all_stock_related_markets.csv not found!")
    print("Please run fetch_stock_markets.py first to generate verified market data.")
    exit(1)

# Load verified stock markets
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

# Download missing 1-minute data
print("="*70)
print("DOWNLOADING MISSING MARKET DATA")
print("="*70)
print()

# Check which markets need data downloaded
markets_needing_data = []
for _, row in tesla_markets.iterrows():
    ticker = row['ticker']
    csv_path = os.path.join(MARKET_DATA_DIR, f"{ticker}_historical_1min.csv")
    if not os.path.exists(csv_path):
        markets_needing_data.append(row)

if len(markets_needing_data) > 0:
    print(f"Need to download 1-minute data for {len(markets_needing_data)} markets...")
    try:
        user_input = input("Download now? (y/n, default=y): ").strip().lower()
    except EOFError:
        # Non-interactive mode - default to yes
        user_input = 'y'
        print("y  (auto-confirmed in non-interactive mode)")

    if user_input != 'n':
        print("Downloading market data...\n")
        collector = KalshiDataCollector()

        # Convert list of rows to DataFrame
        markets_needing_data_df = pd.DataFrame(markets_needing_data)

        # Use the export function with DataFrame
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
