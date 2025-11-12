"""
Top Markets 1-Minute Analysis
Finds the highest liquidity markets (>$2M volume) and generates 3 months of
1-minute granularity data for the top 10 markets.

This script:
1. Fetches all open markets from Kalshi API
2. Filters for markets with >$2M trading volume
3. Selects top 10 by volume
4. Exports 3 months of 1-minute candlestick data for each (using auto-chunking)
5. Creates price and volume visualizations for each market
"""

from kalshi_analysis import (
    KalshiDataCollector,
    KalshiDataProcessor,
    export_market_historical_data,
    KalshiVisualizer
)
import pandas as pd
import os

# Configuration
VOLUME_THRESHOLD_USD = 2_000_000  # $2 million
TOP_N_MARKETS = 10
MONTHS_OF_DATA = 3
OUTPUT_DIR = 'market_data'

def main():
    print("="*70)
    print("TOP MARKETS 1-MINUTE ANALYSIS")
    print("="*70)
    print(f"Configuration:")
    print(f"  - Volume threshold: >${VOLUME_THRESHOLD_USD:,} USD")
    print(f"  - Top markets to analyze: {TOP_N_MARKETS}")
    print(f"  - Data period: {MONTHS_OF_DATA} months")
    print(f"  - Granularity: 1-minute (per-minute volume)")
    print("="*70)
    print()

    # Step 1: Fetch all markets
    print("STEP 1: Fetching all open markets...")
    collector = KalshiDataCollector()
    markets = collector.fetch_markets(status="open")
    print(f"[OK] Retrieved {len(markets)} markets")
    print()

    # Step 2: Process and calculate volume
    print("STEP 2: Processing market data...")
    processor = KalshiDataProcessor()
    df = processor.markets_to_dataframe(markets)
    df = processor.calculate_trading_volume_usd(df)
    print(f"[OK] Processed {len(df)} markets")
    print()

    # Step 3: Filter for high-volume markets
    print(f"STEP 3: Filtering for markets with >${VOLUME_THRESHOLD_USD:,} volume...")
    high_volume_markets = df[df['estimated_volume_usd'] > VOLUME_THRESHOLD_USD].copy()
    print(f"[OK] Found {len(high_volume_markets)} markets above threshold")
    print()

    # Step 4: Select top N markets by volume
    print(f"STEP 4: Selecting top {TOP_N_MARKETS} markets by trading volume...")
    top_markets = high_volume_markets.nlargest(TOP_N_MARKETS, 'estimated_volume_usd')

    print(f"\nTop {TOP_N_MARKETS} Markets:")
    print("-" * 70)
    for idx, (_, market) in enumerate(top_markets.iterrows(), 1):
        ticker = market['ticker']
        title = market.get('title', 'N/A')
        volume = market['estimated_volume_usd']
        print(f"{idx:2d}. {ticker:30s} | ${volume:>12,.0f} | {str(title)[:30]}")
    print("-" * 70)
    print()

    # Step 5: Export 3 months of 1-minute data for each market
    print("="*70)
    print(f"STEP 5: Exporting {MONTHS_OF_DATA} months of 1-minute data")
    print("="*70)
    print(f"Note: This will make multiple API calls per market (3-day chunks)")
    print(f"      Estimated time: ~{TOP_N_MARKETS * MONTHS_OF_DATA * 3} seconds")
    print()

    tickers_to_export = top_markets['ticker'].tolist()

    exported_files = export_market_historical_data(
        collector,
        tickers=tickers_to_export,
        interval='1min',
        months_back=MONTHS_OF_DATA,
        output_dir=OUTPUT_DIR
    )

    # Step 6: Create visualizations
    print("="*70)
    print("STEP 6: Creating price and volume visualizations")
    print("="*70)
    print()

    visualizer = KalshiVisualizer()
    created_charts = []

    for idx, (_, market) in enumerate(top_markets.iterrows(), 1):
        ticker = market['ticker']
        title = market.get('title', 'N/A')
        csv_file = f'{OUTPUT_DIR}/{ticker}_historical_1min.csv'

        if os.path.exists(csv_file):
            print(f"[{idx}/{TOP_N_MARKETS}] Creating visualization for {ticker}...")
            try:
                visualizer.plot_price_history(csv_file, ticker, title)
                chart_name = f'kalshi_price_history_{ticker}.png'
                created_charts.append(chart_name)
                print(f"         [OK] Created: {chart_name}")
            except Exception as e:
                print(f"         [ERROR] Failed to create chart: {e}")
        else:
            print(f"[{idx}/{TOP_N_MARKETS}] [SKIP] No data file found for {ticker}")
        print()

    # Summary
    print("="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print()
    print(f"Summary:")
    print(f"  - Markets analyzed: {TOP_N_MARKETS}")
    print(f"  - CSV files exported: {len(exported_files)}")
    print(f"  - Visualizations created: {len(created_charts)}")
    print()

    if exported_files:
        print("Exported CSV files (1-minute data):")
        for file in exported_files:
            # Get file size
            try:
                size_bytes = os.path.getsize(file)
                size_mb = size_bytes / (1024 * 1024)
                print(f"  - {file} ({size_mb:.1f} MB)")
            except:
                print(f"  - {file}")

    if created_charts:
        print()
        print("Created visualizations:")
        for chart in created_charts:
            print(f"  - {chart}")

    print()
    print("="*70)
    print()
    print("Next steps:")
    print("  - Review the CSV files in the 'market_data/' directory")
    print("  - Examine the PNG visualizations for price/volume trends")
    print("  - Analyze the 1-minute data for high-frequency patterns")
    print()
    print("Data characteristics:")
    print("  - Granularity: 1-minute candles (per-minute volume)")
    print(f"  - Time span: {MONTHS_OF_DATA} months (~{MONTHS_OF_DATA * 30} days)")
    print(f"  - Expected candles per market: ~{MONTHS_OF_DATA * 30 * 1440} (if every minute had data)")
    print("  - Actual candles: Varies by market activity (~10-15% typically)")
    print("="*70)

if __name__ == "__main__":
    main()
