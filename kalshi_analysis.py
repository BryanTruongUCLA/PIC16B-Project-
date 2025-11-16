"""
Kalshi Events Analysis
Fetches and analyzes open Kalshi markets, aggregated by event
Filters for events with >$2M total trading volume
Exports granular 1-minute price history for top 5 most liquid events
Creates visualizations of trading volume and market activity
"""

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
from typing import List, Dict, Optional

# API Configuration
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Volume threshold: $2 million
VOLUME_THRESHOLD_USD = 2_000_000

# Top N events to export detailed historical data
TOP_N_DETAILED_EXPORT = 5


class KalshiDataCollector:
    """Handles data collection from Kalshi API"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'KalshiAnalysis/1.0'
        })

    def fetch_markets(self, status: str = "open", limit: int = 1000) -> List[Dict]:
        """
        Fetch all markets matching the criteria with pagination

        Args:
            status: Market status filter (default: "open" for active/open markets)
            limit: Number of results per page

        Returns:
            List of market dictionaries
        """
        all_markets = []
        cursor = None
        page = 0

        print(f"Fetching {status} markets...")

        while True:
            page += 1
            url = f"{BASE_URL}/markets"
            params = {
                'status': status,
                'limit': limit
            }

            if cursor:
                params['cursor'] = cursor

            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()

                markets = data.get('markets', [])
                all_markets.extend(markets)

                print(f"  Page {page}: Fetched {len(markets)} markets (Total: {len(all_markets)})")

                cursor = data.get('cursor')
                if not cursor:
                    break

                # Be respectful with rate limiting
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                print(f"Error fetching markets: {e}")
                break

        print(f"[OK] Total markets fetched: {len(all_markets)}")
        return all_markets

    def fetch_market_candlesticks(self, ticker: str, event_ticker: str = None, period_interval: int = 1440,
                                   start_ts: Optional[int] = None, end_ts: Optional[int] = None) -> Optional[List[Dict]]:
        """
        Fetch candlestick data for a specific market using the series endpoint

        Args:
            ticker: Market ticker
            event_ticker: Event ticker (required for API call)
            period_interval: 1 (1min), 60 (1hr), 1440 (1day)
            start_ts: Start timestamp (REQUIRED by API)
            end_ts: End timestamp (REQUIRED by API)

        Returns:
            List of candlestick dictionaries or None
        """
        if not event_ticker:
            print(f"  Warning: event_ticker required for candlesticks endpoint")
            return None

        if not start_ts or not end_ts:
            print(f"  Warning: start_ts and end_ts are required by Kalshi API")
            return None

        url = f"{BASE_URL}/series/{event_ticker}/markets/{ticker}/candlesticks"
        params = {
            'period_interval': period_interval,
            'start_ts': start_ts,
            'end_ts': end_ts
        }

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get('candlesticks', [])
        except requests.exceptions.RequestException as e:
            print(f"  Warning: Could not fetch candlesticks for {ticker} (event: {event_ticker}): {e}")
            return None


class KalshiDataProcessor:
    """Processes and filters Kalshi market data"""

    @staticmethod
    def markets_to_dataframe(markets: List[Dict]) -> pd.DataFrame:
        """Convert markets list to pandas DataFrame"""
        if not markets:
            return pd.DataFrame()

        df = pd.DataFrame(markets)

        # Convert timestamps to datetime (use ISO8601 format for flexibility)
        time_columns = ['open_time', 'close_time', 'latest_expiration_time']
        for col in time_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='ISO8601', errors='coerce')

        # Extract year from close_time
        if 'close_time' in df.columns:
            df['year'] = df['close_time'].dt.year

        # Convert prices from cents to dollars
        price_columns = ['yes_bid', 'yes_ask', 'no_bid', 'no_ask', 'last_price',
                        'previous_yes_bid', 'previous_yes_ask', 'previous_price']
        for col in price_columns:
            if col in df.columns:
                df[f'{col}_usd'] = df[col] / 100

        return df

    @staticmethod
    def calculate_trading_volume_usd(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading volume in USD

        Each Kalshi contract is worth $1 (paid out as either $1 for Yes or $1 for No).
        Therefore: Trading Volume (USD) = Number of Contracts Traded × $1
        """
        if 'volume' in df.columns:
            df['estimated_volume_usd'] = df['volume'] * 1.0
        else:
            df['estimated_volume_usd'] = 0

        return df

    @staticmethod
    def filter_high_volume(df: pd.DataFrame, threshold_usd: float = VOLUME_THRESHOLD_USD) -> pd.DataFrame:
        """Filter markets with volume > threshold"""
        if 'estimated_volume_usd' not in df.columns:
            return df

        filtered = df[df['estimated_volume_usd'] > threshold_usd].copy()
        print(f"\n[OK] Filtered to {len(filtered)} events with >${threshold_usd:,.0f} USD volume")
        print(f"  (from {len(df)} total events)")

        return filtered


class KalshiVisualizer:
    """Creates visualizations for Kalshi market data"""

    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
        self.fig_count = 0

    def plot_volume_distribution(self, df: pd.DataFrame, top_n: int = 30):
        """Bar chart of trading volume for top events"""
        if df.empty or 'estimated_volume_usd' not in df.columns:
            print("No data to plot for volume distribution")
            return

        self.fig_count += 1

        # Sort by volume and take top N
        top_events = df.nlargest(top_n, 'estimated_volume_usd')[['ticker', 'estimated_volume_usd', 'title', 'market_count']].copy()

        plt.figure(figsize=(16, 10))
        bars = plt.barh(range(len(top_events)), top_events['estimated_volume_usd'] / 1_000_000)

        # Color gradient
        colors = plt.cm.viridis(range(len(top_events)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        # Create labels with market count
        labels = []
        for _, row in top_events.iterrows():
            market_count = int(row['market_count']) if 'market_count' in row and pd.notna(row['market_count']) else '?'
            labels.append(f"{row['ticker']} ({market_count}m)")

        plt.yticks(range(len(top_events)), labels, fontsize=9)
        plt.xlabel('Total Trading Volume (Millions USD)', fontsize=12, fontweight='bold')
        plt.ylabel('Event (Market Count)', fontsize=12, fontweight='bold')
        plt.title(f'Top {top_n} Kalshi Events by Total Trading Volume',
                 fontsize=14, fontweight='bold', pad=20)
        plt.grid(axis='x', alpha=0.3)

        # Add value labels
        for i, (idx, row) in enumerate(top_events.iterrows()):
            plt.text(row['estimated_volume_usd'] / 1_000_000, i,
                    f" ${row['estimated_volume_usd']/1_000_000:.1f}M",
                    va='center', fontsize=8)

        plt.tight_layout()
        filename = 'kalshi_events_volume_distribution.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n[OK] Saved: {filename}")
        plt.close()

    def plot_volume_by_threshold(self, df: pd.DataFrame):
        """Bar chart showing number of events at different volume thresholds"""
        if df.empty or 'estimated_volume_usd' not in df.columns:
            print("No data to plot for volume thresholds")
            return

        self.fig_count += 1

        thresholds = [100_000, 500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000]
        counts = [len(df[df['estimated_volume_usd'] > t]) for t in thresholds]
        labels = [f'>${t/1_000_000:.1f}M' if t >= 1_000_000 else f'>${t/1_000:.0f}K' for t in thresholds]

        plt.figure(figsize=(12, 7))
        bars = plt.bar(range(len(thresholds)), counts, color='#2E86AB', alpha=0.8, edgecolor='black')

        plt.xticks(range(len(thresholds)), labels, fontsize=11)
        plt.xlabel('Volume Threshold', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Events', fontsize=12, fontweight='bold')
        plt.title('Kalshi Events by Volume Threshold', fontsize=14, fontweight='bold', pad=20)
        plt.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, counts)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        filename = 'kalshi_events_by_threshold.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {filename}")
        plt.close()

    def plot_price_history(self, csv_file: str, ticker: str, market_title: str = None):
        """Plot price history from exported CSV file"""
        try:
            df = pd.read_csv(csv_file)

            if df.empty:
                print(f"No data in {csv_file}")
                return

            # Convert datetime if needed
            if 'datetime' in df.columns:
                df['datetime'] = pd.to_datetime(df['datetime'])

            # Detect interval from filename or data spacing
            interval_label = "Volume"
            if '_1min.csv' in csv_file:
                interval_label = "Per-Minute Volume"
            elif '_hourly.csv' in csv_file:
                interval_label = "Hourly Volume"
            elif '_daily.csv' in csv_file:
                interval_label = "Daily Volume"
            elif len(df) > 1 and 'datetime' in df.columns:
                # Auto-detect from data spacing
                time_diff = (df['datetime'].iloc[1] - df['datetime'].iloc[0]).total_seconds()
                if time_diff < 120:  # < 2 minutes
                    interval_label = "Per-Minute Volume"
                elif time_diff < 7200:  # < 2 hours
                    interval_label = "Hourly Volume"
                else:
                    interval_label = "Daily Volume"

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

            # Price plot
            if 'close_usd' in df.columns:
                ax1.plot(df['datetime'], df['close_usd'], linewidth=2, color='#2E86AB', label='YES Price (Close)')

                # Add open/high/low if available
                if 'high_usd' in df.columns and 'low_usd' in df.columns:
                    ax1.fill_between(df['datetime'], df['low_usd'], df['high_usd'],
                                    alpha=0.2, color='#2E86AB', label='YES Price Range (High-Low)')

                ax1.set_ylabel('YES Price (USD)', fontsize=12, fontweight='bold')

                # Create title with market name if available
                if market_title:
                    title = f'{ticker}\n{market_title[:80]}'
                else:
                    title = f'{ticker} - Price History'
                ax1.set_title(title, fontsize=14, fontweight='bold', pad=20)
                ax1.grid(alpha=0.3)
                ax1.legend(loc='upper left')
                ax1.set_ylim(0, 1)  # Kalshi prices are 0-1

            # Volume plot
            if 'volume' in df.columns:
                ax2.bar(df['datetime'], df['volume'], alpha=0.6, color='#A23B72', label=interval_label)
                ax2.set_ylabel(f'{interval_label} (Contracts)', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
                ax2.grid(alpha=0.3)
                ax2.legend(loc='upper left')

            plt.tight_layout()
            filename = f'kalshi_price_history_{ticker.replace("/", "_")}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"[OK] Saved: {filename}")
            plt.close()

        except Exception as e:
            print(f"[ERROR] Could not plot {csv_file}: {e}")


def export_market_historical_data(collector: KalshiDataCollector,
                                   tickers: List[str] = None,
                                   df: pd.DataFrame = None,
                                   top_n: int = None,
                                   output_dir: str = 'market_data',
                                   months_back: int = 3,
                                   use_1min: bool = False,
                                   interval: str = 'auto'):
    """
    Export historical candlestick data for specified markets to separate CSV files

    Can be called in three ways:
    1. With specific tickers: export_market_historical_data(collector, tickers=['TICKER1', 'TICKER2'])
    2. With DataFrame and top_n: export_market_historical_data(collector, df=df, top_n=5)
    3. With DataFrame alone: export_market_historical_data(collector, df=df) - exports all markets in df

    Args:
        collector: KalshiDataCollector instance
        tickers: Optional list of specific ticker symbols to export
        df: Optional DataFrame with market data (used if tickers not provided)
        top_n: Optional number of top markets by volume to export (used with df)
        output_dir: Directory to save CSV files
        months_back: Number of months of history to fetch (default: 3)
        use_1min: Use 1-minute candles. If False, uses hourly (default: False)
        interval: Candle interval - 'auto', '1min', 'hourly', or 'daily' (default: 'auto')
                 'auto': Selects based on months_back (1min for <=3 days, hourly for <=3 months, daily otherwise)
                 '1min': Per-minute data - NOW SUPPORTS EXTENDED PERIODS via automatic chunking!
                         API limit is 3 days per request, but this function automatically splits
                         longer periods into 3-day chunks and combines results into one CSV.
                         Example: months_back=1 (30 days) → 10 API calls → 1 combined CSV
                 'hourly': Hourly data (recommended for up to 3 months)
                 'daily': Daily data (can fetch many months/years of history)
    """
    import os

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n[OK] Created directory: {output_dir}")

    # Determine which markets to export
    markets_to_export = []

    if tickers:
        # Export specific tickers
        print(f"\n{'='*70}")
        print(f"EXPORTING HISTORICAL DATA FOR {len(tickers)} SPECIFIED MARKETS")
        print(f"{'='*70}")
        for ticker in tickers:
            # Fetch market details to get event_ticker
            try:
                response = collector.session.get(f"{BASE_URL}/markets/{ticker}")
                if response.ok:
                    market_data = response.json()['market']
                    # Parse open_time and close_time from API
                    open_time = None
                    close_time = None
                    if market_data.get('open_time'):
                        open_time = pd.to_datetime(market_data['open_time'])
                    if market_data.get('close_time'):
                        close_time = pd.to_datetime(market_data['close_time'])

                    markets_to_export.append({
                        'ticker': ticker,
                        'title': market_data.get('title', 'N/A'),
                        'volume_usd': None,
                        'open_time': open_time,
                        'close_time': close_time,
                        'event_ticker': market_data.get('event_ticker')
                    })
                else:
                    print(f"Warning: Could not fetch details for {ticker}")
                    markets_to_export.append({
                        'ticker': ticker,
                        'title': 'N/A',
                        'volume_usd': None,
                        'open_time': None,
                        'close_time': None,
                        'event_ticker': None
                    })
            except Exception as e:
                print(f"Error fetching market {ticker}: {e}")
                continue
    elif df is not None:
        if df.empty:
            print("No markets to export")
            return []

        # Export from DataFrame
        if top_n:
            selected = df.nlargest(top_n, 'estimated_volume_usd')
            print(f"\n{'='*70}")
            print(f"EXPORTING HISTORICAL DATA FOR TOP {top_n} EVENTS")
            print(f"{'='*70}")
        else:
            selected = df
            print(f"\n{'='*70}")
            print(f"EXPORTING HISTORICAL DATA FOR {len(df)} EVENTS")
            print(f"{'='*70}")

        for _, market in selected.iterrows():
            markets_to_export.append({
                'ticker': market['ticker'],
                'title': market.get('title', 'N/A'),
                'volume_usd': market.get('estimated_volume_usd'),
                'open_time': market.get('open_time'),
                'close_time': market.get('close_time'),
                'event_ticker': market.get('event_ticker', None)
            })
    else:
        print("[ERROR] Must provide either tickers or df parameter")
        return []

    # Determine interval settings
    if use_1min:
        # Legacy parameter support
        interval = '1min'

    # Auto-select interval based on months_back
    if interval == 'auto':
        days_back = months_back * 30
        if days_back <= 3:
            interval = '1min'
        elif days_back <= 90:  # ~3 months
            interval = 'hourly'
        else:
            interval = 'daily'

    # Set interval parameters
    interval_map = {
        '1min': {'period': 1, 'name': '1-minute', 'suffix': '1min', 'max_days': 3},
        'hourly': {'period': 60, 'name': 'hourly', 'suffix': 'hourly', 'max_days': None},
        'daily': {'period': 1440, 'name': 'daily', 'suffix': 'daily', 'max_days': None}
    }

    if interval not in interval_map:
        print(f"[ERROR] Invalid interval '{interval}'. Use 'auto', '1min', 'hourly', or 'daily'")
        return []

    interval_config = interval_map[interval]
    period_interval = interval_config['period']
    interval_name = interval_config['name']
    interval_suffix = interval_config['suffix']
    max_days = interval_config['max_days']

    print(f"Fetching {interval_name} candlesticks")
    if max_days and interval != '1min':
        print(f"Note: {interval_name} data limited to ~{max_days} days of history")
    print(f"Time range: From market open or last {months_back} months, whichever is shorter")

    # For 1-minute data, explain chunking strategy
    if interval == '1min' and months_back * 30 > 3:
        days_requested = months_back * 30
        num_chunks = int((days_requested + 2) / 3)  # Round up, 3 days per chunk
        print(f"Note: Requested {days_requested:.0f} days of 1-minute data")
        print(f"      Will make {num_chunks} API calls (3-day chunks) and combine results")

    print(f"{'='*70}\n")

    # Calculate months_back timestamp
    months_back_ts = int((datetime.now().timestamp() - (months_back * 30 * 24 * 60 * 60)))

    exported_files = []

    for idx, market in enumerate(markets_to_export, 1):
        ticker = market['ticker']
        title = market['title']
        volume_usd = market['volume_usd']
        open_time = market['open_time']
        close_time = market['close_time']
        event_ticker = market.get('event_ticker')

        print(f"[{idx}/{len(markets_to_export)}] Processing: {ticker}")
        if title and title != 'N/A':
            print(f"      Title: {title[:60]}...")
        if volume_usd:
            print(f"      Volume: ${volume_usd:,.2f}")

        # Determine start time: market open or months_back ago, whichever is LATER (more recent)
        # This prevents fetching data before the market existed
        if open_time and pd.notna(open_time):
            # Parse string to datetime if needed
            if isinstance(open_time, str):
                open_time = datetime.fromisoformat(open_time.replace('Z', '+00:00'))
            market_open_ts = int(open_time.timestamp())
            # Always start from market open if it's more recent than months_back
            start_ts = max(market_open_ts, months_back_ts)
            start_source = "market open" if start_ts == market_open_ts else f"{months_back} months ago"
        else:
            # If no open_time available, use months_back but warn user
            start_ts = months_back_ts
            start_source = f"{months_back} months ago (no market open time available)"

        # End time: market close or now, whichever is earlier (can't get future data)
        now_ts = int(datetime.now().timestamp())
        if close_time and pd.notna(close_time):
            # Parse string to datetime if needed
            if isinstance(close_time, str):
                close_time = datetime.fromisoformat(close_time.replace('Z', '+00:00'))
            end_ts = min(int(close_time.timestamp()), now_ts)
        else:
            end_ts = now_ts

        print(f"      Time range: {datetime.fromtimestamp(start_ts)} ({start_source}) to {datetime.fromtimestamp(end_ts)}")

        # Fetch candlestick data
        # Use the correct endpoint: /series/{series_ticker}/markets/{ticker}/candlesticks
        if not event_ticker:
            print(f"      [WARN] No event_ticker available, skipping candlestick fetch")
            print()
            continue

        # Determine if we need chunking for 1-minute data
        total_duration = end_ts - start_ts
        use_chunking = (interval == '1min' and total_duration > (3 * 24 * 60 * 60))

        if use_chunking:
            # Multi-chunk approach for 1-minute data beyond 3 days
            chunk_size = 3 * 24 * 60 * 60  # 3 days in seconds
            num_chunks = int((total_duration + chunk_size - 1) / chunk_size)  # Round up

            print(f"      [INFO] Splitting into {num_chunks} chunks (3 days each)")

            all_candlesticks = []
            chunk_start = start_ts

            for chunk_num in range(num_chunks):
                chunk_end = min(chunk_start + chunk_size, end_ts)

                # Display progress
                chunk_start_date = datetime.fromtimestamp(chunk_start).strftime('%Y-%m-%d')
                chunk_end_date = datetime.fromtimestamp(chunk_end).strftime('%Y-%m-%d')
                print(f"      Chunk {chunk_num + 1}/{num_chunks}: {chunk_start_date} to {chunk_end_date}...", end='')

                url = f"{BASE_URL}/series/{event_ticker}/markets/{ticker}/candlesticks"
                params = {
                    'period_interval': period_interval,
                    'start_ts': chunk_start,
                    'end_ts': chunk_end
                }

                try:
                    response = collector.session.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                    chunk_candlesticks = data.get('candlesticks', [])

                    if chunk_candlesticks:
                        all_candlesticks.extend(chunk_candlesticks)
                        print(f" {len(chunk_candlesticks)} candles")
                    else:
                        print(" no data")

                    # Rate limiting between chunks
                    if chunk_num < num_chunks - 1:
                        time.sleep(0.2)

                except requests.exceptions.RequestException as e:
                    print(f" ERROR: {e}")
                    # Continue with next chunk even if one fails

                chunk_start = chunk_end

            candlesticks = all_candlesticks
            print(f"      [OK] Combined {len(candlesticks)} total candles from {num_chunks} chunks")

        else:
            # Single API call (original behavior for non-chunked requests)
            # Apply max_days limit if specified and not using chunking
            if max_days and not use_chunking:
                max_duration = max_days * 24 * 60 * 60
                if end_ts - start_ts > max_duration:
                    start_ts = end_ts - max_duration
                    print(f"      [INFO] Limited to last {max_days} days for {interval_name} candles")

            url = f"{BASE_URL}/series/{event_ticker}/markets/{ticker}/candlesticks"
            params = {
                'period_interval': period_interval,
                'start_ts': start_ts,
                'end_ts': end_ts
            }

            try:
                response = collector.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                candlesticks = data.get('candlesticks', [])
            except requests.exceptions.RequestException as e:
                print(f"      [ERROR] Error fetching data for {ticker}: {e}")
                candlesticks = []

        # Process candlesticks (same for both chunked and non-chunked)
        try:
            if candlesticks and len(candlesticks) > 0:
                # Parse the nested candlestick structure
                parsed_data = []
                for cs in candlesticks:
                    price = cs.get('price', {})
                    yes_ask = cs.get('yes_ask', {})
                    yes_bid = cs.get('yes_bid', {})

                    parsed_data.append({
                        'timestamp': cs.get('end_period_ts'),
                        'open': price.get('open'),
                        'high': price.get('high'),
                        'low': price.get('low'),
                        'close': price.get('close'),
                        'volume': cs.get('volume'),
                        'open_interest': cs.get('open_interest'),
                        'yes_ask_open': yes_ask.get('open'),
                        'yes_ask_high': yes_ask.get('high'),
                        'yes_ask_low': yes_ask.get('low'),
                        'yes_ask_close': yes_ask.get('close'),
                        'yes_bid_open': yes_bid.get('open'),
                        'yes_bid_high': yes_bid.get('high'),
                        'yes_bid_low': yes_bid.get('low'),
                        'yes_bid_close': yes_bid.get('close'),
                    })

                # Convert to DataFrame
                cs_df = pd.DataFrame(parsed_data)

                # Sort by timestamp to ensure chronological order (important for chunked data)
                cs_df = cs_df.sort_values('timestamp').reset_index(drop=True)

                # Add human-readable timestamp
                cs_df['datetime'] = pd.to_datetime(cs_df['timestamp'], unit='s')

                # Gap-fill: Create continuous time series
                # Build complete index of expected timestamps
                if len(cs_df) > 0:
                    # Determine interval in seconds
                    interval_seconds = period_interval * 60  # period_interval is in minutes

                    # Create complete datetime range
                    start_dt = cs_df['datetime'].min()
                    end_dt = cs_df['datetime'].max()
                    complete_range = pd.date_range(start=start_dt, end=end_dt, freq=f'{period_interval}min')

                    # Mark original data as from API
                    cs_df['source'] = 'api'

                    # Set datetime as index for reindexing
                    cs_df = cs_df.set_index('datetime')

                    # Reindex to complete range
                    cs_df_filled = cs_df.reindex(complete_range)

                    # Mark imputed rows
                    cs_df_filled['source'] = cs_df_filled['source'].fillna('imputed')

                    # Forward-fill price fields (using last known price)
                    price_fields = ['open', 'high', 'low', 'close',
                                   'yes_ask_open', 'yes_ask_high', 'yes_ask_low', 'yes_ask_close',
                                   'yes_bid_open', 'yes_bid_high', 'yes_bid_low', 'yes_bid_close']
                    for field in price_fields:
                        if field in cs_df_filled.columns:
                            cs_df_filled[field] = cs_df_filled[field].ffill()

                    # Fill volume with 0 for imputed rows
                    if 'volume' in cs_df_filled.columns:
                        cs_df_filled['volume'] = cs_df_filled['volume'].fillna(0)

                    # Forward-fill open_interest
                    if 'open_interest' in cs_df_filled.columns:
                        cs_df_filled['open_interest'] = cs_df_filled['open_interest'].ffill()

                    # Reconstruct timestamp from index
                    cs_df_filled['datetime'] = cs_df_filled.index
                    cs_df_filled['timestamp'] = (cs_df_filled['datetime'].astype(int) // 10**9).astype(int)
                    cs_df_filled = cs_df_filled.reset_index(drop=True)

                    # Use filled dataframe
                    cs_df = cs_df_filled

                    print(f"      [INFO] Gap-filled: {(cs_df['source'] == 'imputed').sum()} imputed candles, {(cs_df['source'] == 'api').sum()} from API")

                # Convert prices from cents to USD
                price_cols = ['open', 'high', 'low', 'close',
                             'yes_ask_open', 'yes_ask_high', 'yes_ask_low', 'yes_ask_close',
                             'yes_bid_open', 'yes_bid_high', 'yes_bid_low', 'yes_bid_close']
                for col in price_cols:
                    if col in cs_df.columns:
                        cs_df[f'{col}_usd'] = cs_df[col] / 100

                # Reorder columns for better readability
                column_order = ['datetime', 'timestamp', 'open', 'high', 'low', 'close',
                               'open_usd', 'high_usd', 'low_usd', 'close_usd', 'volume', 'open_interest']
                existing_cols = [col for col in column_order if col in cs_df.columns]
                other_cols = [col for col in cs_df.columns if col not in existing_cols]
                cs_df = cs_df[existing_cols + other_cols]

                # Create safe filename
                safe_ticker = ticker.replace('/', '_').replace('\\', '_')
                filename = f"{output_dir}/{safe_ticker}_historical_{interval_suffix}.csv"

                # Export to CSV
                cs_df.to_csv(filename, index=False)

                print(f"      [OK] Exported {len(cs_df):,} data points to: {filename}")
                exported_files.append(filename)

            else:
                print(f"      [WARN] No candlestick data available for {ticker}")

        except Exception as e:
            print(f"      [ERROR] Error processing data for {ticker}: {e}")

        # Rate limiting
        time.sleep(0.2)
        print()

    print(f"{'='*70}")
    print(f"[OK] EXPORT COMPLETE!")
    print(f"{'='*70}")
    print(f"Successfully exported {len(exported_files)} market data files")
    print(f"Location: {output_dir}/")
    print(f"{'='*70}\n")

    return exported_files


def visualize_markets_from_csv(collector: KalshiDataCollector,
                                tickers: List[str] = None,
                                csv_file: str = None,
                                all_markets: bool = False,
                                use_1min: bool = False,
                                interval: str = 'auto',
                                months_back: int = 3):
    """
    Create price history visualizations for specific markets or all markets from a CSV

    Usage examples:
        # Visualize specific markets with hourly data
        visualize_markets_from_csv(collector, tickers=['KXSB-26-PHI', 'KXTRUMPOUT-26-TRUMP'])

        # Visualize all markets from the CSV
        visualize_markets_from_csv(collector, csv_file='kalshi_events_over_2000k.csv', all_markets=True)

        # Visualize with 1-minute data
        visualize_markets_from_csv(collector, tickers=['KXSB-26-PHI'], use_1min=True)

        # Visualize with daily data (less granular, more history)
        visualize_markets_from_csv(collector, tickers=['KXSB-26-PHI'], interval='daily', months_back=6)

    Args:
        collector: KalshiDataCollector instance
        tickers: List of specific market tickers to visualize
        csv_file: Path to CSV file containing markets (required if all_markets=True)
        all_markets: If True, visualize all markets from csv_file
        use_1min: Use 1-minute candles instead of hourly (limited to 3 days) - legacy parameter
        interval: Candle interval - 'auto', '1min', 'hourly', or 'daily' (default: 'auto')
        months_back: Number of months of history to fetch (default: 3)
    """
    import os

    visualizer = KalshiVisualizer()
    markets_to_viz = []

    if all_markets and csv_file:
        # Load all markets from CSV
        df = pd.read_csv(csv_file)
        if 'ticker' not in df.columns:
            print("[ERROR] CSV file must have a 'ticker' column")
            return

        print(f"\n{'='*70}")
        print(f"VISUALIZING ALL {len(df)} MARKETS FROM {csv_file}")
        print(f"{'='*70}")

        # For each event ticker, we need to find individual market tickers
        # Load the markets to get individual tickers for each event
        markets = collector.fetch_markets(status="open")
        processor = KalshiDataProcessor()
        markets_df = processor.markets_to_dataframe(markets)
        markets_df = processor.calculate_trading_volume_usd(markets_df)

        for _, row in df.iterrows():
            event_ticker = row['ticker']
            # Find highest volume market for this event
            event_markets = markets_df[markets_df['event_ticker'] == event_ticker]
            if not event_markets.empty:
                highest = event_markets.nlargest(1, 'estimated_volume_usd').iloc[0]
                markets_to_viz.append({
                    'ticker': highest['ticker'],
                    'title': highest.get('title', 'N/A')
                })

    elif tickers:
        # Specific tickers provided
        print(f"\n{'='*70}")
        print(f"VISUALIZING {len(tickers)} SPECIFIED MARKETS")
        print(f"{'='*70}")

        for ticker in tickers:
            # Fetch market details
            try:
                response = collector.session.get(f"{BASE_URL}/markets/{ticker}")
                if response.ok:
                    market_data = response.json()['market']
                    markets_to_viz.append({
                        'ticker': ticker,
                        'title': market_data.get('title', 'N/A')
                    })
                else:
                    markets_to_viz.append({'ticker': ticker, 'title': 'N/A'})
            except:
                markets_to_viz.append({'ticker': ticker, 'title': 'N/A'})
    else:
        print("[ERROR] Must provide either tickers or csv_file with all_markets=True")
        return

    # Export historical data for all markets
    ticker_list = [m['ticker'] for m in markets_to_viz]
    if ticker_list:
        export_market_historical_data(collector, tickers=ticker_list, use_1min=use_1min,
                                      interval=interval, months_back=months_back)

        # Create visualizations
        print(f"\n{'='*70}")
        print("CREATING PRICE HISTORY VISUALIZATIONS")
        print(f"{'='*70}")

        # Determine actual interval suffix used
        interval_map = {
            '1min': '1min',
            'hourly': 'hourly',
            'daily': 'daily',
            'auto': 'hourly'  # Default for auto
        }
        if use_1min:
            interval_suffix = '1min'
        else:
            interval_suffix = interval_map.get(interval, 'hourly')

        created_charts = []

        for market in markets_to_viz:
            ticker = market['ticker']
            title = market['title']
            csv_path = f'market_data/{ticker}_historical_{interval_suffix}.csv'

            if os.path.exists(csv_path):
                visualizer.plot_price_history(csv_path, ticker, title)
                created_charts.append(f'kalshi_price_history_{ticker}.png')
                print(f"  [OK] Created: kalshi_price_history_{ticker}.png")
            else:
                print(f"  [SKIP] No data file found: {csv_path}")

        print(f"\n{'='*70}")
        print(f"[OK] Created {len(created_charts)} visualizations")
        print(f"{'='*70}")

        return created_charts


def main():
    """Main execution function"""
    print("="*70)
    print("KALSHI EVENTS ANALYSIS (AGGREGATED BY EVENT)")
    print("="*70)
    print(f"Filter: Events with >${VOLUME_THRESHOLD_USD:,} USD total volume")
    print(f"Creates visualizations of volume distribution and market activity")
    print("="*70)

    # Step 1: Collect data
    collector = KalshiDataCollector()

    # Fetch only open markets
    markets = collector.fetch_markets(status="open")

    if not markets:
        print("\n[ERROR] No markets found. Exiting.")
        return collector

    # Step 2: Process data
    processor = KalshiDataProcessor()
    df = processor.markets_to_dataframe(markets)

    print(f"\n[OK] Created DataFrame with {len(df)} markets")

    # Calculate volume in USD
    df = processor.calculate_trading_volume_usd(df)

    # Keep original markets dataframe for later use
    markets_df = df.copy()

    # Aggregate by event_ticker to get total volume per event
    print(f"\n{'='*70}")
    print("AGGREGATING MARKETS BY EVENT")
    print(f"{'='*70}")

    if 'event_ticker' in df.columns:
        # Group by event and aggregate
        event_df = df.groupby('event_ticker').agg({
            'estimated_volume_usd': 'sum',
            'volume': 'sum',
            'ticker': 'count',  # Number of markets in this event
            'title': 'first',  # Use first market's title as representative
            'open_time': 'min',
            'close_time': 'max',
            'status': 'first'
        }).reset_index()

        event_df.columns = ['event_ticker', 'estimated_volume_usd', 'volume', 'market_count', 'title', 'open_time', 'close_time', 'status']

        print(f"Aggregated {len(df)} individual markets into {len(event_df)} events")
        print(f"Average markets per event: {len(df) / len(event_df):.1f}")

        # Use event_df for the rest of the analysis
        df = event_df
        df['ticker'] = df['event_ticker']  # For compatibility with downstream code
    else:
        print("Warning: event_ticker not found, analyzing individual markets")
        markets_df = df.copy()

    # Show volume distribution BEFORE filtering
    print(f"\n{'='*70}")
    print("VOLUME DISTRIBUTION (ALL EVENTS)")
    print(f"{'='*70}")
    print(f"Total events: {len(df)}")
    print(f"Total volume (all events): ${df['estimated_volume_usd'].sum():,.2f}")
    print(f"Max volume: ${df['estimated_volume_usd'].max():,.2f}")
    print(f"Median volume: ${df['estimated_volume_usd'].median():,.2f}")
    print(f"Mean volume: ${df['estimated_volume_usd'].mean():,.2f}")

    # Show volume thresholds
    print(f"\nEvents by volume threshold:")
    for threshold in [1_000, 10_000, 100_000, 500_000, 1_000_000, 2_000_000, 5_000_000]:
        count = len(df[df['estimated_volume_usd'] > threshold])
        print(f"  > ${threshold:>10,}: {count:>6} events")

    # Filter high volume events
    df_filtered = processor.filter_high_volume(df, VOLUME_THRESHOLD_USD)

    if df_filtered.empty:
        print("\n[ERROR] No events meet the volume threshold. Try lowering the threshold.")
        print(f"Suggestion: Try a lower threshold like $100,000")
        return collector

    # Display statistics for filtered events
    print(f"\n{'='*70}")
    print(f"EVENTS EXCEEDING ${VOLUME_THRESHOLD_USD:,}")
    print(f"{'='*70}")
    print(f"Total Events: {len(df_filtered)}")
    print(f"Total Volume: ${df_filtered['estimated_volume_usd'].sum():,.2f}")
    print(f"Average Volume per Event: ${df_filtered['estimated_volume_usd'].mean():,.2f}")
    print(f"Median Volume per Event: ${df_filtered['estimated_volume_usd'].median():,.2f}")
    print(f"Max Volume: ${df_filtered['estimated_volume_usd'].max():,.2f}")

    # Export all high volume events to CSV
    csv_filename = f'kalshi_events_over_{VOLUME_THRESHOLD_USD//1000}k.csv'

    # Select columns that exist after aggregation
    export_columns = ['ticker', 'title', 'estimated_volume_usd', 'volume', 'open_time', 'close_time', 'status']
    if 'market_count' in df_filtered.columns:
        export_columns.insert(4, 'market_count')  # Add market count if available

    df_filtered_export = df_filtered[export_columns].copy()
    df_filtered_export = df_filtered_export.sort_values('estimated_volume_usd', ascending=False)
    df_filtered_export.to_csv(csv_filename, index=False)
    print(f"\n[OK] Exported {len(df_filtered)} events to: {csv_filename}")

    # Display top 20 for reference
    print(f"\n{'='*70}")
    print(f"TOP 20 HIGHEST VOLUME EVENTS")
    print(f"{'='*70}")
    for idx, (_, row) in enumerate(df_filtered_export.head(20).iterrows(), 1):
        market_info = f" ({int(row['market_count'])} markets)" if 'market_count' in row else ""
        print(f"{idx:2d}. {row['ticker']:35s} | ${row['estimated_volume_usd']:>12,.0f}{market_info:15s} | {str(row['title'])[:35]}")

    # Create visualizations
    print(f"\n{'='*70}")
    print("CREATING VISUALIZATIONS")
    print(f"{'='*70}")
    visualizer = KalshiVisualizer()

    # Plot volume distribution for all events
    visualizer.plot_volume_distribution(df, top_n=30)

    # Plot volume by threshold
    visualizer.plot_volume_by_threshold(df)

    # Export historical data for top events and create price visualizations
    print(f"\n{'='*70}")
    print(f"EXPORTING HISTORICAL DATA FOR TOP {TOP_N_DETAILED_EXPORT} EVENTS")
    print(f"{'='*70}")

    top_event_tickers = df_filtered_export.head(TOP_N_DETAILED_EXPORT)['ticker'].tolist()

    # For each top event, find the highest volume individual market
    individual_market_tickers = []
    market_titles = {}  # Store market titles for visualization
    price_charts = []
    if top_event_tickers and 'event_ticker' in markets_df.columns:
        for event_ticker in top_event_tickers:
            # Find all markets in this event
            event_markets = markets_df[markets_df['event_ticker'] == event_ticker]
            if not event_markets.empty:
                # Get the market with highest volume in this event
                highest_vol_market = event_markets.nlargest(1, 'estimated_volume_usd')
                if not highest_vol_market.empty:
                    market_ticker = highest_vol_market.iloc[0]['ticker']
                    market_title = highest_vol_market.iloc[0].get('title', 'N/A')
                    individual_market_tickers.append(market_ticker)
                    market_titles[market_ticker] = market_title
                    print(f"  Event {event_ticker}: Using market {market_ticker} (${highest_vol_market.iloc[0]['estimated_volume_usd']:,.0f})")

    if individual_market_tickers:
        export_market_historical_data(collector, tickers=individual_market_tickers)

        # Create price visualizations for exported CSV files
        print(f"\n{'='*70}")
        print("CREATING PRICE HISTORY VISUALIZATIONS")
        print(f"{'='*70}")

        import os
        for ticker in individual_market_tickers:
            csv_file = f'market_data/{ticker}_historical_hourly.csv'
            if os.path.exists(csv_file):
                market_title = market_titles.get(ticker, None)
                visualizer.plot_price_history(csv_file, ticker, market_title)
                price_charts.append(f'kalshi_price_history_{ticker}.png')
                print(f"  [OK] Created price chart for {ticker}")

    print(f"\n{'='*70}")
    print("[OK] ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print(f"Generated files:")
    print(f"  - {csv_filename} ({len(df_filtered)} events over ${VOLUME_THRESHOLD_USD:,})")
    print(f"  - kalshi_events_volume_distribution.png (Top 30 events by volume)")
    print(f"  - kalshi_events_by_threshold.png (Events at different thresholds)")
    if individual_market_tickers:
        print(f"\nHistorical data CSVs (top {len(individual_market_tickers)} markets from top events):")
        for ticker in individual_market_tickers:
            print(f"  - market_data/{ticker}_historical_hourly.csv")
        if price_charts:
            print(f"\nPrice history visualizations:")
            for chart in price_charts:
                print(f"  - {chart}")
    print(f"{'='*70}")
    print(f"\nNote: Results are aggregated by EVENT (event_ticker).")
    print(f"Each event may contain multiple related markets.")
    print(f"\nTo visualize additional markets:")
    print(f"  # Visualize specific markets with hourly data:")
    print(f"  from kalshi_analysis import KalshiDataCollector, visualize_markets_from_csv")
    print(f"  collector = KalshiDataCollector()")
    print(f"  visualize_markets_from_csv(collector, tickers=['KXSB-26-PHI', 'KXTRUMPOUT-26-TRUMP'])")
    print(f"\n  # Visualize ALL 57 markets from CSV:")
    print(f"  visualize_markets_from_csv(collector, csv_file='{csv_filename}', all_markets=True)")
    print(f"\n  # Visualize with 1-minute data (per-minute volume, limited to 3 days):")
    print(f"  visualize_markets_from_csv(collector, tickers=['KXSB-26-PHI'], use_1min=True)")
    print(f"{'='*70}")

    return collector


if __name__ == "__main__":
    main()
