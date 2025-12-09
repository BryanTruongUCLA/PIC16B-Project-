"""
Download 6 Months of Kalshi Tesla Market Data (1-minute candles)
Finds all Tesla-related markets from Kalshi API
Validates expiration times AND volumes via webscraping Kalshi market pages
   - Extracts data from Next.js JSON embedded in page HTML
   - Corrects discrepancies between API and web data
Downloads 6 months of historical 1-minute candle data for each market
"""

import requests
import pandas as pd
import os
from datetime import datetime, timedelta
import time
import re

# Configuration
KALSHI_BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
KALSHI_WEB_URL = "https://kalshi.com"
MARKET_DATA_DIR = "market_data"

# Create market_data directory if it doesn't exist
os.makedirs(MARKET_DATA_DIR, exist_ok=True)

# Calculate date range 6 months back from time of running
end_date = datetime.now()
start_date = end_date - timedelta(days=180)

print(f"Date range: {start_date.date()} to {end_date.date()}")
print()

def fetch_all_markets():
    """Fetch all open markets from Kalshi API"""
    all_markets = []
    cursor = None
    page = 0

    while True:
        page += 1
        url = f"{KALSHI_BASE_URL}/markets"
        params = {
            'status': 'open',
            'limit': 1000
        }

        if cursor:
            params['cursor'] = cursor

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            markets = data.get('markets', [])
            all_markets.extend(markets)

            print(f"  Page {page}: Fetched {len(markets)} markets (Total: {len(all_markets)})")

            cursor = data.get('cursor')
            if not cursor:
                break

            time.sleep(0.2)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching markets: {e}")
            break

    return all_markets

def validate_market_expiration_via_webscrape(ticker, api_close_time, api_volume):
    """
    Validate market expiration time and volume by webscraping Kalshi market page
    Extracts data from Next.js hydration JSON embedded in the page
    Returns: corrected_close_time, corrected_volume, was_corrected, status
    """
    try:
        # Build Kalshi market URL - try both formats
        market_url = f"{KALSHI_WEB_URL}/markets/{ticker.lower()}"

        # Fetch the page
        response = requests.get(market_url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        if response.status_code != 200:
            return api_close_time, api_volume, False, f"HTTP_{response.status_code}"

        html = response.text

        # Look for embedded JSON data in script tags
        # Next.js embeds data in self.__next_f.push() calls
        import json

        # Search for close_time in the raw HTML
        close_time_matches = re.findall(r'"close_time"\s*:\s*"([^"]+)"', html)

        # Search for volume data
        volume_matches = re.findall(r'"volume"\s*:\s*(\d+)', html)

        # Extract close time
        web_close_time = None
        if close_time_matches:
            # Find the close_time that matches our ticker if multiple found
            for ct in close_time_matches:
                try:
                    parsed_time = pd.to_datetime(ct)
                    api_time = pd.to_datetime(api_close_time)
                    time_diff = abs((parsed_time - api_time).total_seconds())

                    # If itsb within 7 days its probably the same market we are talking about
                    if time_diff < 604800:
                        web_close_time = ct
                        break
                except:
                    continue

        # Extract volume
        web_volume = None
        if volume_matches:
            # Use the first volume found
            try:
                web_volume = int(volume_matches[0])
            except:
                pass

        # Compare and determine if corrections needed
        corrections_made = False
        corrections_list = []

        api_time = pd.to_datetime(api_close_time)

        # Check close time
        if web_close_time:
            web_time = pd.to_datetime(web_close_time)
            time_diff = abs((web_time - api_time).total_seconds())

            if time_diff > 60:
                print(f"Corrected time - API: {api_time}")
                print(f"Web: {web_time}")
                api_close_time = web_close_time
                corrections_made = True
                corrections_list.append("TIME")

        # Check volume
        if web_volume is not None and api_volume != web_volume:
            print(f"Corrected Volume - API: {api_volume:,}")
            print(f"Web: {web_volume:,}")
            api_volume = web_volume
            corrections_made = True
            corrections_list.append("VOLUME")

        if corrections_made:
            status = "CORRECTED_" + "_".join(corrections_list)
            return api_close_time, api_volume, True, status
        elif web_close_time or web_volume is not None:
            return api_close_time, api_volume, False, "MATCH"
        else:
            return api_close_time, api_volume, False, "NO_WEB_DATA"

    except Exception as e:
        return api_close_time, api_volume, False, f"ERROR: {str(e)[:50]}"

all_markets = fetch_all_markets()
print()

# Filter for Tesla markets
tesla_markets = []
for market in all_markets:
    title = market.get('title', '')
    title_lower = title.lower()

    # Check if it's about Tesla
    if 'tesla' in title_lower or 'tsla' in title_lower:
        tesla_markets.append(market)
        print(f"  Found: {title[:70]}")

print(f"\nFound {len(tesla_markets)} Tesla-related markets")
print()


# Validate expiration times and volumes via webscraping

corrections_made = 0
validation_stats = {}

for i, market in enumerate(tesla_markets, 1):
    ticker = market.get('ticker', '')
    title = market.get('title', 'Unknown')
    api_close_time = market.get('close_time', '')
    api_volume = market.get('volume', 0)

    if not api_close_time or not ticker:
        print(f"[{i}/{len(tesla_markets)}] {ticker} - SKIPPED (no close time)")
        continue

    print(f"[{i}/{len(tesla_markets)}] Validating {ticker[:30]}...")

    # Validate via webscraping
    corrected_time, corrected_volume, was_corrected, status = validate_market_expiration_via_webscrape(
        ticker, api_close_time, api_volume
    )

    # Track statistics
    if status.startswith('ERROR') or status.startswith('HTTP_'):
        validation_stats['ERROR'] = validation_stats.get('ERROR', 0) + 1
    else:
        validation_stats[status] = validation_stats.get(status, 0) + 1

    if was_corrected:
        # Update market data with corrected values
        market['close_time'] = corrected_time
        market['expiration_time'] = corrected_time
        market['volume'] = corrected_volume
        corrections_made += 1
    else:
        if status == 'MATCH':
            print(f"Data matches")
        else:
            print(f"Warning: {status}")

    # Rate limit
    time.sleep(1.0)

print()
print(f"Validated {len(tesla_markets)} markets")
print(f"Corrections made: {corrections_made}")
print(f"Validation results:")
for status_type, count in sorted(validation_stats.items()):
    if count > 0:
        print(f"- {status_type}: {count}")
print()

# Download 6 months of historical 1-minute data for each market
def fetch_market_history_3day_batches(ticker, title, series_ticker, close_time, expiration_time, start_dt, end_dt):
    """
    Fetch historical 1-minute candles for a market in 3-day batches
    Each batch downloads ALL 1-minute candles for that 3-day period
    """
    all_candles = []
    current_start = start_dt
    batch_num = 0

    # Calculate total number of batches for progress tracking
    total_days = (end_dt - start_dt).days
    total_batches = (total_days + 2) // 3  # Round up

    while current_start < end_dt:
        batch_num += 1
        # 3-day batch (or remaining days if less than 3)
        current_end = min(current_start + timedelta(days=3), end_dt)

        # Convert to Unix timestamps
        from_ts = int(current_start.timestamp())
        to_ts = int(current_end.timestamp())

        # Kalshi Market Candlesticks endpoint
        history_url = f"{KALSHI_BASE_URL}/series/{series_ticker}/markets/{ticker}/candlesticks"
        params = {
            'start_ts': from_ts,
            'end_ts': to_ts,
            'period_interval': 1  # 1 minute candles
        }

        try:
            response = requests.get(history_url, params=params)

            if response.status_code == 200:
                data = response.json()
                # Response has ticker and candlesticks array
                candles_batch = data.get('candlesticks', [])

                if candles_batch:
                    # Convert to our format
                    for candle in candles_batch:
                        # Parse nested bid/ask prices (API returns prices in cents)
                        yes_bid = candle.get('yes_bid', {})
                        yes_ask = candle.get('yes_ask', {})

                        row = {
                            'datetime': pd.to_datetime(candle.get('end_period_ts'), unit='s'),
                            'market_ticker': ticker,
                            'title': title,
                            'close_time': close_time,
                            'expiration_time': expiration_time,
                            'yes_bid_close': yes_bid.get('close') if yes_bid else None,
                            'yes_ask_close': yes_ask.get('close') if yes_ask else None,
                            'volume': candle.get('volume', 0),
                            'open_interest': candle.get('open_interest', 0)
                        }
                        all_candles.append(row)

                    print(f"    Batch {batch_num}/{total_batches} ({current_start.date()} to {current_end.date()}): {len(candles_batch):,} 1-min candles | Total so far: {len(all_candles):,}")
                else:
                    print(f"    Batch {batch_num}/{total_batches} ({current_start.date()} to {current_end.date()}): No data")

            else:
                print(f"    Batch {batch_num}/{total_batches}: Error {response.status_code}")

        except Exception as e:
            print(f"    Batch {batch_num}/{total_batches}: Error - {e}")

        # Move to next 3-day batch
        current_start = current_end

    return all_candles

# Download historical data for each market
for i, market in enumerate(tesla_markets, 1):
    ticker = market.get('ticker', '')
    title = market.get('title', 'Unknown')
    close_time = market.get('close_time', '')
    expiration_time = market.get('expiration_time', '')

    # Get event_ticker
    event_ticker = market.get('event_ticker', ticker)

    if not ticker or not event_ticker:
        print(f"[{i}/{len(tesla_markets)}] {title[:70]}")
        print(f"  [SKIP] Missing ticker or event_ticker (ticker={ticker})")
        print()
        continue

    print(f"[{i}/{len(tesla_markets)}] {title[:70]}")
    print(f"  Ticker: {ticker}")
    print(f"  Event: {event_ticker}")


    # Fetch 6 months of historical data in 3-day batches
    candles = fetch_market_history_3day_batches(ticker, title, event_ticker, close_time, expiration_time, start_date, end_date)


    # Create dataframe
    df = pd.DataFrame(candles)
    df = df.sort_values('datetime').reset_index(drop=True)

    # Forward-fill missing minutes to create complete 1-minute dataset
    # Keeps bid/ask spread changes even when there are no trades
    min_time = df['datetime'].min()
    max_time = df['datetime'].max()

    # Create complete 1-minute range
    complete_range = pd.date_range(start=min_time, end=max_time, freq='1min')

    # Reindex and forward-fill
    df_complete = df.set_index('datetime').reindex(complete_range)
    df_complete['market_ticker'] = ticker
    df_complete['title'] = title
    df_complete['close_time'] = close_time
    df_complete['expiration_time'] = expiration_time
    df_complete = df_complete.ffill()
    df_complete = df_complete.reset_index()
    df_complete = df_complete.rename(columns={'index': 'datetime'})

    # Save to CSV
    filename = os.path.join(MARKET_DATA_DIR, f"{ticker}_historical_1min.csv")
    df_complete.to_csv(filename, index=False)

    print(f"  [OK] Saved {len(df_complete):,} candles to {filename} (imputed from {len(df):,} sparse candles)")
    print(f"       Date range: {df_complete['datetime'].min()} to {df_complete['datetime'].max()}")
    print()


print(f"Downloaded {len(tesla_markets)} Tesla market files")
print(f"Saved to: {MARKET_DATA_DIR}/")