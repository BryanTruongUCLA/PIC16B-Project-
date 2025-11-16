"""
Process Tesla Volume-Weighted Aggregate
Uses ChatGPT to determine bullish direction
Uses 24-hour rolling volume weights and quantity-weighted averaging
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import time
from datetime import datetime, timedelta

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai")
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

OPENAI_API_KEY = "API-Key-Here"

MARKET_DATA_DIR = "market_data"
DIRECTION_CACHE_FILE = "market_directions_cache.json"

# Initialize OpenAI client
openai_client = OpenAI(api_key=OPENAI_API_KEY)


def load_direction_cache():
    """Load cached market directions from file"""
    if os.path.exists(DIRECTION_CACHE_FILE):
        with open(DIRECTION_CACHE_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_direction_cache(cache):
    """Save market directions to cache file"""
    with open(DIRECTION_CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


def determine_market_direction(ticker: str, title: str, cache: dict) -> str:
    """
    Determine if YES resolving increases Tesla stock price based on manual rules.

    Returns:
        'yes' if YES resolving would increase Tesla stock price
        'no' if NO resolving would increase Tesla stock price (YES would decrease it)
        'neutral' if the market is not relevant or unclear - will be excluded
    """
    # Check cache first
    if ticker in cache:
        return cache[ticker]

    # Manual direction determination based on market title keywords
    title_lower = title.lower()

    # Markets where YES = bullish (good for Tesla stock)
    bullish_yes_keywords = [
        'delivery', 'deliveries', 'production', 'produce', 'revenue', 'profit',
        'sales', 'growth', 'increase', 'above', 'higher', 'factory', 'open',
        'robotaxi', 'fsd', 'optimus', 'roadster', 'cybertruck production',
        'semi truck', 'energy business', 'location open', 'drive-in'
    ]

    # Markets where NO = bullish (bad things not happening)
    bullish_no_keywords = [
        'discontinue', 'tariff', 'ceo', 'musk out', 'fail', 'recall',
        'lawsuit', 'investigation', 'ban', 'restriction'
    ]

    # Check for bullish NO keywords first (these are more specific)
    for keyword in bullish_no_keywords:
        if keyword in title_lower:
            direction = 'no'
            cache[ticker] = direction
            save_direction_cache(cache)
            return direction

    # Check for bullish YES keywords
    for keyword in bullish_yes_keywords:
        if keyword in title_lower:
            direction = 'yes'
            cache[ticker] = direction
            save_direction_cache(cache)
            return direction

    # If no clear match, mark as neutral (exclude from analysis)
    print(f"  [WARN] Could not determine direction for {ticker}: '{title}' - marking as neutral")
    direction = 'neutral'
    cache[ticker] = direction
    save_direction_cache(cache)
    return direction


def validate_market_merge(titles: list, cache: dict) -> bool:
    """
    Use ChatGPT to determine if markets should be merged together.

    Args:
        titles: List of market titles to check for merging
        cache: Cache dict for storing validation results

    Returns:
        True if markets should be merged, False otherwise
    """
    # Create cache key from sorted titles
    cache_key = "||".join(sorted(titles))

    # Check cache first
    if cache_key in cache:
        return cache[cache_key]

    # Create a formatted list of titles for the prompt
    titles_formatted = "\n".join([f"{i+1}. {title}" for i, title in enumerate(titles)])

    prompt = f"""Are these Kalshi prediction markets asking the same fundamental question (just with different quantities or thresholds)?

Markets:
{titles_formatted}

Answer with exactly one word:
- "yes" if they ask the same question with different quantities/thresholds (e.g., "Tesla above 450,000" vs "Tesla above 500,000")
- "no" if they are fundamentally different questions"""

    try:
        response = openai_client.chat.completions.create(
            model="gpt-5-nano",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers with only 'yes' or 'no'. Never explain, just answer yes or no."},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=10
        )

        answer = response.choices[0].message.content.strip().lower()
        should_merge = 'yes' in answer

        # Cache the result
        cache[cache_key] = should_merge
        save_direction_cache(cache)

        return should_merge

    except Exception as e:
        print(f"  [ERROR] ChatGPT API error for merge validation: {e}")
        return False  # Default to not merging if API fails


def detect_duplicate_markets(markets_df: pd.DataFrame) -> dict:
    """
    Detect markets that are duplicates with different quantities.
    Groups them together for combined processing.

    Returns:
        dict mapping base_title -> list of (ticker, title, volume_usd, quantity)
    """
    import re

    # Extract base question and quantity from titles
    market_groups = {}

    for _, row in markets_df.iterrows():
        ticker = row['ticker']
        title = row['title']
        volume_usd = row.get('volume_usd', 0)

        # Try to extract quantity patterns like "450,000", "500000", "$500k", etc.
        # Remove common quantity patterns to get base question
        base_title = re.sub(r'\b\d{1,3}(,\d{3})+\b', 'N', title)  # 450,000 -> N
        base_title = re.sub(r'\b\d{4,}\b', 'N', base_title)  # 500000 -> N
        base_title = re.sub(r'\$\d+[km]?\b', 'N', base_title, flags=re.IGNORECASE)  # $500k -> N
        base_title = base_title.strip()

        # Try to extract the actual quantity
        quantity_match = re.search(r'\b(\d{1,3}(?:,\d{3})+|\d{4,})\b', title)
        quantity = int(quantity_match.group(1).replace(',', '')) if quantity_match else None

        if base_title not in market_groups:
            market_groups[base_title] = []

        market_groups[base_title].append({
            'ticker': ticker,
            'title': title,
            'volume_usd': volume_usd,
            'quantity': quantity
        })

    # Only return groups with multiple markets
    duplicates = {k: v for k, v in market_groups.items() if len(v) > 1}

    return duplicates


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

# Detect duplicate markets with different quantities
print("Detecting duplicate markets with different quantities...")
duplicates = detect_duplicate_markets(tesla_markets)

if duplicates:
    print(f"Found {len(duplicates)} market groups with multiple quantities:")
    for base_title, markets in duplicates.items():
        if len(markets) > 1:
            print(f"\n  Base question: {base_title[:80]}...")
            for m in markets:
                qty_str = f"qty={m['quantity']:,}" if m['quantity'] else "no qty"
                print(f"    - {m['ticker']}: {qty_str}, ${m['volume_usd']:,.0f} volume")
    print()
else:
    print("No duplicate markets detected")
    print()

# Build market info list for Tesla
tesla_market_data = []
for _, row in tesla_markets.iterrows():
    ticker = row['ticker']
    csv_path = os.path.join(MARKET_DATA_DIR, f"{ticker}_historical_1min.csv")

    if os.path.exists(csv_path):
        tesla_market_data.append({
            'ticker': ticker,
            'title': row.get('title', ''),
            'volume_usd': row.get('volume_usd', 0),
            'csv_path': csv_path
        })

print(f"Found {len(tesla_market_data)} Tesla markets with historical data")
print()

# Calculate volume-weighted aggregate
print("="*70)
print("CALCULATING VOLUME-WEIGHTED AGGREGATE FOR TESLA")
print("="*70)

# Load direction cache
direction_cache = load_direction_cache()

# Step 1: Determine market directions using keyword-based rules
print(f"\nStep 1: Determining bullish direction for each market...")
market_directions = {}
excluded_markets = []
for market_info in tesla_market_data:
    ticker = market_info['ticker']
    title = market_info['title']
    direction = determine_market_direction(ticker, title, direction_cache)
    market_directions[ticker] = direction

    if direction == 'neutral':
        print(f"  {ticker}: NEUTRAL - excluding from analysis")
        excluded_markets.append(ticker)
    else:
        print(f"  {ticker}: {direction.upper()} side is bullish")

if excluded_markets:
    print(f"\n  [INFO] Excluded {len(excluded_markets)} neutral markets from analysis")

# Step 2: Load all market CSVs
print(f"\nStep 2: Loading market data...")
all_dfs = []

for market_info in tesla_market_data:
    csv_path = market_info['csv_path']
    ticker = market_info['ticker']
    direction = market_directions[ticker]

    # Skip neutral markets
    if direction == 'neutral':
        continue

    try:
        df = pd.read_csv(csv_path)

        if df.empty:
            print(f"  [SKIP] {ticker}: Empty CSV")
            continue

        # Parse datetime
        df['datetime'] = pd.to_datetime(df['datetime'])

        # Select price based on direction
        if direction == 'yes':
            df['price'] = df['yes_bid_close_usd']  # YES side price
        elif direction == 'no':
            df['price'] = 1.0 - df['yes_ask_close_usd']  # NO side price (inverse of yes)

        # Add market identifier
        df['ticker'] = ticker

        # Keep only needed columns
        df = df[['datetime', 'ticker', 'price', 'volume']].copy()

        all_dfs.append(df)
        print(f"  [OK] {ticker}: {len(df)} rows ({direction} side)")

    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        continue

if not all_dfs:
    print("[ERROR] No valid market data loaded for Tesla")
    exit(1)

# Step 2.5: Merge duplicate markets with different quantities
print(f"\nStep 2.5: Merging duplicate markets...")

# Build a mapping of ticker to base question (for duplicates)
ticker_to_base = {}
for base_title, markets in duplicates.items():
    if len(markets) > 1:
        for m in markets:
            ticker_to_base[m['ticker']] = base_title

# Group dataframes by base question
merged_dfs = []
processed_bases = set()

for df in all_dfs:
    ticker = df['ticker'].iloc[0] if len(df) > 0 else None

    if not ticker:
        continue

    # Check if this is part of a duplicate group
    base_title = ticker_to_base.get(ticker)

    if base_title and base_title not in processed_bases:
        # This is a duplicate - find all related markets
        related_markets = [m for m in duplicates[base_title]]
        related_tickers = [m['ticker'] for m in related_markets]
        related_titles = [m['title'] for m in related_markets]
        related_quantities = {m['ticker']: m['quantity'] for m in related_markets if m['quantity']}

        # Get all dataframes for this group
        group_dfs = [d for d in all_dfs if d['ticker'].iloc[0] in related_tickers]

        if len(group_dfs) > 1 and related_quantities:
            # Validate merge using ChatGPT
            print(f"  Validating merge for {len(group_dfs)} markets: {base_title[:60]}...")
            should_merge = validate_market_merge(related_titles, direction_cache)

            if should_merge:
                print(f"    ✓ ChatGPT confirmed merge is valid")

                # Merge by averaging prices relative to quantity changes
                # Combine all on same timestamps
                merged = pd.concat(group_dfs, ignore_index=True)
                merged['quantity'] = merged['ticker'].map(related_quantities)

                # For each timestamp, calculate quantity-weighted average price
                def quantity_weighted_avg(group):
                    if group['quantity'].isna().all():
                        return pd.Series({'price': group['price'].mean(), 'volume': group['volume'].sum()})

                    total_qty = group['quantity'].sum()
                    if total_qty == 0:
                        return pd.Series({'price': group['price'].mean(), 'volume': group['volume'].sum()})

                    # Weight by quantity
                    weighted_price = (group['price'] * group['quantity']).sum() / total_qty
                    total_volume = group['volume'].sum()

                    return pd.Series({'price': weighted_price, 'volume': total_volume})

                merged_result = merged.groupby('datetime').apply(quantity_weighted_avg, include_groups=False).reset_index()
                merged_result['ticker'] = related_tickers[0]  # Use first ticker as representative

                merged_dfs.append(merged_result)
                processed_bases.add(base_title)

                print(f"    → Combined into {len(merged_result)} rows (quantity-weighted average)")
            else:
                print(f"    ✗ ChatGPT rejected merge - keeping markets separate")
                # Add all markets separately
                for gdf in group_dfs:
                    merged_dfs.append(gdf)
                processed_bases.add(base_title)

        elif len(group_dfs) == 1:
            # Only one market in this group has data
            merged_dfs.append(group_dfs[0])
            processed_bases.add(base_title)

    elif base_title is None:
        # Not a duplicate - include as-is
        merged_dfs.append(df)

print(f"  Result: {len(all_dfs)} markets → {len(merged_dfs)} after merging duplicates")
print()

# Step 3: Combine all market data
print(f"Step 3: Combining data from {len(merged_dfs)} markets...")
combined_df = pd.concat(merged_dfs, ignore_index=True)
combined_df = combined_df.sort_values('datetime')

# Step 4: Calculate 7-day rolling volume for each market
print(f"Step 4: Calculating 7-day rolling volume weights...")

# Group by ticker and calculate rolling 7-day volume
combined_df['volume_7d'] = 0.0
for ticker in combined_df['ticker'].unique():
    mask = combined_df['ticker'] == ticker
    ticker_df = combined_df[mask].copy()
    ticker_df = ticker_df.set_index('datetime')

    # Rolling 7-day sum of volume (7 days = 168 hours)
    ticker_df['volume_7d'] = ticker_df['volume'].rolling(
        window='7D', min_periods=1
    ).sum()

    combined_df.loc[mask, 'volume_7d'] = ticker_df['volume_7d'].values

# Step 5: Calculate weights based on 7-day volume at each timestamp
print(f"Step 5: Calculating time-varying weights...")

# For each datetime, calculate total 7-day volume across all markets
total_volume_7d = combined_df.groupby('datetime')['volume_7d'].transform('sum')
combined_df['weight'] = combined_df['volume_7d'] / total_volume_7d

# Handle zero weights
combined_df['weight'] = combined_df['weight'].fillna(0)

# Remove rows with NaN prices or zero weights
combined_df = combined_df.dropna(subset=['price'])
combined_df = combined_df[combined_df['weight'] > 0]

if combined_df.empty:
    print("[ERROR] No valid prices after filtering for Tesla")
    exit(1)

# Step 6: Calculate weighted aggregate price at each timestamp
print(f"Step 6: Calculating weighted aggregate across {len(combined_df)} data points...")

# Calculate weighted contribution
combined_df['weighted_contribution'] = combined_df['price'] * combined_df['weight']

# Group by datetime and sum weighted contributions
result = combined_df.groupby('datetime').agg({
    'weighted_contribution': 'sum',
    'weight': 'sum',
    'ticker': 'count'
}).reset_index()

result.rename(columns={'ticker': 'num_markets'}, inplace=True)

# Calculate final weighted price (already weighted, just need to ensure normalization)
result['weighted_price'] = result['weighted_contribution'] / result['weight']

# Remove invalid rows
result = result.dropna(subset=['weighted_price'])
result = result.sort_values('datetime')

print(f"[OK] Generated {len(result)} weighted aggregate data points")
print(f"     Date range: {result['datetime'].min()} to {result['datetime'].max()}")
print(f"     Price range: ${result['weighted_price'].min():.4f} to ${result['weighted_price'].max():.4f}")
print(f"{'='*70}\n")

# Export to CSV
output_csv = 'stock_aggregate_weighted_prices.csv'
result[['datetime', 'weighted_price']].to_csv(output_csv, index=False)
print(f"[OK] Exported Tesla weighted prices to: {output_csv}")
print(f"     {len(result)} data points\n")

# Create visualization
print("Creating visualization...")
plt.figure(figsize=(16, 8))

# Plot the weighted price
plt.plot(result['datetime'], result['weighted_price'],
         linewidth=1.5, alpha=0.8, color='#2E86AB', label='Volume-Weighted Price (7-day rolling)')

plt.title(f'Tesla - Volume-Weighted Aggregate Price\n'
          f'{len(result):,} data points from {result["datetime"].min().strftime("%Y-%m-%d")} to {result["datetime"].max().strftime("%Y-%m-%d")}',
          fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Weighted Aggregate Price (USD)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=10)
plt.tight_layout()

output_png = 'stock_aggregate_Tesla.png'
plt.savefig(output_png, dpi=150, bbox_inches='tight')
plt.close()

print(f"[OK] Saved visualization to: {output_png}")

# Create 7-day rolling volume chart for each market
print("\nCreating 7-day rolling volume chart for each market...")
plt.figure(figsize=(16, 10))

# Get unique tickers and plot each market's 7-day rolling volume
for ticker in combined_df['ticker'].unique():
    ticker_data = combined_df[combined_df['ticker'] == ticker].copy()
    ticker_data = ticker_data.sort_values('datetime')
    plt.plot(ticker_data['datetime'], ticker_data['volume_7d'],
             linewidth=1.5, alpha=0.7, label=ticker)

plt.title(f'Tesla Markets - 7-Day Rolling Volume\n'
          f'All {len(combined_df["ticker"].unique())} markets',
          fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('7-Day Rolling Volume', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8, loc='best', ncol=2)
plt.tight_layout()

output_volume_png = 'stock_aggregate_Tesla_7day_volumes.png'
plt.savefig(output_volume_png, dpi=150, bbox_inches='tight')
plt.close()

print(f"[OK] Saved 7-day volume chart to: {output_volume_png}")
print()
print("="*70)
print("TESLA ANALYSIS COMPLETE")
print("="*70)
print("\nGenerated files:")
print(f"  1. {output_csv} - Tesla weighted aggregate prices")
print(f"  2. {output_png} - Tesla price visualization")
print(f"  3. {output_volume_png} - Tesla 7-day rolling volumes by market")
print()
