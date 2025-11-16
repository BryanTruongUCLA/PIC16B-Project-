# Kalshi Stock Market Analysis & Tesla Prediction Market Aggregation

This project analyzes prediction markets from Kalshi (a CFTC-regulated prediction market platform) to create volume-weighted aggregate signals for stock price movements, with a specific focus on Tesla (TSLA). The system fetches market data, normalizes it with stock prices, and uses sophisticated weighting algorithms to create a unified market sentiment indicator.

## Overview

The project combines Kalshi prediction markets with real stock market data to:
1. Identify high-volume stock-related prediction markets
2. Download historical trading data for Tesla-related markets
3. Normalize stock prices using S&P 500 as a benchmark
4. Aggregate multiple prediction markets using volume-weighted averaging
5. Create visualizations showing market sentiment trends

---

## Project Structure

### Python Files

| File | Purpose |
|------|---------|
| `kalshi_analysis.py` | Core library for fetching and analyzing Kalshi markets |
| `fetch_stock_markets.py` | Identifies and verifies stock-related markets |
| `download_tesla_data.py` | Downloads 1-minute historical data for Tesla markets |
| `fetch_normalized_tesla_stock.py` | Fetches and normalizes Tesla stock data |
| `process_tesla_aggregate.py` | Calculates volume-weighted aggregate prices |

### Data Files (Generated)

| File | Created By | Contains |
|------|-----------|----------|
| `all_stock_related_markets.csv` | `fetch_stock_markets.py` | All verified stock-related markets |
| `normalized_tesla_stock_1min.csv` | `fetch_normalized_tesla_stock.py` | Tesla/SPY normalized prices |
| `stock_aggregate_weighted_prices.csv` | `process_tesla_aggregate.py` | Final weighted aggregate |
| `market_data/*.csv` | Various scripts | Individual market historical data |
| `market_directions_cache.json` | `process_tesla_aggregate.py` | Market direction classifications |

---

## How to Run (Recommended Order)

### Prerequisites

```bash
pip install pandas matplotlib seaborn requests openai beautifulsoup4
```

### Step 1: Fetch Stock-Related Markets

```bash
python fetch_stock_markets.py
```

**What it does:**
- Fetches all open markets from Kalshi API
- Filters for markets mentioning major companies (Tesla, Apple, etc.)
- Verifies each market is truly open using:
  - API status field
  - Close time validation
  - Trading volume check
  - Website scraping verification
- Exports verified markets to `all_stock_related_markets.csv`

**Output:**
- `all_stock_related_markets.csv` - List of verified stock-related prediction markets
- Console report showing market counts by company

**Time:** ~5-10 minutes (due to verification checks)

---

### Step 2: Download Tesla Market Data

```bash
python download_tesla_data.py
```

**What it does:**
- Loads verified Tesla markets from Step 1
- Downloads 1-minute historical candlestick data for each market
- Implements automatic chunking for extended time periods (API limit: 3 days per request)
- Fills gaps in data with forward-filled prices
- Exports individual market CSVs to `market_data/` directory

**Output:**
- `market_data/*_historical_1min.csv` - One file per Tesla market
- Each file contains: timestamp, open, high, low, close, volume, bid/ask prices

**Time:** ~5-15 minutes (depends on number of markets and data range)

---

### Step 3: Fetch Normalized Tesla Stock Data

```bash
python fetch_normalized_tesla_stock.py
```

**What it does:**
- Scans market data to determine time range
- Fetches 1-minute TSLA and SPY (S&P 500) data from EODHD API
- Normalizes Tesla price using formula: `normalized_price = TSLA / SPY`
- Creates visualization of TSLA, SPY, and normalized prices
- Exports to CSV

**Why normalize with S&P 500?**
- Removes market-wide movements (e.g., entire market going up/down)
- Isolates Tesla-specific price movements
- Enables better beta calculation and market-relative performance analysis

**Output:**
- `normalized_tesla_stock_1min.csv` - Tesla prices normalized by S&P 500
- `normalized_tesla_stock_1min.png` - Three-panel visualization

**Time:** ~1-2 minutes

---

### Step 4: Calculate Volume-Weighted Aggregate

```bash
python process_tesla_aggregate.py
```

**What it does:**
- Loads all Tesla market data
- Determines bullish direction for each market using keyword matching
- Detects and merges duplicate markets with different quantities
- Calculates 7-day rolling volume weights
- Aggregates all markets using volume-weighted averaging
- Exports final aggregate price series

**Output:**
- `stock_aggregate_weighted_prices.csv` - Final weighted aggregate price
- `stock_aggregate_Tesla.png` - Aggregate price visualization
- `stock_aggregate_Tesla_7day_volumes.png` - Volume trends by market
- `market_directions_cache.json` - Cached market classifications

**Time:** ~2-5 minutes

---

## Mathematical Formula: Volume-Weighted Aggregation

The core innovation of this project is the **7-day rolling volume-weighted aggregation** algorithm.

### The Weighting Formula

For each timestamp `t`, the aggregate price is calculated as:

```
P_aggregate(t) = Σ [P_i(t) × w_i(t)]
```

Where:
- `P_i(t)` = Price of market `i` at time `t`
- `w_i(t)` = Weight of market `i` at time `t`
- The sum is over all markets `i`

### Weight Calculation

The weight for market `i` at time `t` is:

```
w_i(t) = V_i^7d(t) / Σ V_j^7d(t)
          ─────────────────────
           (sum over all markets j)

where:
V_i^7d(t) = 7-day rolling sum of volume for market i at time t
```

In other words:
- `V_i^7d(t)` = Total trading volume in market `i` over the past 7 days ending at time `t`
- The weight is the **proportion** of total 7-day volume that market `i` represents

### Step-by-Step Calculation

#### 1. **Determine Market Direction**

For each market, classify whether "YES" or "NO" is bullish for Tesla stock:

- **YES is bullish** for markets like:
  - "Tesla deliveries above 500,000"
  - "Tesla revenue exceeds $100B"
  - "Cybertruck production starts"

- **NO is bullish** for markets like:
  - "Tesla recalls over 100,000 vehicles" (NO = no recall = good)
  - "Elon Musk steps down as CEO" (NO = stays as CEO = good for stock)
  - "Tesla faces tariffs" (NO = no tariffs = good)

The selected price is:
```
P_i(t) = {
    yes_bid_price(t)           if YES is bullish
    1 - yes_ask_price(t)       if NO is bullish (inverse)
}
```

#### 2. **Calculate 7-Day Rolling Volume**

For each market `i`, calculate the 7-day rolling volume at each timestamp:

```
V_i^7d(t) = Σ volume_i(s)
            s ∈ [t-7 days, t]
```

This gives more weight to markets with recent trading activity (indicating current relevance).

#### 3. **Normalize Weights to Sum to 1**

At each timestamp `t`, normalize weights across all markets:

```
Total_Volume^7d(t) = Σ V_i^7d(t)  (sum over all markets i)

w_i(t) = V_i^7d(t) / Total_Volume^7d(t)

Therefore: Σ w_i(t) = 1  (weights sum to 100%)
```

#### 4. **Calculate Weighted Aggregate Price**

```
P_aggregate(t) = Σ [P_i(t) × w_i(t)]
                 i=1 to N

where N = number of markets at time t
```

### Example Calculation

Suppose at time `t` we have 3 Tesla markets:

| Market | Description | Direction | Price P_i(t) | 7-day Volume V^7d(t) |
|--------|------------|-----------|--------------|---------------------|
| Market A | "Deliveries above 500k" | YES bullish | 0.65 | 100,000 |
| Market B | "Deliveries above 450k" | YES bullish | 0.78 | 50,000 |
| Market C | "CEO steps down" | NO bullish | 0.85 | 150,000 |

**Step 1:** Calculate total 7-day volume
```
Total = 100,000 + 50,000 + 150,000 = 300,000
```

**Step 2:** Calculate weights
```
w_A(t) = 100,000 / 300,000 = 0.333
w_B(t) =  50,000 / 300,000 = 0.167
w_C(t) = 150,000 / 300,000 = 0.500
```

**Step 3:** Calculate aggregate price
```
P_aggregate(t) = (0.65 × 0.333) + (0.78 × 0.167) + (0.85 × 0.500)
               = 0.217 + 0.130 + 0.425
               = 0.772
```

### Why This Formula Works

1. **Time-Varying Weights**: Markets that are currently active get higher weights; inactive markets fade away naturally
2. **Volume as Signal**: High trading volume indicates market confidence and information flow
3. **7-Day Window**: Balances recent activity vs. smoothing noise (adjustable parameter)
4. **Direction-Aware**: Correctly interprets whether YES/NO is bullish
5. **Normalized Output**: Always between 0 and 1, representing aggregate bullish sentiment

### Duplicate Market Handling

For markets asking the same question with different quantities (e.g., "Deliveries above 450k" vs "Deliveries above 500k"), the system:

1. **Detects duplicates** using regex pattern matching
2. **Validates with ChatGPT** whether markets should be merged
3. **Merges using quantity weighting**:

```
P_merged(t) = Σ [P_k(t) × Q_k] / Σ Q_k
              k ∈ duplicate_set

where Q_k = quantity threshold for market k
```

Example: Markets with thresholds 450k, 500k, 550k get weighted by their quantities when averaged.

---

## General Usage: Analyzing Other Markets

While this project focuses on Tesla, you can analyze any stock-related markets.

### Analyze Specific Markets

```python
from kalshi_analysis import KalshiDataCollector, visualize_markets_from_csv

collector = KalshiDataCollector()

# Visualize specific market tickers
visualize_markets_from_csv(
    collector,
    tickers=['KXSB-26-PHI', 'KXTRUMPOUT-26-TRUMP'],
    interval='hourly',
    months_back=3
)
```

### Analyze All High-Volume Events

```python
# Visualize all markets from the CSV
visualize_markets_from_csv(
    collector,
    csv_file='kalshi_events_over_2000k.csv',
    all_markets=True
)
```

### Use Different Time Intervals

```python
# 1-minute data (best for short-term, limited to 3 days per API call but auto-chunks)
visualize_markets_from_csv(collector, tickers=['TICKER'], interval='1min', months_back=1)

# Hourly data (good for medium-term, up to 3 months)
visualize_markets_from_csv(collector, tickers=['TICKER'], interval='hourly', months_back=3)

# Daily data (best for long-term analysis)
visualize_markets_from_csv(collector, tickers=['TICKER'], interval='daily', months_back=12)
```

---

## API Keys Required

### 1. EODHD API (for stock data)

- **File:** `fetch_normalized_tesla_stock.py`
- **Variable:** `EODHD_API_KEY`
- **Get key:** [https://eodhd.com/](https://eodhd.com/)
- **Used for:** Fetching Tesla and S&P 500 1-minute price data

### 2. OpenAI API (for market merging validation)

- **File:** `process_tesla_aggregate.py`
- **Variable:** `OPENAI_API_KEY`
- **Get key:** [https://platform.openai.com/](https://platform.openai.com/)
- **Used for:** Validating whether duplicate markets should be merged

**Note:** The current code contains placeholder keys that need to be replaced with your own API keys.

---

## Key Features

### Comprehensive Market Verification

The system doesn't just trust the API's "open" status. It performs multi-step verification:
1. Checks API status field
2. Validates close time hasn't passed
3. Confirms market has trading volume
4. Scrapes market webpage for closure indicators

### Automatic Data Gap Filling

When fetching historical data, the system:
- Detects missing time periods
- Forward-fills prices (last known price)
- Sets volume to 0 for imputed periods
- Marks data source ('api' vs 'imputed') for transparency

### Intelligent Market Merging

- Detects markets asking the same question with different quantities
- Uses ChatGPT to validate merge decisions
- Applies quantity-weighted averaging for merged markets
- Prevents over-weighting duplicate information

### Time-Varying Volume Weights

Unlike simple averaging, the 7-day rolling window means:
- Recent trading activity matters more
- Markets naturally fade in importance as they become inactive
- System adapts to changing market conditions

---

## Output Files Explained

### 1. `stock_aggregate_weighted_prices.csv`

**Columns:**
- `datetime`: Timestamp (1-minute intervals)
- `weighted_price`: Aggregate bullish sentiment (0 to 1 scale)

**Interpretation:**
- Higher values = More bullish sentiment for Tesla
- Lower values = More bearish sentiment for Tesla
- Compare with actual Tesla stock movements to assess predictive power

### 2. `normalized_tesla_stock_1min.csv`

**Columns:**
- `timestamp`: Datetime
- `tsla_close`: Raw Tesla closing price
- `spy_close`: S&P 500 closing price
- `normalized_close`: TSLA / SPY ratio
- `tsla_volume`, `spy_volume`: Trading volumes

**Use cases:**
- Calculate beta between prediction markets and stock
- Remove market-wide noise from Tesla analysis
- Identify Tesla-specific movements

### 3. `all_stock_related_markets.csv`

**Columns:**
- `ticker`, `event_ticker`: Market identifiers
- `title`, `subtitle`: Market descriptions
- `company`: Detected company name
- `volume_usd`: Total trading volume in USD
- `yes_bid`, `yes_ask`, `last_price`: Current market prices
- `open_time`, `close_time`: Market active period

**Use cases:**
- Find markets for other stocks (Apple, Microsoft, etc.)
- Identify high-liquidity markets
- Track market creation/expiration dates

---

## Troubleshooting

### "No Tesla markets found in CSV"

**Solution:** Run `fetch_stock_markets.py` first to generate the verified markets file.

### "EODHD API error"

**Solution:**
1. Check your API key is valid
2. Ensure you have sufficient API quota
3. Verify the time range isn't too large for your plan

### "ChatGPT API error for merge validation"

**Solution:**
1. Check OpenAI API key
2. Verify you have API credits
3. Model "gpt-5-nano" may need to be changed to "gpt-4o-mini" or "gpt-3.5-turbo" depending on availability

### Markets showing as "neutral" (excluded)

**Explanation:** The keyword-based direction classifier couldn't determine if YES or NO is bullish. These markets are excluded from aggregation.

**Solution:**
- Add relevant keywords to `bullish_yes_keywords` or `bullish_no_keywords` in `process_tesla_aggregate.py`
- Or manually add direction to `market_directions_cache.json`

### Data files already exist

Most scripts check if output files exist and skip regeneration. Delete the files if you want fresh data:
```bash
# Delete all generated data (start fresh)
rm all_stock_related_markets.csv
rm normalized_tesla_stock_1min.csv
rm stock_aggregate_weighted_prices.csv
rm -rf market_data/
```

---

## Technical Notes

### API Rate Limiting

The code includes automatic rate limiting:
- 0.1s delay between market fetches
- 0.2s delay for verification requests
- 0.2s delay for candlestick chunks

If you hit rate limits, increase these values in the respective files.

### Memory Usage

For large numbers of markets with 1-minute data:
- Each market can be 10,000+ rows
- Processing 50 markets = ~500,000 rows in memory
- Recommend 4GB+ RAM for full Tesla analysis

### Data Freshness

- Market data: Update by re-running `download_tesla_data.py`
- Stock prices: Update by deleting `normalized_tesla_stock_1min.csv` and re-running
- Market list: Update by re-running `fetch_stock_markets.py`

---

## Future Enhancements

Potential improvements to consider:

1. **Beta Calculation**: Calculate rolling correlation between aggregate price and normalized Tesla stock
2. **Prediction Accuracy**: Backtest how well aggregate prices predict future Tesla movements
3. **Real-time Updates**: Schedule automatic data refreshes
4. **Other Stocks**: Extend to Apple, Microsoft, NVIDIA, etc.
5. **Machine Learning**: Train models on aggregate prices to predict stock movements
6. **Custom Weighting**: Experiment with different weighting schemes (recency, accuracy, etc.)

---

## License

This project is for educational and research purposes.

**Important:**
- Kalshi data is subject to Kalshi's Terms of Service
- EODHD data is subject to their licensing terms
- This is not financial advice

---

## Credits

**Created by:** Komal
**Project:** Py16B_Project
**Data Sources:**
- Kalshi API (prediction markets)
- EODHD API (stock prices)
- OpenAI API (market merge validation)

---

## Questions?

Common questions:

**Q: What does the aggregate price represent?**
A: It's a volume-weighted average of all bullish Tesla prediction market prices, representing aggregate market sentiment about Tesla's future.

**Q: Can I use this for trading?**
A: This is a research tool. Any trading decisions should be made with proper risk management and professional advice.

**Q: How accurate are prediction markets?**
A: Research shows prediction markets can be quite accurate, but past performance doesn't guarantee future results. Always validate with backtesting.

**Q: Why 7 days for rolling volume?**
A: It balances recency (capturing current sentiment) vs. smoothing (reducing noise). You can adjust this in the code.

**Q: What if a market has no direction classification?**
A: It's marked "neutral" and excluded. You can manually classify it by editing `market_directions_cache.json` or adding keywords to the classifier.

---

**Last Updated:** 2025-11-16
**Version:** 1.0
