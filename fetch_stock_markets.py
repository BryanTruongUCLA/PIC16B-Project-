"""
Fetch Stock-Related Markets from Kalshi
Creates CSV file with all currently active stock-related markets
Verifies market status and provides detailed logging
"""

import requests
import pandas as pd
import time
import re
from datetime import datetime, timezone
from typing import List, Dict, Optional

# Try to import BeautifulSoup for web scraping
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

# API Configuration
BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

# Major stock companies to track
MAJOR_COMPANIES = [
    'Apple', 'Microsoft', 'Amazon', 'Google', 'Alphabet', 'Meta', 'Facebook',
    'Netflix', 'Tesla', 'Nvidia', 'Intel', 'AMD', 'Oracle', 'Salesforce',
    'Adobe', 'IBM', 'Cisco', 'JPMorgan', 'Bank of America', 'Wells Fargo',
    'Goldman Sachs', 'Morgan Stanley', 'Citigroup', 'Berkshire Hathaway',
    'Walmart', 'Target', 'Costco', 'Home Depot', 'Nike', 'Starbucks',
    'McDonald\'s', 'Johnson & Johnson', 'Pfizer', 'UnitedHealth', 'Moderna',
    'Disney', 'Warner Bros', 'Paramount', 'Ford', 'GM', 'General Motors',
    'Toyota', 'ExxonMobil', 'Chevron', 'Delta Air Lines', 'United Airlines',
    'American Airlines', 'Southwest Airlines', 'Coinbase', 'MicroStrategy',
    'Lockheed Martin', 'Boeing', 'Raytheon'
]


class StockMarketFetcher:
    """Fetches and validates stock-related markets from Kalshi"""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'KalshiStockAnalysis/1.0'
        })

    def fetch_all_open_markets(self, limit: int = 1000) -> List[Dict]:
        """
        Fetch all open markets from Kalshi API with pagination

        Args:
            limit: Number of results per page

        Returns:
            List of market dictionaries
        """
        all_markets = []
        cursor = None
        page = 0

        print("Fetching all open markets from Kalshi API...")

        while True:
            page += 1
            url = f"{BASE_URL}/markets"
            params = {
                'status': 'open',
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

                # Rate limiting
                time.sleep(0.1)

            except requests.exceptions.RequestException as e:
                print(f"[ERROR] Error fetching markets: {e}")
                break

        print(f"[OK] Total open markets fetched: {len(all_markets)}\n")
        return all_markets

    def check_close_time(self, close_time_str: str) -> bool:
        """
        Check if market's close time has passed

        Args:
            close_time_str: ISO format timestamp string

        Returns:
            True if market is still open (close time hasn't passed)
        """
        if not close_time_str:
            return True  # No close time means indefinite

        try:
            # Parse the close time (format: "2025-12-31T23:59:59Z")
            close_time = datetime.fromisoformat(close_time_str.replace('Z', '+00:00'))
            current_time = datetime.now(timezone.utc)

            return current_time < close_time
        except Exception as e:
            print(f"  [WARN] Could not parse close_time '{close_time_str}': {e}")
            return True  # Assume open if we can't parse

    def scrape_market_page(self, ticker: str) -> Optional[Dict]:
        """
        Scrape Kalshi market page to verify if it's actually open

        Args:
            ticker: Market ticker

        Returns:
            Dict with scraping results or None if scraping failed
        """
        if not HAS_BS4:
            return None

        url = f"https://kalshi.com/markets/{ticker.lower()}"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Look for common indicators that market is closed
            page_text = soup.get_text().lower()

            indicators = {
                'has_closed_indicator': 'market closed' in page_text or 'closed at' in page_text,
                'has_settled_indicator': 'settled' in page_text or 'resolved' in page_text,
                'has_expired_indicator': 'expired' in page_text,
            }

            is_likely_open = not any(indicators.values())

            return {
                'scraped': True,
                'likely_open': is_likely_open,
                'indicators': indicators,
                'url': url
            }

        except Exception as e:
            return {
                'scraped': False,
                'error': str(e),
                'url': url
            }

    def verify_market_status(self, ticker: str, event_ticker: str) -> Dict:
        """
        Comprehensively verify a market's status using multiple methods

        Args:
            ticker: Market ticker
            event_ticker: Event ticker

        Returns:
            Dict with verification results
        """
        url = f"{BASE_URL}/markets/{ticker}"

        try:
            response = self.session.get(url)
            response.raise_for_status()
            market_data = response.json().get('market', {})

            api_status = market_data.get('status', 'unknown')
            close_time = market_data.get('close_time')
            volume = market_data.get('volume', 0)

            # Check 1: API status (can be 'open' or 'active')
            api_says_open = api_status in ['open', 'active']

            # Check 2: Close time hasn't passed
            close_time_valid = self.check_close_time(close_time)

            # Check 3: Has trading volume (any activity ever)
            has_volume = volume > 0

            # Check 4: Scrape website (if available)
            scrape_result = self.scrape_market_page(ticker)
            website_says_open = scrape_result.get('likely_open', True) if scrape_result and scrape_result.get('scraped') else None

            # Determine if market is truly open
            checks_passed = {
                'api_status': api_says_open,
                'close_time_valid': close_time_valid,
                'has_volume': has_volume,
            }

            if website_says_open is not None:
                checks_passed['website_says_open'] = website_says_open

            # Market is verified open only if all checks pass
            is_verified_open = all(checks_passed.values())

            return {
                'ticker': ticker,
                'verified': True,
                'status': api_status,
                'is_truly_open': is_verified_open,
                'checks': checks_passed,
                'open_time': market_data.get('open_time'),
                'close_time': close_time,
                'can_close_early': market_data.get('can_close_early', False),
                'volume': volume,
                'scrape_result': scrape_result
            }

        except requests.exceptions.RequestException as e:
            print(f"  [WARN] Could not verify {ticker}: {e}")
            return {
                'ticker': ticker,
                'verified': False,
                'is_truly_open': False,
                'status': 'error',
                'error': str(e)
            }

    def is_stock_related(self, title: str, subtitle: str = "") -> tuple:
        """
        Determine if a market is stock-related by checking for company names

        Args:
            title: Market title
            subtitle: Market subtitle

        Returns:
            Tuple of (is_stock_related: bool, company_name: str or None)
        """
        combined_text = f"{title} {subtitle}".lower()

        for company in MAJOR_COMPANIES:
            # Use word boundary regex to match company names
            # This prevents false positives like "target" matching "targeted"
            pattern = r'\b' + re.escape(company.lower()) + r'\b'

            if re.search(pattern, combined_text):
                return True, company

        return False, None

    def fetch_stock_markets(self, verify_status: bool = True) -> pd.DataFrame:
        """
        Fetch all stock-related markets and optionally verify their status

        Args:
            verify_status: Whether to verify each market's status individually

        Returns:
            DataFrame with stock-related markets
        """
        # Fetch all open markets
        all_markets = self.fetch_all_open_markets()

        # Filter for stock-related markets
        print("Filtering for stock-related markets...")
        stock_markets = []

        for market in all_markets:
            ticker = market.get('ticker', '')
            title = market.get('title', '')
            subtitle = market.get('subtitle', '')

            is_stock, company = self.is_stock_related(title, subtitle)

            if is_stock:
                stock_markets.append({
                    'ticker': ticker,
                    'event_ticker': market.get('event_ticker', ''),
                    'title': title,
                    'subtitle': subtitle,
                    'company': company,
                    'status': market.get('status', ''),
                    'open_time': market.get('open_time', ''),
                    'close_time': market.get('close_time', ''),
                    'volume': market.get('volume', 0),
                    'volume_usd': market.get('volume', 0) / 100.0,  # Convert cents to dollars
                    'yes_bid': market.get('yes_bid', 0) / 100.0,
                    'yes_ask': market.get('yes_ask', 0) / 100.0,
                    'last_price': market.get('last_price', 0) / 100.0,
                })

        print(f"[OK] Found {len(stock_markets)} stock-related markets\n")

        if not stock_markets:
            print("[WARN] No stock-related markets found!")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(stock_markets)

        # Verify status for each market if requested
        if verify_status:
            print(f"Verifying status of {len(df)} markets using comprehensive checks...")
            print("  - API status field")
            print("  - Close time validation")
            print("  - Volume check")
            if HAS_BS4:
                print("  - Website scraping")
            print("(This may take a few minutes...)\n")

            verified_statuses = []
            failed_verifications = []
            verification_failures_by_type = {
                'api_status': [],
                'close_time_valid': [],
                'has_volume': [],
                'website_says_open': []
            }

            for idx, row in df.iterrows():
                if (idx + 1) % 10 == 0:
                    print(f"  Verified {idx + 1}/{len(df)} markets...")

                verification = self.verify_market_status(row['ticker'], row['event_ticker'])
                verified_statuses.append(verification)

                # Check if market truly passed all verifications
                is_truly_open = verification.get('is_truly_open', False)

                if not is_truly_open:
                    failed_verifications.append({
                        'ticker': row['ticker'],
                        'title': row['title'],
                        'company': row['company'],
                        'api_status': verification.get('status', 'unknown'),
                        'checks': verification.get('checks', {}),
                        'close_time': verification.get('close_time', 'N/A')
                    })

                    # Track which specific checks failed
                    checks = verification.get('checks', {})
                    for check_name, passed in checks.items():
                        if not passed:
                            verification_failures_by_type[check_name].append(row['ticker'])

                # Rate limiting for both API and web scraping
                time.sleep(0.2)

            print(f"\n[OK] Verification complete!")
            truly_open_count = len([v for v in verified_statuses if v.get('is_truly_open', False)])
            print(f"  Markets passing all checks: {truly_open_count}/{len(df)}")
            print(f"  Markets failing verification: {len(failed_verifications)}")

            if failed_verifications:
                print("\n" + "="*70)
                print("VERIFICATION FAILURES BY CHECK TYPE")
                print("="*70)

                for check_type, failed_tickers in verification_failures_by_type.items():
                    if failed_tickers:
                        print(f"\n{check_type.upper()} - {len(failed_tickers)} failures:")
                        for ticker in failed_tickers[:5]:  # Show first 5
                            print(f"  - {ticker}")
                        if len(failed_tickers) > 5:
                            print(f"  ... and {len(failed_tickers) - 5} more")

                print("\n" + "="*70)
                print("DETAILED FAILURE LIST")
                print("="*70)
                for issue in failed_verifications[:10]:  # Show first 10
                    print(f"\n  {issue['ticker']} ({issue['company']})")
                    print(f"    Title: {issue['title'][:60]}...")
                    print(f"    API Status: {issue['api_status']}")
                    print(f"    Close Time: {issue['close_time']}")
                    print(f"    Failed Checks:")
                    for check, passed in issue['checks'].items():
                        status = "[PASS]" if passed else "[FAIL]"
                        print(f"      {status} {check}")

                if len(failed_verifications) > 10:
                    print(f"\n  ... and {len(failed_verifications) - 10} more failures")

                # Filter out markets that didn't pass comprehensive verification
                verified_tickers = [v['ticker'] for v in verified_statuses if v.get('is_truly_open', False)]
                df = df[df['ticker'].isin(verified_tickers)]
                print(f"\n[OK] Filtered to {len(df)} truly open markets (passing all checks)")

        return df


def main():
    """Main execution function"""
    print("="*70)
    print("KALSHI STOCK-RELATED MARKETS FETCHER")
    print("="*70)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Initialize fetcher
    fetcher = StockMarketFetcher()

    # Fetch and verify stock markets
    stock_df = fetcher.fetch_stock_markets(verify_status=True)

    if stock_df.empty:
        print("\n[ERROR] No stock markets found or all failed verification")
        return

    # Display summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    print(f"Total stock-related markets: {len(stock_df)}")
    print(f"Total trading volume: ${stock_df['volume_usd'].sum():,.2f}")
    print(f"Average volume per market: ${stock_df['volume_usd'].mean():,.2f}")
    print(f"Median volume per market: ${stock_df['volume_usd'].median():,.2f}")
    print()

    # Company breakdown
    print("Markets by company:")
    company_counts = stock_df['company'].value_counts()
    for company, count in company_counts.head(10).items():
        company_volume = stock_df[stock_df['company'] == company]['volume_usd'].sum()
        print(f"  {company}: {count} markets, ${company_volume:,.2f} volume")

    if len(company_counts) > 10:
        print(f"  ... and {len(company_counts) - 10} more companies")
    print()

    # Export to CSV
    output_file = 'all_stock_related_markets.csv'
    stock_df.to_csv(output_file, index=False)
    print(f"[OK] Exported to: {output_file}")
    print(f"     {len(stock_df)} markets saved")
    print()

    # Show top 5 markets by volume
    print("Top 5 markets by trading volume:")
    top_markets = stock_df.nlargest(5, 'volume_usd')
    for idx, row in top_markets.iterrows():
        print(f"  {row['ticker']}: ${row['volume_usd']:,.2f}")
        print(f"    {row['title'][:70]}...")
        print()

    print("="*70)
    print("FETCH COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
