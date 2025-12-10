"""
Natural Gas Price Data - Yahoo Finance Granularity Testing
Tests what time intervals are available from Yahoo Finance (NG=F futures)

This is for testing/exploration only. Yahoo provides FUTURES prices, not spot prices.
For production, use FRED or EIA for actual Henry Hub spot prices.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os

# Hard-coded project directories
SRC_DIR = r"D:\Users\williamyun\proj\energy_trading\src\fuel\NG"
INPUT_DIR = r"D:\Users\williamyun\proj\energy_trading\input_data\fuel\NG"
PROCESSED_DIR = r"D:\Users\williamyun\proj\energy_trading\data_processed\fuel\NG"

def fetch_ng_yahoo(start_date, end_date, interval='1d'):
    """
    Fetch natural gas prices from Yahoo Finance
    
    Parameters:
    -----------
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    interval : str
        Data interval: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', 
                       '1d', '5d', '1wk', '1mo', '3mo'
        Note: Intraday data (< 1d) only available for last 60 days
    
    Returns:
    --------
    pd.DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
    """
    
    # NG=F is Henry Hub Natural Gas Futures (front month)
    ticker = "NG=F"
    
    print(f"Fetching {ticker} data from {start_date} to {end_date} at {interval} interval...")
    
    ng = yf.Ticker(ticker)
    df = ng.history(start=start_date, end=end_date, interval=interval)
    
    print(f"✓ Retrieved {len(df)} data points")
    
    if len(df) > 0:
        print(f"  Columns: {list(df.columns)}")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"\n  Sample data:")
        print(df.head())
        print(f"\n  Price statistics (Close):")
        print(df['Close'].describe())
    else:
        print("  ⚠ No data returned")
    
    return df

def test_all_granularities():
    """
    Test what granularities are available from Yahoo Finance
    """
    
    print("="*80)
    print("YAHOO FINANCE NATURAL GAS DATA - GRANULARITY TESTING")
    print("Symbol: NG=F (Henry Hub Natural Gas Futures - Front Month)")
    print("="*80)
    
    end_date = datetime.now()
    results = {}
    
    # Test 1: Daily data for long historical period
    print("\n\n" + "="*80)
    print("TEST 1: DAILY DATA (Historical - Last 2 years)")
    print("="*80)
    start_date = end_date - timedelta(days=730)
    try:
        df_daily = fetch_ng_yahoo(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            interval='1d'
        )
        results['daily'] = {'success': True, 'records': len(df_daily)}
    except Exception as e:
        print(f"✗ Error: {e}")
        results['daily'] = {'success': False, 'error': str(e)}
    
    # Test 2: Hourly data (only last 60 days available)
    print("\n\n" + "="*80)
    print("TEST 2: HOURLY DATA (Last 60 days - maximum for intraday)")
    print("="*80)
    start_date = end_date - timedelta(days=60)
    try:
        df_hourly = fetch_ng_yahoo(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            interval='1h'
        )
        results['hourly'] = {'success': True, 'records': len(df_hourly)}
    except Exception as e:
        print(f"✗ Error: {e}")
        results['hourly'] = {'success': False, 'error': str(e)}
    
    # Test 3: 30-minute data
    print("\n\n" + "="*80)
    print("TEST 3: 30-MINUTE DATA (Last 60 days)")
    print("="*80)
    start_date = end_date - timedelta(days=60)
    try:
        df_30min = fetch_ng_yahoo(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            interval='30m'
        )
        results['30min'] = {'success': True, 'records': len(df_30min)}
    except Exception as e:
        print(f"✗ Error: {e}")
        results['30min'] = {'success': False, 'error': str(e)}
    
    # Test 4: 5-minute data (only last 60 days)
    print("\n\n" + "="*80)
    print("TEST 4: 5-MINUTE DATA (Last 7 days - higher frequency)")
    print("="*80)
    start_date = end_date - timedelta(days=7)
    try:
        df_5min = fetch_ng_yahoo(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            interval='5m'
        )
        results['5min'] = {'success': True, 'records': len(df_5min)}
    except Exception as e:
        print(f"✗ Error: {e}")
        results['5min'] = {'success': False, 'error': str(e)}
    
    # Test 5: Weekly data
    print("\n\n" + "="*80)
    print("TEST 5: WEEKLY DATA (Last 5 years)")
    print("="*80)
    start_date = end_date - timedelta(days=1825)
    try:
        df_weekly = fetch_ng_yahoo(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            interval='1wk'
        )
        results['weekly'] = {'success': True, 'records': len(df_weekly)}
    except Exception as e:
        print(f"✗ Error: {e}")
        results['weekly'] = {'success': False, 'error': str(e)}
    
    # Print summary
    print("\n\n" + "="*80)
    print("SUMMARY: YAHOO FINANCE DATA AVAILABILITY")
    print("="*80)
    
    print("\n✓ AVAILABLE GRANULARITIES:")
    for interval, result in results.items():
        if result['success']:
            print(f"  • {interval.upper():12s}: {result['records']:,} data points")
    
    print("\n✗ LIMITATIONS:")
    print("  • Intraday data (hourly, 30-min, 5-min): Only last 60 days")
    print("  • These are FUTURES prices, not spot prices")
    print("  • Front month contract rolls over monthly (discontinuities)")
    print("  • No minute-by-minute historical data available")
    
    print("\n" + "="*80)
    print("RECOMMENDATION FOR ERCOT PROJECT")
    print("="*80)
    print("❌ Yahoo Finance is NOT ideal for ERCOT price modeling because:")
    print("   - Futures prices ≠ spot prices Texas plants actually pay")
    print("   - Contract rollover creates artificial price jumps")
    print("   - Limited historical intraday data")
    print("\n✅ Better alternatives:")
    print("   - EIA API: Henry Hub spot prices (daily, 1997-present)")
    print("   - FRED: Same data, no API key needed")
    print("   - These match actual gas prices ERCOT plants use")
    
    return results

def save_sample_data():
    """
    Save sample data from different granularities for inspection
    """
    
    print("\n\n" + "="*80)
    print("SAVING SAMPLE DATA TO PROCESSED DIRECTORY")
    print("="*80)
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    end_date = datetime.now()
    
    # Save daily data
    print("\nSaving daily data (last 2 years)...")
    start_date = end_date - timedelta(days=730)
    df_daily = fetch_ng_yahoo(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        interval='1d'
    )
    if len(df_daily) > 0:
        path = os.path.join(PROCESSED_DIR, "ng_yahoo_futures_daily_sample.csv")
        df_daily.to_csv(path)
        print(f"✓ Saved to: {path}")
    
    # Save hourly data
    print("\nSaving hourly data (last 60 days)...")
    start_date = end_date - timedelta(days=60)
    df_hourly = fetch_ng_yahoo(
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d'),
        interval='1h'
    )
    if len(df_hourly) > 0:
        path = os.path.join(PROCESSED_DIR, "ng_yahoo_futures_hourly_sample.csv")
        df_hourly.to_csv(path)
        print(f"✓ Saved to: {path}")
    
    print(f"\n✓ Sample files save  d to: {PROCESSED_DIR}")

if __name__ == "__main__":
    # Run all tests
    results = test_all_granularities()
    
    # Save sample data
    save_sample_data()
    
    print("\n\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review sample data files in processed directory")
    print("2. Get free EIA API key for better spot price data")
    print("3. Use fetch_ng_production.py for actual data collection")