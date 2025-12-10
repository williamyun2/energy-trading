"""
Natural Gas Price Data Sources - Comprehensive Comparison Guide
Compares free data sources for Henry Hub natural gas prices

This guide helps you choose the best data source for ERCOT price modeling.
Includes code examples for FRED, EIA, and Yahoo Finance.
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import os

# Hard-coded project directories
SRC_DIR = r"D:\Users\williamyun\proj\energy_trading\src\fuel\NG"
INPUT_DIR = r"D:\Users\williamyun\proj\energy_trading\input_data\fuel\NG"
PROCESSED_DIR = r"D:\Users\williamyun\proj\energy_trading\data_processed\fuel\NG"

"""
===============================================================================
COMPREHENSIVE COMPARISON OF FREE NATURAL GAS DATA SOURCES
===============================================================================

WHY NATURAL GAS PRICES MATTER FOR ERCOT:
- 50-70% of ERCOT electricity prices are set by natural gas units (marginal pricing)
- Texas gas plants use SPOT prices for fuel costs, not long-term contracts
- Henry Hub is the benchmark pricing point for US natural gas
- Strong correlation between Henry Hub daily spot price and ERCOT DAM prices
- This is THE most important external feature for your model


===============================================================================
SOURCE 1: EIA API (Energy Information Administration)
===============================================================================

🏆 RECOMMENDED FOR PRODUCTION

✓ PROS:
  • Official US government data - highest reliability
  • Henry Hub SPOT prices (actual physical market, not futures)
  • Daily data from 1997 to present
  • Multiple series: spot, futures, wellhead, citygate prices
  • Free API key (just register, no credit card)
  • Well-documented API
  • Covers production, consumption, storage data too

✗ CONS:
  • Daily granularity only (no hourly/intraday)
  • Requires API key registration (takes 2 minutes)
  • Data has 1-2 day publication lag
  • API rate limits (but generous for free tier)

📊 DATA CHARACTERISTICS:
  • Frequency: Daily
  • History: 1997-01-07 to present
  • Update time: Usually by 3 PM ET next business day
  • Price type: Spot (physical market settlements)
  • Unit: $/MMBtu (Dollars per Million British Thermal Units)

🔑 KEY SERIES IDS:
  • NG.RNGWHHD.D  - Henry Hub Natural Gas Spot Price (Daily) ← PRIMARY
  • NG.RNGC1.D    - Natural Gas Futures Contract 1 (Daily)
  • NG.N9190US3.M - Natural Gas Wellhead Price (Monthly)

📍 API ENDPOINT:
  https://api.eia.gov/v2/natural-gas/pri/spt/data/

🔗 GET FREE API KEY:
  https://www.eia.gov/opendata/register.php

⭐ RATING FOR ERCOT PROJECT: 10/10
   This is what you should use for production.


===============================================================================
SOURCE 2: FRED (Federal Reserve Economic Data)
===============================================================================

🥈 BEST FOR QUICK START (NO API KEY NEEDED)

✓ PROS:
  • No API key required
  • Simple CSV download
  • Reliable, clean data
  • Same source as EIA (they pull from EIA)
  • Easy to integrate

✗ CONS:
  • Daily only (no intraday)
  • Less flexibility than EIA API
  • Only has Henry Hub spot, not other series
  • May have slightly longer lag than EIA

📊 DATA CHARACTERISTICS:
  • Frequency: Daily (business days)
  • History: 1997-01-07 to present
  • Price type: Spot (same as EIA)
  • Unit: $/MMBtu

🔑 SERIES ID:
  • DHHNGSP - Henry Hub Natural Gas Spot Price

📍 DIRECT CSV URL:
  https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP

⭐ RATING FOR ERCOT PROJECT: 9/10
   Perfect for quick prototyping, good enough for production.


===============================================================================
SOURCE 3: YAHOO FINANCE
===============================================================================

⚠️ USE WITH CAUTION

✓ PROS:
  • No API key needed
  • Easy to use (yfinance library)
  • Hourly data available (last 60 days)
  • 5-minute data available (last 60 days)
  • Daily data going back many years

✗ CONS:
  • FUTURES prices, NOT spot prices
  • Front month contract rolls over monthly → price discontinuities
  • Limited historical intraday data (only 60 days)
  • Less reliable than government sources
  • Futures ≠ what Texas gas plants actually pay

📊 DATA CHARACTERISTICS:
  • Frequency: Daily, hourly (60d), 5-min (60d)
  • History: Many years for daily
  • Price type: FUTURES (not spot!)
  • Unit: $/MMBtu
  • Symbol: NG=F (front month futures contract)

⚠️ CRITICAL ISSUE - CONTRACT ROLLOVER:
  Yahoo provides front month futures, which rolls to next month before expiry.
  This creates artificial price jumps that don't reflect actual market moves.
  
  Example: On Nov 25, contract rolls from Dec to Jan futures
  - Nov 24: $3.45 (Dec contract)
  - Nov 25: $3.15 (Jan contract) ← NOT a real $0.30 price drop!

⭐ RATING FOR ERCOT PROJECT: 5/10
   OK for testing/prototyping, NOT recommended for production.
   Futures prices don't match what ERCOT gas plants pay.


===============================================================================
SOURCE 4: ALPHA VANTAGE
===============================================================================

✗ NOT RECOMMENDED

✗ CONS:
  • Only 25 API calls per day (free tier)
  • Natural gas coverage is poor
  • Primarily focused on stocks/forex
  • Limited historical data

⭐ RATING FOR ERCOT PROJECT: 2/10
   Skip this one.


===============================================================================
OTHER CONSIDERATIONS
===============================================================================

WHY SPOT PRICES > FUTURES PRICES FOR ERCOT:

Most Texas gas plants are merchant generators that:
1. Buy gas on the spot market daily/weekly
2. Don't hedge with long-term contracts (unlike regulated utilities)
3. Base their bid prices on TODAY'S spot gas price + heat rate

Therefore: Henry Hub SPOT price >> Futures price for modeling

CORRELATION WITH ERCOT DAM:
- Henry Hub spot vs ERCOT DAM: ~0.7-0.8 correlation
- Even higher during summer (gas plant heavy dispatch)
- Critical for predicting price spikes


===============================================================================
FINAL RECOMMENDATION FOR YOUR ERCOT PROJECT
===============================================================================

🎯 PRIMARY SOURCE: EIA API
   - Most accurate spot prices
   - Same data Texas plants use for fuel cost calculations
   - Professional-grade data quality
   - Get free API key: https://www.eia.gov/opendata/register.php

🎯 BACKUP/QUICK START: FRED
   - No setup required
   - Good enough for initial development
   - Switch to EIA when ready for production

❌ AVOID: Yahoo Finance futures
   - Only use for comparison/validation
   - Don't use futures prices in your model


===============================================================================
CODE EXAMPLES BELOW
===============================================================================
"""

def fetch_eia_demo(api_key):
    """
    Demonstration of fetching from EIA API
    
    Get your free API key at: https://www.eia.gov/opendata/register.php
    """
    
    url = "https://api.eia.gov/v2/natural-gas/pri/spt/data/"
    
    params = {
        'api_key': api_key,
        'frequency': 'daily',
        'data[0]': 'value',
        'facets[series][]': 'RNGWHHD',
        'start': '2023-01-01',
        'end': '2023-12-31',
        'sort[0][column]': 'period',
        'sort[0][direction]': 'asc',
        'offset': 0,
        'length': 5000
    }
    
    print("Fetching EIA data...")
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        records = data['response']['data']
        
        df = pd.DataFrame(records)
        df['Date'] = pd.to_datetime(df['period'])
        df = df.rename(columns={'value': 'Price'})
        df = df[['Date', 'Price']].sort_values('Date')
        
        print(f"✓ Retrieved {len(df)} records")
        print(df.head())
        return df
    else:
        print(f"✗ Error: {response.status_code}")
        return None

def fetch_fred_demo():
    """
    Demonstration of fetching from FRED (no API key needed)
    """
    
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DHHNGSP"
    
    print("Fetching FRED data...")
    df = pd.read_csv(url)
    df.columns = ['Date', 'Price']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.dropna()
    
    # Get last year
    one_year_ago = datetime.now() - timedelta(days=365)
    df = df[df['Date'] >= one_year_ago]
    
    print(f"✓ Retrieved {len(df)} records")
    print(df.head())
    return df

def fetch_yahoo_demo():
    """
    Demonstration of fetching from Yahoo Finance
    
    WARNING: This is FUTURES data, not spot prices!
    """
    
    try:
        import yfinance as yf
        
        print("Fetching Yahoo Finance data (FUTURES - not spot)...")
        ng = yf.Ticker("NG=F")
        df = ng.history(start='2023-01-01', end='2023-12-31', interval='1d')
        
        print(f"✓ Retrieved {len(df)} records")
        print("⚠ WARNING: These are FUTURES prices, not spot prices!")
        print(df.head())
        return df
        
    except ImportError:
        print("✗ yfinance not installed. Run: pip install yfinance")
        return None

def compare_all_sources(eia_api_key=None):
    """
    Fetch and compare data from all sources
    """
    
    print("="*80)
    print("COMPARING ALL DATA SOURCES")
    print("="*80)
    
    # FRED (always available)
    print("\n--- FRED (No API Key) ---")
    df_fred = fetch_fred_demo()
    
    # EIA (if API key provided)
    if eia_api_key:
        print("\n--- EIA (Official Source) ---")
        df_eia = fetch_eia_demo(eia_api_key)
    else:
        print("\n--- EIA (Skipped - No API Key) ---")
        print("Get free key at: https://www.eia.gov/opendata/register.php")
        df_eia = None
    
    # Yahoo Finance
    print("\n--- Yahoo Finance (Futures) ---")
    df_yahoo = fetch_yahoo_demo()
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    
    if df_fred is not None:
        print(f"\n✓ FRED:")
        print(f"  Records: {len(df_fred)}")
        print(f"  Mean price: ${df_fred['Price'].mean():.2f}/MMBtu")
        print(f"  Type: SPOT PRICES ✓")
    
    if df_eia is not None:
        print(f"\n✓ EIA:")
        print(f"  Records: {len(df_eia)}")
        print(f"  Mean price: ${df_eia['Price'].mean():.2f}/MMBtu")
        print(f"  Type: SPOT PRICES ✓")
    
    if df_yahoo is not None:
        print(f"\n⚠ Yahoo Finance:")
        print(f"  Records: {len(df_yahoo)}")
        print(f"  Mean close: ${df_yahoo['Close'].mean():.2f}/MMBtu")
        print(f"  Type: FUTURES PRICES (not spot) ✗")
    
    return df_fred, df_eia, df_yahoo

def save_comparison_data():
    """
    Save sample data from different sources for visual comparison
    """
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("\n" + "="*80)
    print("SAVING COMPARISON DATA")
    print("="*80)
    
    # FRED
    print("\nFetching FRED data...")
    df_fred = fetch_fred_demo()
    if df_fred is not None:
        path = os.path.join(PROCESSED_DIR, "comparison_fred_spot.csv")
        df_fred.to_csv(path, index=False)
        print(f"✓ Saved FRED data: {path}")
    
    # Yahoo
    print("\nFetching Yahoo data...")
    df_yahoo = fetch_yahoo_demo()
    if df_yahoo is not None:
        path = os.path.join(PROCESSED_DIR, "comparison_yahoo_futures.csv")
        df_yahoo.to_csv(path)
        print(f"✓ Saved Yahoo data: {path}")
    
    print(f"\n✓ Comparison files saved to: {PROCESSED_DIR}")
    print("\nYou can now plot these side-by-side to see spot vs futures differences")

if __name__ == "__main__":
    # Print the comprehensive guide
    print(__doc__)
    
    # Save comparison data
    save_comparison_data()
    
    # If you have EIA API key, uncomment and add it here:
    # df_fred, df_eia, df_yahoo = compare_all_sources(eia_api_key='YOUR_KEY_HERE')
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Register for free EIA API key (2 minutes)")
    print("   → https://www.eia.gov/opendata/register.php")
    print("\n2. Use fetch_ng_production.py for actual data collection")
    print("\n3. For your ERCOT model, use:")
    print("   • EIA Henry Hub spot prices (best)")
    print("   • FRED Henry Hub spot prices (good enough)")
    print("   • NOT Yahoo futures prices")