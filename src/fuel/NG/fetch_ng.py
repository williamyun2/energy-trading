"""
Natural Gas Price Data Fetcher
Production script for ERCOT energy trading project

Fetches Henry Hub natural gas spot prices - the key driver for ERCOT DAM prices
Uses FRED as default (no API key), with EIA as recommended alternative
"""

import pandas as pd
import requests
from datetime import datetime, timedelta
import os
import sys

# Hard-coded project directories
SRC_DIR = r"D:\Users\williamyun\proj\power_trading\src\fuel\NG"
INPUT_DIR = r"D:\Users\williamyun\proj\power_trading\input_data\fuel\NG"
PROCESSED_DIR = r"D:\Users\williamyun\proj\power_trading\data_processed\fuel\NG"

class NGPriceFetcher:
    """
    Natural Gas Price Data Fetcher
    
    Supports multiple sources:
    - FRED: Free, no API key, daily data (default)
    - EIA: Free API key required, daily data (recommended for accuracy)
    - Yahoo Finance: Free, no API key, daily + hourly for last 60 days
    """
    
    def __init__(self, eia_api_key=None):
        """
        Initialize fetcher
        
        Parameters:
        -----------
        eia_api_key : str, optional
            EIA API key (get free at https://www.eia.gov/opendata/register.php)
        """
        self.eia_api_key = eia_api_key
        
        # Ensure directories exist
        os.makedirs(INPUT_DIR, exist_ok=True)
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        os.makedirs(SRC_DIR, exist_ok=True)
    
    def fetch_fred(self, start_date='2017-01-01', end_date=None):
        """
        Fetch from FRED (Federal Reserve Economic Data)
        
        Pros: No API key, reliable, clean data
        Cons: Daily only, may have slight lag vs EIA
        
        Returns: DataFrame with columns [Date, Price]
        """
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        series_id = "DHHNGSP"  # Henry Hub Natural Gas Spot Price
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        
        print(f"Fetching FRED data: {start_date} to {end_date}")
        
        try:
            df = pd.read_csv(url)
            df.columns = ['Date', 'Price']
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Filter by date range
            df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
            df = df.dropna().reset_index(drop=True)
            
            print(f"✓ Retrieved {len(df)} records")
            print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
            print(f"  Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"✗ Error fetching FRED data: {e}")
            return None
    
    def fetch_eia(self, start_date='2017-01-01', end_date=None):
        """
        Fetch from EIA (Energy Information Administration)
        
        Pros: Official source, most accurate spot prices, comprehensive
        Cons: Requires free API key
        
        Returns: DataFrame with columns [Date, Price]
        """
        
        if self.eia_api_key is None:
            print("✗ EIA API key required. Get free key at: https://www.eia.gov/opendata/register.php")
            return None
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        url = "https://api.eia.gov/v2/natural-gas/pri/spt/data/"
        
        params = {
            'api_key': self.eia_api_key,
            'frequency': 'daily',
            'data[0]': 'value',
            'facets[series][]': 'RNGWHHD',
            'start': start_date,
            'end': end_date,
            'sort[0][column]': 'period',
            'sort[0][direction]': 'asc',
            'offset': 0,
            'length': 5000
        }
        
        print(f"Fetching EIA data: {start_date} to {end_date}")
        
        try:
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'response' in data and 'data' in data['response']:
                    records = data['response']['data']
                    
                    df = pd.DataFrame(records)
                    df['Date'] = pd.to_datetime(df['period'])
                    df = df.rename(columns={'value': 'Price'})
                    df = df[['Date', 'Price']].sort_values('Date').reset_index(drop=True)
                    
                    print(f"✓ Retrieved {len(df)} records")
                    print(f"  Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
                    print(f"  Price range: ${df['Price'].min():.2f} to ${df['Price'].max():.2f}")
                    
                    return df
                else:
                    print(f"✗ Unexpected API response structure")
                    return None
            else:
                print(f"✗ API error {response.status_code}: {response.text}")
                return None
                
        except Exception as e:
            print(f"✗ Error fetching EIA data: {e}")
            return None
    
    def fetch_yahoo(self, start_date='2017-01-01', end_date=None, interval='1d'):
        """
        Fetch from Yahoo Finance
        
        Pros: Easy, supports hourly data for last 60 days
        Cons: FUTURES prices (not spot), contract rollover discontinuities
        
        Parameters:
        -----------
        interval : str
            '1d' for daily, '1h' for hourly (only last 60 days)
        
        Returns: DataFrame with OHLCV columns
        """
        
        try:
            import yfinance as yf
        except ImportError:
            print("✗ yfinance not installed. Run: pip install yfinance")
            return None
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Check if hourly data requested but date range too old
        if interval in ['1h', '60m'] and start_date < (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d'):
            print(f"⚠ Hourly data only available for last 60 days")
            start_date = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
        
        print(f"Fetching Yahoo Finance (NG=F futures): {start_date} to {end_date} ({interval})")
        
        try:
            ng = yf.Ticker("NG=F")
            df = ng.history(start=start_date, end=end_date, interval=interval)
            
            print(f"✓ Retrieved {len(df)} records")
            if len(df) > 0:
                print(f"  Date range: {df.index.min()} to {df.index.max()}")
                print(f"  Close price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
            
            return df
            
        except Exception as e:
            print(f"✗ Error fetching Yahoo data: {e}")
            return None
    
    def fetch_and_save(self, source='fred', start_date='2017-01-01', end_date=None):
        """
        Fetch data and save to processed directory
        
        Parameters:
        -----------
        source : str
            'fred', 'eia', or 'yahoo'
        start_date : str
            Start date 'YYYY-MM-DD'
        end_date : str, optional
            End date 'YYYY-MM-DD' (None = today)
        
        Returns:
        --------
        DataFrame with fetched data
        """
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Fetch data
        if source.lower() == 'fred':
            df = self.fetch_fred(start_date, end_date)
            filename = f"ng_henry_hub_spot_fred_{start_date}_{end_date}.csv"
        elif source.lower() == 'eia':
            df = self.fetch_eia(start_date, end_date)
            filename = f"ng_henry_hub_spot_eia_{start_date}_{end_date}.csv"
        elif source.lower() == 'yahoo':
            df = self.fetch_yahoo(start_date, end_date, interval='1d')
            filename = f"ng_futures_yahoo_daily_{start_date}_{end_date}.csv"
        else:
            print(f"✗ Unknown source: {source}. Use 'fred', 'eia', or 'yahoo'")
            return None
        
        # Save if successful
        if df is not None and len(df) > 0:
            output_path = os.path.join(PROCESSED_DIR, filename)
            df.to_csv(output_path, index=(source=='yahoo'))  # Yahoo has datetime index
            print(f"\n✓ Saved to: {output_path}")
            print(f"  File size: {os.path.getsize(output_path) / 1024:.1f} KB")
            
            # Print sample
            print(f"\nSample data:")
            print(df.head())
            
            return df
        else:
            print(f"✗ No data fetched")
            return None

def main():
    """
    Main execution function
    """
    
    print("="*80)
    print("NATURAL GAS PRICE DATA FETCHER")
    print("For ERCOT Energy Trading Project")
    print("="*80)
    
    # Initialize fetcher
    # To use EIA: fetcher = NGPriceFetcher(eia_api_key='YOUR_KEY_HERE')
    fetcher = NGPriceFetcher()
    
    # Fetch historical data from FRED (default, no API key needed)
    print("\n--- Fetching from FRED (recommended for quick start) ---")
    df_fred = fetcher.fetch_and_save(
        source='fred',
        start_date='2010-12-01'
    )
    
    # Optionally also fetch from Yahoo Finance for comparison
    print("\n\n--- Fetching from Yahoo Finance (futures, for comparison) ---")
    df_yahoo = fetcher.fetch_and_save(
        source='yahoo',
        start_date='2010-12-01'
    )
    
    # If you have EIA API key, uncomment below:
    # print("\n\n--- Fetching from EIA (most accurate) ---")
    # fetcher_eia = NGPriceFetcher(eia_api_key='YOUR_KEY_HERE')
    # df_eia = fetcher_eia.fetch_and_save(
    #     source='eia',
    #     start_date='2017-01-01'
    # )
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print(f"\nData saved to: {PROCESSED_DIR}")
    print("\nRECOMMENDATION:")
    print("- Use FRED data for now (already fetched)")
    print("- Get free EIA API key for most accurate spot prices")
    print("  Register at: https://www.eia.gov/opendata/register.php")
    print("\nWhy Henry Hub matters for ERCOT:")
    print("- 50-70% of ERCOT electricity prices set by natural gas units")
    print("- Strong correlation between NG spot price and DAM prices")
    print("- Critical feature for your price forecasting model")

if __name__ == "__main__":
    main()