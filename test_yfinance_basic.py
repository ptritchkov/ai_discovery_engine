#!/usr/bin/env python3
"""
Basic test to isolate YFinance issues.
"""

import yfinance as yf
import asyncio
from datetime import datetime

def test_basic_yfinance():
    """Test basic YFinance functionality."""
    print("Testing basic YFinance functionality...")
    
    test_symbols = ['AAPL', 'MSFT']
    
    for symbol in test_symbols:
        print(f"\n--- Testing {symbol} ---")
        
        try:
            # Test 1: Create ticker object
            ticker = yf.Ticker(symbol)
            print(f"✅ Created ticker object for {symbol}")
            
            # Test 2: Get basic info (this often fails first)
            try:
                info = ticker.info
                print(f"✅ Got info for {symbol}: {info.get('shortName', 'Unknown')}")
            except Exception as e:
                print(f"❌ Info failed for {symbol}: {str(e)}")
            
            # Test 3: Get historical data with different periods
            for period in ['1d', '5d']:
                try:
                    hist = ticker.history(period=period, timeout=30)
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        print(f"✅ Got {period} history for {symbol}: ${current_price:.2f}")
                        break
                    else:
                        print(f"⚠️  Empty history for {symbol} with period {period}")
                except Exception as e:
                    print(f"❌ History failed for {symbol} ({period}): {str(e)}")
            
        except Exception as e:
            print(f"❌ Complete failure for {symbol}: {str(e)}")

def test_alternative_approach():
    """Test alternative data fetching approach."""
    print(f"\n{'='*50}")
    print("Testing alternative approach with requests...")
    
    import requests
    import json
    
    # Try Yahoo Finance API directly
    symbol = 'AAPL'
    try:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if 'chart' in data and data['chart']['result']:
                result = data['chart']['result'][0]
                meta = result['meta']
                current_price = meta.get('regularMarketPrice', 0)
                print(f"✅ Direct API call successful: {symbol} = ${current_price:.2f}")
                return True
        
        print(f"❌ Direct API call failed")
        return False
        
    except Exception as e:
        print(f"❌ Direct API error: {str(e)}")
        return False

if __name__ == "__main__":
    print(f"YFinance Basic Test - {datetime.now()}")
    print("="*50)
    
    test_basic_yfinance()
    success = test_alternative_approach()
    
    if not success:
        print(f"\n{'='*50}")
        print("DIAGNOSIS:")
        print("- YFinance library may be blocked or rate-limited")
        print("- Network connectivity issues")
        print("- Yahoo Finance API changes")
        print("- Consider using alternative data sources")
        print("="*50)
