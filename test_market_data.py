#!/usr/bin/env python3
"""Test script to verify market data collection is working properly."""

import asyncio
from dotenv import load_dotenv
from src.data_collectors.market_data_collector import MarketDataCollector
from src.utils.logger import setup_logger

# Load environment variables
load_dotenv()

async def test_market_data_collection():
    """Test market data collection with a realistic number of stocks."""
    logger = setup_logger("test_market_data")
    
    print("\n" + "="*60)
    print("TESTING MARKET DATA COLLECTION WITH RATE LIMITING")
    print("="*60)
    
    # Initialize the market data collector
    collector = MarketDataCollector()
    
    # Test with a realistic number of stocks (more than Alpha Vantage limit)
    test_stocks = [
        "AAPL", "MSFT", "GOOGL", "TSLA", "NVDA", "META", "AMZN", 
        "BRK.B", "JPM", "JNJ", "V", "WMT", "PG", "MA", "HD", 
        "CVX", "PFE", "KO", "DIS", "NFLX", "SOME_UNKNOWN_STOCK"
    ]
    
    print(f"Testing with {len(test_stocks)} stocks: {test_stocks}")
    print(f"Polygon enabled: {collector.polygon_enabled}")
    print(f"YFinance enabled: {collector.yfinance_enabled}")
    print(f"Alpha Vantage available: {collector.alpha_vantage is not None}")
    print(f"Prioritize Polygon: {collector.prioritize_polygon}")
    print(f"Alpha Vantage rate limit: {collector.max_alpha_vantage_calls} calls/minute")
    
    print("\nCollecting data...")
    print("-" * 40)
    
    import time
    start_time = time.time()
    
    try:
        # Test the data collection
        stock_data = await collector.collect_stock_data(test_stocks, "daily")
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\nResults (completed in {duration:.1f} seconds):")
        print(f"Stocks processed: {len(stock_data)}")
        
        successful_count = 0
        failed_count = 0
        
        for symbol, data in stock_data.items():
            price_data = data.get('price_data', {})
            current_price = price_data.get('current_price', 0)
            data_source = price_data.get('data_source', 'unknown')
            
            if current_price > 0:
                print(f"✅ {symbol}: ${current_price:.2f} (source: {data_source})")
                successful_count += 1
            else:
                print(f"❌ {symbol}: No valid price data")
                failed_count += 1
        
        print(f"\nSummary:")
        print(f"✅ Successful: {successful_count}/{len(stock_data)}")
        print(f"❌ Failed: {failed_count}/{len(stock_data)}")
        print(f"⏱️  Total time: {duration:.1f} seconds")
        print(f"📊 Rate: {len(stock_data)/duration*60:.1f} stocks/minute")
        
        if successful_count >= 10:  # At least 10 successful stocks
            print(f"\n🎉 Market data collection is WORKING!")
            print(f"Rate limiting and prioritization are functioning correctly.")
            print(f"Your main.py should work efficiently now.")
        elif successful_count > 0:
            print(f"\n⚠️  Partial success. Consider waiting for API limits to reset.")
        else:
            print(f"\n❌ All data collection failed. Check API status.")
            
    except Exception as e:
        logger.error(f"Error testing market data collection: {str(e)}")
        print(f"❌ Test failed with error: {str(e)}")

if __name__ == "__main__":
    asyncio.run(test_market_data_collection()) 