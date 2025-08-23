#!/usr/bin/env python3
"""
Simple test script to verify ticker fetching works without running the full analysis.
This bypasses LLM calls and ML analysis to quickly test market data collection.
"""

import asyncio
import logging
from dotenv import load_dotenv
from src.data_collectors.market_data_collector import MarketDataCollector
from src.utils.logger import setup_logger

# Load environment variables
load_dotenv()

async def test_ticker_fetch():
    """Test fetching data for a few sample tickers."""
    logger = setup_logger(__name__)
    logger.info("Starting ticker fetch test...")
    
    # Initialize market data collector
    collector = MarketDataCollector()
    
    # Test with a few popular stocks
    test_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
    
    print(f"\n{'='*60}")
    print("TICKER FETCH TEST")
    print(f"{'='*60}")
    print(f"Testing {len(test_stocks)} stocks: {', '.join(test_stocks)}")
    print(f"{'='*60}\n")
    
    # Collect data for test stocks
    results = await collector.collect_stock_data(test_stocks, "daily")
    
    # Display results
    for stock, data in results.items():
        print(f"📊 {stock}:")
        
        if 'error' in data:
            print(f"   ❌ Error: {data.get('error', 'Unknown error')}")
            continue
            
        price_data = data.get('price_data', {})
        current_price = price_data.get('current_price', 0)
        price_change_pct = price_data.get('price_change_pct', 0)
        volume = price_data.get('volume', 0)
        data_source = price_data.get('data_source', 'unknown')
        
        if current_price > 0:
            print(f"   ✅ Price: ${current_price:.2f}")
            print(f"   📈 Change: {price_change_pct:+.2f}%")
            print(f"   📊 Volume: {volume:,}")
            print(f"   🔗 Source: {data_source}")
        else:
            print(f"   ⚠️  No valid price data retrieved")
        
        print()
    
    # Summary
    successful_fetches = sum(1 for data in results.values() 
                           if data.get('price_data', {}).get('current_price', 0) > 0)
    
    print(f"{'='*60}")
    print(f"SUMMARY: {successful_fetches}/{len(test_stocks)} stocks successfully fetched")
    print(f"{'='*60}")
    
    if successful_fetches == len(test_stocks):
        print("🎉 All ticker fetches successful!")
    elif successful_fetches > 0:
        print("⚠️  Some ticker fetches failed - check logs above")
    else:
        print("❌ All ticker fetches failed - check API configuration")
    
    return successful_fetches > 0

if __name__ == "__main__":
    success = asyncio.run(test_ticker_fetch())
    exit(0 if success else 1)
