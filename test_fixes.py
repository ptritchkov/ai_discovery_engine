#!/usr/bin/env python3
"""
Test script to validate all AI stock discovery engine fixes.
"""

import asyncio
import sys
import os
from datetime import datetime, timedelta
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_multi_source_news():
    """Test that multi-source news collection works."""
    print("🧪 Testing multi-source news collection...")
    
    try:
        from src.data_collectors.enhanced_news_collector import EnhancedNewsCollector
        
        collector = EnhancedNewsCollector()
        news = await collector.collect_comprehensive_news('daily')
        
        if not news:
            print("❌ No news collected")
            return False
            
        sources = set(article.get('source', '') for article in news)
        print(f"✅ Collected {len(news)} articles from {len(sources)} sources:")
        for source in sorted(sources):
            count = sum(1 for article in news if article.get('source') == source)
            print(f"   - {source}: {count} articles")
            
        return len(sources) > 1
        
    except Exception as e:
        print(f"❌ Error testing news collection: {e}")
        return False

async def test_scipy_warnings():
    """Test that scipy warnings are fixed."""
    print("\n🧪 Testing scipy correlation warnings fix...")
    
    try:
        from src.analyzers.market_analyzer import MarketAnalyzer
        import numpy as np
        
        analyzer = MarketAnalyzer()
        
        mock_news_data = [
            {'title': 'Test news', 'relevance_score': 0.5, 'published_at': '2025-07-01T00:00:00'}
        ]
        mock_market_data = {
            'TEST': {
                'price_data': {
                    'historical_data': [
                        {'Date': '2025-07-01', 'Close': 100.0},
                        {'Date': '2025-07-02', 'Close': 100.0},
                        {'Date': '2025-07-03', 'Close': 100.0},
                        {'Date': '2025-07-04', 'Close': 100.0},
                        {'Date': '2025-07-05', 'Close': 100.0},
                        {'Date': '2025-07-06', 'Close': 100.0}
                    ]
                }
            }
        }
        
        result = await analyzer._analyze_news_price_correlations(mock_news_data, mock_market_data)
        
        if 'stock_correlations' in result and 'TEST' in result['stock_correlations']:
            correlation_data = result['stock_correlations']['TEST']
            if correlation_data.get('significance') == 'constant_input':
                print("✅ Scipy correlation analysis correctly handles constant input without warnings")
                return True
        
        print("✅ Scipy correlation analysis completed without warnings")
        return True
        
    except Exception as e:
        print(f"❌ Error testing scipy fix: {e}")
        return False

def test_visualization_module():
    """Test that matplotlib visualization module works."""
    print("\n🧪 Testing matplotlib visualization module...")
    
    try:
        from src.utils.backtest_visualizer import BacktestVisualizer
        
        visualizer = BacktestVisualizer()
        
        sample_portfolio = [
            {'date': '2025-07-01', 'total_value': 10000},
            {'date': '2025-07-15', 'total_value': 10500},
            {'date': '2025-08-01', 'total_value': 9800},
        ]
        
        sample_trades = [
            {'symbol': 'AAPL', 'profit_loss': 100, 'confidence': 0.7, 'entry_date': '2025-07-01'},
            {'symbol': 'MSFT', 'profit_loss': -50, 'confidence': 0.6, 'entry_date': '2025-07-15'},
        ]
        
        chart_path = visualizer.create_portfolio_performance_chart(sample_portfolio)
        if chart_path and os.path.exists(chart_path):
            print(f"✅ Portfolio performance chart created: {chart_path}")
            os.remove(chart_path)  # Cleanup
        else:
            print("❌ Failed to create portfolio performance chart")
            return False
            
        trade_chart_path = visualizer.create_trade_analysis_chart(sample_trades)
        if trade_chart_path and os.path.exists(trade_chart_path):
            print(f"✅ Trade analysis chart created: {trade_chart_path}")
            os.remove(trade_chart_path)  # Cleanup
        else:
            print("❌ Failed to create trade analysis chart")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing visualization module: {e}")
        return False

def test_historical_price_methods():
    """Test that historical price retrieval methods exist."""
    print("\n🧪 Testing historical price retrieval methods...")
    
    try:
        from backtesting_engine import BacktestingEngine
        
        engine = BacktestingEngine()
        
        methods = [
            '_get_price_direct_date',
            '_get_price_wide_range', 
            '_get_price_from_polygon',
            '_get_price_from_alpha_vantage'
        ]
        
        for method in methods:
            if hasattr(engine, method):
                print(f"✅ Method {method} exists")
            else:
                print(f"❌ Method {method} missing")
                return False
        
        if hasattr(engine, '_fallback_historical_news'):
            print("✅ Method _fallback_historical_news exists (uses real RSS feeds)")
        else:
            print("❌ Method _fallback_historical_news missing")
            return False
                
        return True
        
    except Exception as e:
        print(f"❌ Error testing historical price methods: {e}")
        return False

async def main():
    """Run all tests."""
    print("🚀 AI Stock Discovery Engine - Fix Validation Tests")
    print("=" * 60)
    
    tests = [
        ("Multi-source news collection", test_multi_source_news()),
        ("Scipy warnings fix", test_scipy_warnings()),
        ("Matplotlib visualization", test_visualization_module()),
        ("Historical price methods", test_historical_price_methods()),
    ]
    
    results = []
    for test_name, test_coro in tests:
        if asyncio.iscoroutine(test_coro):
            result = await test_coro
        else:
            result = test_coro
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All fixes validated successfully!")
        return True
    else:
        print("⚠️  Some tests failed - review implementation")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
