#!/usr/bin/env python3
"""
AI Stock Discovery Engine - Unified Main Entry Point
Consolidates all functionality into a single script with multiple operation modes.
"""

import asyncio
import argparse
import sys
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

from consolidated_main import ConsolidatedStockDiscoveryEngine
from backtesting_engine import BacktestingEngine
from test_suite import ConsolidatedTestSuite

load_dotenv()

async def run_discovery(args):
    """Run live stock discovery analysis."""
    print("\n" + "="*80)
    print("🎯 AI STOCK DISCOVERY ENGINE - LIVE ANALYSIS")
    print("="*80)
    
    engine = ConsolidatedStockDiscoveryEngine()
    results = await engine.run_discovery_pipeline(args.period)
    
    print(f"\n🎉 Analysis completed successfully!")
    print(f"   News articles analyzed: {results.get('news_stories', 0)}")
    print(f"   Stocks identified: {results.get('stocks_analyzed', 0)}")
    print(f"   Recommendations generated: {len(results.get('recommendations', {}).get('recommendations', []))}")
    
    return results

async def run_backtesting(args):
    """Run historical backtesting with portfolio simulation."""
    print("\n" + "="*80)
    print("📊 AI STOCK DISCOVERY ENGINE - HISTORICAL BACKTESTING")
    print("="*80)
    
    engine = BacktestingEngine()
    results = await engine.run_historical_backtest(
        start_weeks_ago=args.weeks_back,
        end_weeks_ago=1,
        interval_days=args.interval,
        initial_portfolio=args.portfolio
    )
    
    if 'error' not in results:
        portfolio_perf = results.get('performance_analysis', {}).get('portfolio_performance', {})
        print(f"\n🎉 Backtesting completed successfully!")
        print(f"   Total periods: {results.get('backtest_summary', {}).get('total_periods', 0)}")
        print(f"   Success rate: {results.get('backtest_summary', {}).get('success_rate', 0):.1%}")
        if portfolio_perf:
            print(f"   Portfolio return: {portfolio_perf.get('total_return_pct', 0):+.2f}%")
            print(f"   Final value: ${portfolio_perf.get('final_value', 0):,.2f}")
    
    return results

async def run_tests():
    """Run comprehensive test suite."""
    print("\n" + "="*80)
    print("🧪 AI STOCK DISCOVERY ENGINE - COMPREHENSIVE TESTING")
    print("="*80)
    
    test_suite = ConsolidatedTestSuite()
    results = await test_suite.run_all_tests()
    
    return results

async def main():
    """Main entry point with command-line interface."""
    parser = argparse.ArgumentParser(
        description="AI Stock Discovery Engine - Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                                    # Run live discovery (default)
  python main.py --mode discovery --period weekly  # Weekly discovery analysis
  python main.py --mode backtest --weeks-back 12   # 3-month backtesting
  python main.py --mode backtest --portfolio 50000 # Backtest with $50K portfolio
  python main.py --mode test                        # Run test suite
        """
    )
    
    parser.add_argument('--mode', choices=['discovery', 'backtest', 'test'], 
                       default='discovery', help='Operation mode (default: discovery)')
    parser.add_argument('--period', choices=['daily', 'weekly'], 
                       default='daily', help='Analysis time period (default: daily)')
    parser.add_argument('--weeks-back', type=int, default=12, 
                       help='Weeks back for backtesting (default: 12)')
    parser.add_argument('--interval', type=int, default=3, 
                       help='Days between backtest points (default: 3)')
    parser.add_argument('--portfolio', type=float, default=10000.0, 
                       help='Initial portfolio value for backtesting (default: 10000)')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting AI Stock Discovery Engine in {args.mode.upper()} mode...")
    print(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    
    try:
        if args.mode == 'discovery':
            results = await run_discovery(args)
        elif args.mode == 'backtest':
            results = await run_backtesting(args)
        elif args.mode == 'test':
            results = await run_tests()
        
        print(f"\n✅ {args.mode.capitalize()} operation completed successfully!")
        return results
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        return None
    except Exception as e:
        print(f"\n❌ Fatal error in {args.mode} mode: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        results = asyncio.run(main())
        if results is None:
            sys.exit(1)
        sys.exit(0)
    except Exception as e:
        print(f"❌ Application error: {str(e)}")
        sys.exit(1)
