#!/usr/bin/env python3
"""
Comprehensive test suite for the AI Stock Discovery Engine.
Consolidates functionality from scattered test scripts into a unified testing framework.
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Dict, Any
from dotenv import load_dotenv

from src.data_collectors.market_data_collector import MarketDataCollector
from src.data_collectors.enhanced_news_collector import EnhancedNewsCollector
from src.analyzers.enhanced_llm_analyzer import EnhancedLLMAnalyzer
from src.decision_engine.investment_engine import InvestmentEngine
from src.utils.logger import setup_logger

load_dotenv()

class ConsolidatedTestSuite:
    """Comprehensive test suite for all engine components."""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.test_results = {}
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all component tests and return comprehensive results."""
        print("\n" + "="*80)
        print("🧪 AI STOCK DISCOVERY ENGINE - COMPREHENSIVE TEST SUITE")
        print("="*80)
        
        tests = [
            ("Market Data Collection", self.test_market_data_collection),
            ("News Collection", self.test_news_collection),
            ("LLM Analysis", self.test_llm_analysis),
            ("Recommendation Generation", self.test_recommendation_generation),
            ("End-to-End Pipeline", self.test_end_to_end_pipeline)
        ]
        
        overall_success = True
        
        for test_name, test_func in tests:
            print(f"\n{'='*60}")
            print(f"🔬 RUNNING: {test_name}")
            print(f"{'='*60}")
            
            try:
                start_time = time.time()
                result = await test_func()
                duration = time.time() - start_time
                
                self.test_results[test_name] = {
                    'success': result,
                    'duration': duration,
                    'timestamp': datetime.now().isoformat()
                }
                
                if result:
                    print(f"✅ {test_name} PASSED ({duration:.1f}s)")
                else:
                    print(f"❌ {test_name} FAILED ({duration:.1f}s)")
                    overall_success = False
                    
            except Exception as e:
                duration = time.time() - start_time if 'start_time' in locals() else 0
                print(f"❌ {test_name} ERROR: {str(e)} ({duration:.1f}s)")
                self.test_results[test_name] = {
                    'success': False,
                    'duration': duration,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                overall_success = False
        
        await self.display_test_summary(overall_success)
        return self.test_results
    
    async def test_market_data_collection(self) -> bool:
        """Test market data collection with rate limiting."""
        collector = MarketDataCollector()
        
        test_stocks = ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
        
        print(f"Testing with {len(test_stocks)} stocks: {test_stocks}")
        print(f"Polygon enabled: {collector.polygon_enabled}")
        print(f"YFinance enabled: {collector.yfinance_enabled}")
        
        try:
            stock_data = await collector.collect_stock_data(test_stocks, "daily")
            
            successful_count = 0
            for symbol, data in stock_data.items():
                price_data = data.get('price_data', {})
                current_price = price_data.get('current_price', 0)
                data_source = price_data.get('data_source', 'unknown')
                
                if current_price > 0:
                    print(f"✅ {symbol}: ${current_price:.2f} (source: {data_source})")
                    successful_count += 1
                else:
                    print(f"❌ {symbol}: No valid price data")
            
            print(f"Success rate: {successful_count}/{len(stock_data)}")
            return successful_count >= len(test_stocks) * 0.8
            
        except Exception as e:
            print(f"Market data collection error: {str(e)}")
            return False
    
    async def test_news_collection(self) -> bool:
        """Test enhanced news collection from multiple sources."""
        collector = EnhancedNewsCollector()
        
        try:
            print("Collecting daily financial news...")
            news_data = await collector.collect_comprehensive_news("daily")
            
            print(f"Collected {len(news_data)} articles")
            
            if len(news_data) >= 5:
                print("Sample articles:")
                for i, article in enumerate(news_data[:3], 1):
                    print(f"  {i}. {article.get('title', 'N/A')[:60]}...")
                    print(f"     Source: {article.get('source', 'N/A')} | Relevance: {article.get('relevance_score', 0):.2f}")
                
                return True
            else:
                print("Insufficient news articles collected")
                return False
                
        except Exception as e:
            print(f"News collection error: {str(e)}")
            return False
    
    async def test_llm_analysis(self) -> bool:
        """Test LLM analysis with sample news data."""
        analyzer = EnhancedLLMAnalyzer()
        
        if not analyzer.enabled:
            print("⚠️  OpenAI API not available - skipping LLM test")
            return True
        
        sample_news = [
            {
                'title': 'Apple reports strong quarterly earnings with record iPhone sales',
                'description': 'AAPL beats expectations with 15% revenue growth',
                'source': 'Reuters',
                'published_at': datetime.now().isoformat()
            },
            {
                'title': 'Tesla announces new Gigafactory in Texas',
                'description': 'TSLA expands manufacturing capacity with $5B investment',
                'source': 'Bloomberg',
                'published_at': datetime.now().isoformat()
            }
        ]
        
        try:
            print("Analyzing sample news with LLM...")
            stock_analysis = await analyzer.analyze_news_and_identify_stocks(sample_news)
            
            identified_stocks = stock_analysis.get('stocks', [])
            analysis_data = stock_analysis.get('analysis', {})
            
            print(f"Identified {len(identified_stocks)} stocks: {identified_stocks}")
            
            for stock in identified_stocks[:3]:
                stock_info = analysis_data.get(stock, {})
                print(f"  {stock}: {stock_info.get('impact_direction', 'N/A')} impact, confidence: {stock_info.get('confidence', 0)}")
            
            return len(identified_stocks) > 0
            
        except Exception as e:
            print(f"LLM analysis error: {str(e)}")
            return False
    
    async def test_recommendation_generation(self) -> bool:
        """Test recommendation generation with mock data."""
        engine = InvestmentEngine()
        
        mock_news_data = [
            {
                'title': 'Apple reports strong quarterly earnings',
                'description': 'AAPL beats expectations with record iPhone sales',
                'source': 'Reuters'
            }
        ]
        
        mock_sentiment_analysis = {
            'combined_sentiment': {
                'stock_sentiments': {
                    'AAPL': {
                        'combined_score': 0.7,
                        'confidence': 0.8,
                        'social_sentiment': 0.6,
                        'market_sentiment': 0.8
                    }
                }
            }
        }
        
        mock_market_analysis = {
            'market_inefficiencies': [],
            'volume_patterns': {},
            'momentum_analysis': {}
        }
        
        mock_ml_predictions = {
            'individual_predictions': {
                'AAPL': {
                    'predicted_return': 0.08,
                    'confidence': 0.7,
                    'direction': 'bullish'
                }
            }
        }
        
        try:
            print("Generating recommendations with mock data...")
            recommendations = await engine.generate_recommendations(
                mock_news_data,
                mock_sentiment_analysis,
                mock_market_analysis,
                mock_ml_predictions
            )
            
            recs = recommendations.get('recommendations', [])
            print(f"Generated {len(recs)} recommendations")
            
            if recs:
                for i, rec in enumerate(recs[:2], 1):
                    print(f"  {i}. {rec.get('symbol', 'N/A')} - {rec.get('action', 'N/A')}")
                    print(f"     Confidence: {rec.get('confidence', 0):.2%}")
            
            return len(recs) > 0
            
        except Exception as e:
            print(f"Recommendation generation error: {str(e)}")
            return False
    
    async def test_end_to_end_pipeline(self) -> bool:
        """Test the complete pipeline with minimal data."""
        try:
            from consolidated_main import ConsolidatedStockDiscoveryEngine
            
            print("Testing end-to-end pipeline...")
            engine = ConsolidatedStockDiscoveryEngine()
            
            results = await engine.run_discovery_pipeline("daily")
            
            required_keys = ['timestamp', 'news_stories', 'stocks_analyzed', 'recommendations']
            
            for key in required_keys:
                if key not in results:
                    print(f"Missing required key: {key}")
                    return False
            
            print(f"Pipeline completed successfully:")
            print(f"  News stories: {results.get('news_stories', 0)}")
            print(f"  Stocks analyzed: {results.get('stocks_analyzed', 0)}")
            print(f"  Recommendations: {len(results.get('recommendations', {}).get('recommendations', []))}")
            
            return True
            
        except Exception as e:
            print(f"End-to-end pipeline error: {str(e)}")
            return False
    
    async def display_test_summary(self, overall_success: bool):
        """Display comprehensive test results summary."""
        print(f"\n{'='*80}")
        print("📊 TEST SUITE SUMMARY")
        print(f"{'='*80}")
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['success'])
        failed_tests = total_tests - passed_tests
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Success Rate: {passed_tests/total_tests*100:.1f}%")
        
        total_duration = sum(result['duration'] for result in self.test_results.values())
        print(f"Total Duration: {total_duration:.1f}s")
        
        if overall_success:
            print(f"\n🎉 ALL TESTS PASSED - System is ready for production!")
        else:
            print(f"\n⚠️  SOME TESTS FAILED - Review errors above")
            
        print(f"{'='*80}")

async def main():
    """Run the comprehensive test suite."""
    test_suite = ConsolidatedTestSuite()
    results = await test_suite.run_all_tests()
    return results

if __name__ == "__main__":
    asyncio.run(main())
