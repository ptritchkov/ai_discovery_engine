#!/usr/bin/env python3
"""
Consolidated AI Stock Discovery Engine
Unified entry point combining enhanced analysis with configurable data sources.

This consolidates functionality from main.py and enhanced_main.py into a single,
clean pipeline that supports both live analysis and backtesting.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv

from src.data_collectors.enhanced_news_collector import EnhancedNewsCollector
from src.analyzers.enhanced_llm_analyzer import EnhancedLLMAnalyzer
from src.data_collectors.market_data_collector import MarketDataCollector
from src.analyzers.market_analyzer import MarketAnalyzer
from src.ml_models.prediction_model import PredictionModel
from src.decision_engine.investment_engine import InvestmentEngine
from src.utils.logger import setup_logger
from src.utils.price_visualizer import format_price_summary
from src.utils.config import config

load_dotenv()

class ConsolidatedStockDiscoveryEngine:
    """Unified AI Stock Discovery Engine with enhanced analysis capabilities."""
    
    def __init__(self):
        self.logger = setup_logger(__name__)
        
        if not config.validate_configuration():
            raise ValueError("Invalid configuration: At least one data source must be enabled")
        
        self.news_collector = EnhancedNewsCollector()
        self.llm_analyzer = EnhancedLLMAnalyzer()
        self.market_collector = MarketDataCollector()
        self.market_analyzer = MarketAnalyzer()
        self.ml_model = PredictionModel()
        self.investment_engine = InvestmentEngine()
        
        enabled_services = config.get_enabled_services()
        enabled_count = sum(enabled_services.values())
        self.logger.info(f"Consolidated AI Stock Discovery Engine initialized with {enabled_count} enabled services")
        
    async def run_discovery_pipeline(self, timeframe: str = "daily") -> Dict[str, Any]:
        """
        Run the complete discovery pipeline with enhanced analysis.
        
        Args:
            timeframe: "daily" or "weekly" analysis timeframe
            
        Returns:
            Dictionary containing comprehensive analysis results and recommendations
        """
        self.logger.info(f"🚀 Starting Consolidated AI Stock Discovery Pipeline ({timeframe})...")
        
        try:
            # Step 1: Collect comprehensive real news
            print("📰 Step 1: Collecting comprehensive financial news...")
            start_time = datetime.now()
            
            news_data = await self.news_collector.collect_comprehensive_news(timeframe)
            
            collection_time = (datetime.now() - start_time).total_seconds()
            print(f"   ✅ Collected {len(news_data)} articles in {collection_time:.1f}s")
            
            if len(news_data) < 5:
                print("   ⚠️  Warning: Limited news data may affect analysis quality")
            
            # Step 2: Deep LLM analysis to identify affected stocks
            print("🧠 Step 2: Performing deep LLM analysis to identify affected stocks...")
            start_time = datetime.now()
            
            stock_analysis = await self.llm_analyzer.analyze_news_and_identify_stocks(news_data)
            identified_stocks = stock_analysis.get('stocks', [])
            
            analysis_time = (datetime.now() - start_time).total_seconds()
            print(f"   ✅ Identified {len(identified_stocks)} potentially affected stocks in {analysis_time:.1f}s")
            print(f"   📊 Stocks: {', '.join(identified_stocks[:10])}{'...' if len(identified_stocks) > 10 else ''}")
            
            if not identified_stocks:
                print("   ❌ No stocks identified - check news quality and LLM analysis")
                return self._create_empty_results(timeframe, news_data)
            
            # Step 3: Collect market data for identified stocks
            print("📈 Step 3: Collecting market data and historical prices...")
            start_time = datetime.now()
            
            market_data = await self.market_collector.collect_stock_data(identified_stocks, timeframe)
            
            market_time = (datetime.now() - start_time).total_seconds()
            print(f"   ✅ Collected market data for {len(market_data)} stocks in {market_time:.1f}s")
            
            # Step 4: Analyze news impact on price movements
            print("🔍 Step 4: Analyzing news impact on stock price movements...")
            start_time = datetime.now()
            
            price_impact_analyses = {}
            for stock in identified_stocks[:10]:
                if stock in market_data:
                    impact_analysis = await self.llm_analyzer.analyze_stock_price_impact(
                        stock, news_data, market_data
                    )
                    if impact_analysis:
                        price_impact_analyses[stock] = impact_analysis
                    
                    await asyncio.sleep(1)
            
            impact_time = (datetime.now() - start_time).total_seconds()
            print(f"   ✅ Analyzed price impact for {len(price_impact_analyses)} stocks in {impact_time:.1f}s")
            
            # Step 5: Market pattern analysis
            print("📊 Step 5: Analyzing market patterns and correlations...")
            start_time = datetime.now()
            
            placeholder_sentiment = {'combined_sentiment': {'stock_sentiments': {}}}
            market_analysis = await self.market_analyzer.analyze_market_reactions(
                news_data, market_data, placeholder_sentiment
            )
            
            pattern_time = (datetime.now() - start_time).total_seconds()
            print(f"   ✅ Market analysis completed in {pattern_time:.1f}s")
            
            # Step 6: ML predictions enhanced with news analysis
            print("🤖 Step 6: Generating ML predictions with news context...")
            start_time = datetime.now()
            
            enhanced_market_analysis = {
                **market_analysis,
                'llm_stock_analysis': stock_analysis.get('analysis', {}),
                'price_impact_analyses': price_impact_analyses
            }
            
            ml_predictions = await self.ml_model.predict_stock_movements(
                enhanced_market_analysis, 
                placeholder_sentiment,
                market_data
            )
            
            ml_time = (datetime.now() - start_time).total_seconds()
            print(f"   ✅ ML predictions generated in {ml_time:.1f}s")
            
            # Step 7: Generate investment recommendations
            print("💡 Step 7: Generating investment recommendations...")
            start_time = datetime.now()
            
            recommendations = await self.investment_engine.generate_recommendations(
                news_data,
                placeholder_sentiment,
                enhanced_market_analysis,
                ml_predictions
            )
            
            rec_time = (datetime.now() - start_time).total_seconds()
            print(f"   ✅ Recommendations generated in {rec_time:.1f}s")
            
            results = {
                "timestamp": datetime.now().isoformat(),
                "timeframe": timeframe,
                "news_stories": len(news_data),
                "stocks_analyzed": len(identified_stocks),
                "news_data": news_data,
                "stock_analysis": stock_analysis,
                "price_impact_analyses": price_impact_analyses,
                "market_analysis": enhanced_market_analysis,
                "ml_predictions": ml_predictions,
                "recommendations": recommendations,
                "market_data": market_data
            }
            
            await self.display_results(results)
            
            self.logger.info(f"Discovery pipeline completed successfully")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in discovery pipeline: {str(e)}")
            print(f"❌ Pipeline error: {str(e)}")
            return self._create_empty_results(timeframe, news_data if 'news_data' in locals() else [])
    
    def _create_empty_results(self, timeframe: str, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create empty results structure for error cases."""
        return {
            "timestamp": datetime.now().isoformat(),
            "timeframe": timeframe,
            "news_stories": len(news_data),
            "stocks_analyzed": 0,
            "news_data": news_data,
            "stock_analysis": {'stocks': [], 'analysis': {}},
            "price_impact_analyses": {},
            "market_analysis": {},
            "ml_predictions": {},
            "recommendations": {'recommendations': [], 'summary': {}},
            "market_data": {}
        }
    
    async def display_results(self, results: Dict[str, Any]):
        """Display comprehensive analysis results."""
        
        print("\n" + "="*80)
        print("🎯 CONSOLIDATED AI STOCK DISCOVERY ENGINE - ANALYSIS RESULTS")
        print("="*80)
        
        news_data = results.get('news_data', [])
        stock_analysis = results.get('stock_analysis', {})
        price_impact_analyses = results.get('price_impact_analyses', {})
        recommendations = results.get('recommendations', {})
        market_data = results.get('market_data', {})
        
        # News analysis summary
        print(f"\n📰 NEWS ANALYSIS SUMMARY")
        print(f"   • Total articles analyzed: {len(news_data)}")
        print(f"   • Stocks identified by AI: {len(stock_analysis.get('stocks', []))}")
        print(f"   • Deep price analysis performed: {len(price_impact_analyses)}")
        
        # Top news headlines
        print(f"\n📋 KEY NEWS HEADLINES:")
        for i, article in enumerate(news_data[:5], 1):
            print(f"   {i}. {article.get('title', 'N/A')[:80]}...")
            print(f"      Source: {article.get('source', 'N/A')} | Relevance: {article.get('relevance_score', 0):.2f}")
        
        # AI-identified stocks with reasoning
        print(f"\n🧠 AI-IDENTIFIED INVESTMENT OPPORTUNITIES:")
        analysis_data = stock_analysis.get('analysis', {})
        
        for i, stock in enumerate(stock_analysis.get('stocks', [])[:10], 1):
            stock_info = analysis_data.get(stock, {})
            print(f"\n   {i}. {stock} - {stock_info.get('company_name', 'Unknown Company')}")
            print(f"      Impact: {stock_info.get('impact_direction', 'MIXED')} | Score: {stock_info.get('impact_score', 0):.1f}/10")
            print(f"      Category: {stock_info.get('category', 'DIRECT')} | Confidence: {stock_info.get('confidence', 0)}/10")
            print(f"      Reasoning: {stock_info.get('reasoning', 'N/A')[:100]}...")
            
            if stock in price_impact_analyses:
                impact = price_impact_analyses[stock]
                prediction = impact.get('future_price_prediction', {})
                print(f"      💹 Price Prediction: {prediction.get('direction', 'N/A')} {prediction.get('magnitude', 'N/A')}")
                print(f"      🎯 Opportunity Score: {impact.get('opportunity_score', 'N/A')}/10")
        
        # Final recommendations
        recs = recommendations.get('recommendations', [])
        print(f"\n🎯 FINAL INVESTMENT RECOMMENDATIONS:")
        
        if recs:
            print(f"   Found {len(recs)} high-confidence opportunities:")
            
            for i, rec in enumerate(recs[:5], 1):
                symbol = rec.get('symbol', 'N/A')
                print(f"\n   {i}. {symbol} - {rec.get('action', 'N/A').upper()}")
                print(f"      Confidence: {rec.get('confidence', 0):.1%}")
                print(f"      Expected Return: {rec.get('expected_return', 0)*100:+.1f}%")
                print(f"      Position Size: {rec.get('position_size', 'N/A')}")
                print(f"      Risk Level: {rec.get('risk_level', 'N/A')}")
                print(f"      Time Horizon: {rec.get('time_horizon', 'N/A')}")
                print(f"      Reasoning: {rec.get('reasoning', 'N/A')[:120]}...")
                
                if symbol in market_data:
                    stock_data = market_data[symbol]
                    price_data = stock_data.get('price_data', {})
                    historical_data = price_data.get('historical_data', [])
                    
                    if historical_data and len(historical_data) >= 5:
                        price_summary = format_price_summary(historical_data)
                        print(f"      📊 Recent Price Movement:")
                        for line in price_summary.split('\n')[1:4]:
                            if line.strip():
                                print(f"         {line}")
            
            summary = recommendations.get('summary', {})
            print(f"\n📈 MARKET OUTLOOK: {summary.get('market_outlook', 'Unknown').upper()}")
            print(f"   • Buy Recommendations: {summary.get('buy_recommendations', 0)}")
            print(f"   • Sell Recommendations: {summary.get('sell_recommendations', 0)}")
            print(f"   • Total Opportunities: {recommendations.get('total_opportunities', 0)}")
            
        else:
            print("   ❌ No high-confidence recommendations generated")
            print("   💡 Consider adjusting confidence thresholds or expanding news sources")
        
        print(f"\n🕒 Analysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

async def main():
    """Main function to run the consolidated stock discovery engine."""
    engine = ConsolidatedStockDiscoveryEngine()
    
    results = await engine.run_discovery_pipeline("daily")
    
    return results

if __name__ == "__main__":
    asyncio.run(main())
