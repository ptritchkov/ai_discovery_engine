"""
AI Stock Discovery Engine
Main entry point for the stock discovery and analysis system.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

from src.data_collectors.news_collector import NewsCollector
from src.data_collectors.twitter_collector import TwitterCollector
from src.data_collectors.polymarket_collector import PolymarketCollector
from src.data_collectors.market_data_collector import MarketDataCollector
from src.analyzers.sentiment_analyzer import SentimentAnalyzer
from src.analyzers.llm_analyzer import LLMAnalyzer
from src.analyzers.market_analyzer import MarketAnalyzer
from src.ml_models.prediction_model import PredictionModel
from src.decision_engine.investment_engine import InvestmentEngine
from src.utils.logger import setup_logger
from src.utils.database import Database
from src.utils.config import config

load_dotenv()

class StockDiscoveryEngine:
    def __init__(self):
        self.logger = setup_logger(__name__)
        self.db = Database()
        
        if not config.validate_configuration():
            raise ValueError("Invalid configuration: At least one data source must be enabled")
        
        self.news_collector = NewsCollector() if config.is_enabled('news') else None
        self.twitter_collector = TwitterCollector() if config.is_enabled('twitter') else None
        self.polymarket_collector = PolymarketCollector() if config.is_enabled('polymarket') else None
        self.market_data_collector = MarketDataCollector() if (config.is_enabled('polygon') or config.is_enabled('yfinance')) else None
        
        self.sentiment_analyzer = SentimentAnalyzer()
        self.llm_analyzer = LLMAnalyzer()
        self.market_analyzer = MarketAnalyzer()
        
        self.prediction_model = PredictionModel()
        self.investment_engine = InvestmentEngine()
        
        enabled_services = config.get_enabled_services()
        enabled_count = sum(enabled_services.values())
        self.logger.info(f"AI Stock Discovery Engine initialized with {enabled_count} enabled services: {enabled_services}")
        
    async def run_discovery_cycle(self, timeframe: str = "daily") -> Dict[str, Any]:
        """
        Run a complete discovery cycle to analyze market opportunities.
        
        Args:
            timeframe: "daily" or "weekly" analysis timeframe
            
        Returns:
            Dictionary containing analysis results and recommendations
        """
        self.logger.info(f"Starting {timeframe} discovery cycle...")
        
        try:
            self.logger.info("Collecting news data...")
            news_data = []
            if self.news_collector:
                news_data = await self.news_collector.collect_latest_news(timeframe)
            
            self.logger.info("Analyzing news for stock implications...")
            affected_stocks = await self.llm_analyzer.identify_affected_stocks(news_data)
            
            self.logger.info("Collecting sentiment data...")
            twitter_sentiment = {}
            if self.twitter_collector:
                twitter_sentiment = await self.twitter_collector.collect_sentiment_data(affected_stocks)
            
            polymarket_data = {}
            if self.polymarket_collector:
                polymarket_data = await self.polymarket_collector.collect_market_sentiment(affected_stocks)
            
            self.logger.info("Collecting market data...")
            market_data = {}
            if self.market_data_collector:
                market_data = await self.market_data_collector.collect_stock_data(affected_stocks, timeframe)
            
            self.logger.info("Analyzing sentiment...")
            sentiment_analysis = await self.sentiment_analyzer.analyze_comprehensive_sentiment(
                news_data, twitter_sentiment, polymarket_data
            )
            
            self.logger.info("Analyzing market patterns...")
            market_analysis = await self.market_analyzer.analyze_market_reactions(
                news_data, market_data, sentiment_analysis
            )
            
            self.logger.info("Generating ML predictions...")
            ml_predictions = await self.prediction_model.predict_stock_movements(
                market_analysis, sentiment_analysis, market_data
            )
            
            self.logger.info("Generating investment recommendations...")
            recommendations = await self.investment_engine.generate_recommendations(
                news_data, sentiment_analysis, market_analysis, ml_predictions
            )
            
            results = {
                "timestamp": datetime.now().isoformat(),
                "timeframe": timeframe,
                "news_stories": len(news_data),
                "stocks_analyzed": len(affected_stocks),
                "sentiment_analysis": sentiment_analysis,
                "market_analysis": market_analysis,
                "ml_predictions": ml_predictions,
                "recommendations": recommendations
            }
            
            await self.db.store_analysis_results(results)
            
            self.logger.info(f"Discovery cycle completed. Found {len(recommendations)} recommendations.")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in discovery cycle: {str(e)}")
            raise
    
    async def get_top_stories_summary(self, timeframe: str = "daily") -> Dict[str, Any]:
        """Get a summary of top stories and their market implications."""
        results = await self.run_discovery_cycle(timeframe)
        
        summary = {
            "top_stories": results.get("news_stories", [])[:10],
            "market_movers": results.get("recommendations", [])[:5],
            "sentiment_overview": results.get("sentiment_analysis", {}),
            "confidence_scores": [rec.get("confidence", 0) for rec in results.get("recommendations", [])]
        }
        
        return summary

async def main():
    """Main function to run the stock discovery engine."""
    engine = StockDiscoveryEngine()
    
    daily_results = await engine.run_discovery_cycle("daily")
    
    print("\n" + "="*80)
    print("AI STOCK DISCOVERY ENGINE - DAILY ANALYSIS")
    print("="*80)
    
    recommendations_data = daily_results.get("recommendations", {})
    recommendations = recommendations_data.get("recommendations", []) if isinstance(recommendations_data, dict) else recommendations_data
    
    if recommendations:
        print(f"\nFound {len(recommendations)} investment opportunities:")
        for i, rec in enumerate(recommendations[:5], 1):
            print(f"\n{i}. {rec.get('symbol', 'N/A')} - {rec.get('action', 'N/A')}")
            print(f"   Confidence: {rec.get('confidence', 0):.2%}")
            print(f"   Expected Return: {rec.get('expected_return', 0)*100:.1f}%")
            print(f"   Reasoning: {rec.get('reasoning', 'N/A')[:100]}...")
        
        summary = recommendations_data.get("summary", {}) if isinstance(recommendations_data, dict) else {}
        if summary:
            print(f"\nMarket Outlook: {summary.get('market_outlook', 'Unknown')}")
            print(f"Buy Recommendations: {summary.get('buy_recommendations', 0)}")
            print(f"Sell Recommendations: {summary.get('sell_recommendations', 0)}")
    else:
        print("\nNo strong investment opportunities identified today.")
    
    print(f"\nAnalysis completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
