#!/usr/bin/env python3
"""
Enhanced AI Stock Discovery Engine - Deep Analysis Pipeline
Implements the full AI-driven analysis process:
1. Real news collection from multiple sources
2. LLM analysis to identify affected stocks (including niche companies)
3. Historical price movement analysis
4. News impact correlation analysis
5. ML-enhanced predictions
6. Comprehensive recommendations
"""

import asyncio
import logging
from datetime import datetime
from dotenv import load_dotenv

# Enhanced components
from src.data_collectors.enhanced_news_collector import EnhancedNewsCollector
from src.analyzers.enhanced_llm_analyzer import EnhancedLLMAnalyzer
from src.data_collectors.market_data_collector import MarketDataCollector
from src.analyzers.market_analyzer import MarketAnalyzer
from src.ml_models.prediction_model import PredictionModel
from src.decision_engine.investment_engine import InvestmentEngine
from src.utils.logger import setup_logger
from src.utils.price_visualizer import format_price_summary

# Load environment variables
load_dotenv()

async def run_enhanced_discovery_pipeline():
    """Run the enhanced AI-driven stock discovery pipeline."""
    
    logger = setup_logger(__name__)
    logger.info("🚀 Starting Enhanced AI Stock Discovery Engine...")
    
    # Initialize enhanced components
    news_collector = EnhancedNewsCollector()
    llm_analyzer = EnhancedLLMAnalyzer()
    market_collector = MarketDataCollector()
    market_analyzer = MarketAnalyzer()
    ml_model = PredictionModel()
    investment_engine = InvestmentEngine()
    
    try:
        # Step 1: Collect comprehensive real news
        print("📰 Step 1: Collecting comprehensive financial news...")
        start_time = datetime.now()
        
        news_data = await news_collector.collect_comprehensive_news("daily")
        
        collection_time = (datetime.now() - start_time).total_seconds()
        print(f"   ✅ Collected {len(news_data)} articles in {collection_time:.1f}s")
        
        if len(news_data) < 5:
            print("   ⚠️  Warning: Limited news data may affect analysis quality")
        
        # Step 2: Deep LLM analysis to identify affected stocks
        print("🧠 Step 2: Performing deep LLM analysis to identify affected stocks...")
        start_time = datetime.now()
        
        stock_analysis = await llm_analyzer.analyze_news_and_identify_stocks(news_data)
        identified_stocks = stock_analysis.get('stocks', [])
        
        analysis_time = (datetime.now() - start_time).total_seconds()
        print(f"   ✅ Identified {len(identified_stocks)} potentially affected stocks in {analysis_time:.1f}s")
        print(f"   📊 Stocks: {', '.join(identified_stocks[:10])}{'...' if len(identified_stocks) > 10 else ''}")
        
        if not identified_stocks:
            print("   ❌ No stocks identified - check news quality and LLM analysis")
            return
        
        # Step 3: Collect market data for identified stocks
        print("📈 Step 3: Collecting market data and historical prices...")
        start_time = datetime.now()
        
        market_data = await market_collector.collect_stock_data(identified_stocks, "daily")
        
        market_time = (datetime.now() - start_time).total_seconds()
        print(f"   ✅ Collected market data for {len(market_data)} stocks in {market_time:.1f}s")
        
        # Step 4: Analyze news impact on price movements
        print("🔍 Step 4: Analyzing news impact on stock price movements...")
        start_time = datetime.now()
        
        price_impact_analyses = {}
        for stock in identified_stocks[:10]:  # Analyze top 10 stocks
            if stock in market_data:
                impact_analysis = await llm_analyzer.analyze_stock_price_impact(
                    stock, news_data, market_data
                )
                if impact_analysis:
                    price_impact_analyses[stock] = impact_analysis
                
                # Rate limiting for API calls
                await asyncio.sleep(1)
        
        impact_time = (datetime.now() - start_time).total_seconds()
        print(f"   ✅ Analyzed price impact for {len(price_impact_analyses)} stocks in {impact_time:.1f}s")
        
        # Step 5: Market pattern analysis
        print("📊 Step 5: Analyzing market patterns and correlations...")
        start_time = datetime.now()
        
        # Create placeholder sentiment for market analyzer
        placeholder_sentiment = {'combined_sentiment': {'stock_sentiments': {}}}
        market_analysis = await market_analyzer.analyze_market_reactions(news_data, market_data, placeholder_sentiment)
        
        pattern_time = (datetime.now() - start_time).total_seconds()
        print(f"   ✅ Market analysis completed in {pattern_time:.1f}s")
        
        # Step 6: ML predictions enhanced with news analysis
        print("🤖 Step 6: Generating ML predictions with news context...")
        start_time = datetime.now()
        
        # Enhance market analysis with LLM insights
        enhanced_market_analysis = {
            **market_analysis,
            'llm_stock_analysis': stock_analysis.get('analysis', {}),
            'price_impact_analyses': price_impact_analyses
        }
        
        ml_predictions = await ml_model.predict_stock_movements(
            enhanced_market_analysis, 
            {'combined_sentiment': {'stock_sentiments': {}}},  # Placeholder
            market_data
        )
        
        ml_time = (datetime.now() - start_time).total_seconds()
        print(f"   ✅ ML predictions generated in {ml_time:.1f}s")
        
        # Step 7: Generate investment recommendations
        print("💡 Step 7: Generating investment recommendations...")
        start_time = datetime.now()
        
        recommendations = await investment_engine.generate_recommendations(
            news_data,
            {'combined_sentiment': {'stock_sentiments': {}}},  # Placeholder
            enhanced_market_analysis,
            ml_predictions
        )
        
        rec_time = (datetime.now() - start_time).total_seconds()
        print(f"   ✅ Recommendations generated in {rec_time:.1f}s")
        
        # Display results
        await display_enhanced_results(
            news_data, 
            stock_analysis, 
            price_impact_analyses, 
            recommendations, 
            market_data
        )
        
    except Exception as e:
        logger.error(f"Error in enhanced discovery pipeline: {str(e)}")
        print(f"❌ Pipeline error: {str(e)}")

async def display_enhanced_results(news_data, stock_analysis, price_impact_analyses, 
                                 recommendations, market_data):
    """Display comprehensive analysis results."""
    
    print("\n" + "="*80)
    print("🎯 ENHANCED AI STOCK DISCOVERY ENGINE - DEEP ANALYSIS RESULTS")
    print("="*80)
    
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
        
        # Price impact analysis if available
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
            
            # Historical price context
            if symbol in market_data:
                stock_data = market_data[symbol]
                price_data = stock_data.get('price_data', {})
                historical_data = price_data.get('historical_data', [])
                
                if historical_data and len(historical_data) >= 5:
                    price_summary = format_price_summary(historical_data)
                    print(f"      📊 Recent Price Movement:")
                    for line in price_summary.split('\n')[1:4]:  # Show key lines
                        if line.strip():
                            print(f"         {line}")
        
        # Summary
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

if __name__ == "__main__":
    asyncio.run(run_enhanced_discovery_pipeline())
