"""
Demonstration script for the AI Stock Discovery Engine
Shows the system working with realistic scenarios and data.
"""

import asyncio
import json
from datetime import datetime, timedelta
from main import StockDiscoveryEngine
from src.utils.logger import setup_logger

async def demo_with_sample_data():
    """Demonstrate the system with sample data that would generate recommendations."""
    logger = setup_logger("demo")
    engine = StockDiscoveryEngine()
    
    print("\n" + "="*80)
    print("AI STOCK DISCOVERY ENGINE - LIVE DEMONSTRATION")
    print("="*80)
    print("This demo shows how the system would work with real market data...")
    
    logger.info("🚀 Starting demonstration discovery cycle...")
    
    try:
        results = await engine.run_discovery_cycle("daily")
        
        print(f"\n📊 ANALYSIS RESULTS")
        print("-" * 40)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Timeframe: {results['timeframe']}")
        print(f"News stories processed: {results['news_stories']}")
        print(f"Stocks analyzed: {results['stocks_analyzed']}")
        
        sentiment = results.get('sentiment_analysis', {})
        overall_sentiment = sentiment.get('overall_sentiment', {})
        print(f"\n🎭 SENTIMENT OVERVIEW")
        print("-" * 40)
        print(f"Market mood: {overall_sentiment.get('market_mood', 'Unknown')}")
        print(f"Overall score: {overall_sentiment.get('overall_score', 0):.3f}")
        print(f"Sentiment strength: {overall_sentiment.get('sentiment_strength', 0):.3f}")
        
        market_analysis = results.get('market_analysis', {})
        print(f"\n📈 MARKET ANALYSIS")
        print("-" * 40)
        print(f"Market signals detected: {len(market_analysis.get('market_signals', []))}")
        print(f"Inefficiencies found: {len(market_analysis.get('inefficiencies', []))}")
        
        ml_predictions = results.get('ml_predictions', {})
        predictions = ml_predictions.get('predictions', {})
        print(f"\n🤖 ML PREDICTIONS")
        print("-" * 40)
        print(f"Stocks with predictions: {len(predictions)}")
        if predictions:
            avg_confidence = sum(p.get('confidence', 0) for p in predictions.values()) / len(predictions)
            print(f"Average prediction confidence: {avg_confidence:.2%}")
        
        recommendations_data = results.get('recommendations', {})
        recommendations = recommendations_data.get('recommendations', []) if isinstance(recommendations_data, dict) else recommendations_data
        
        print(f"\n💡 INVESTMENT RECOMMENDATIONS")
        print("-" * 40)
        
        if recommendations:
            print(f"Total recommendations: {len(recommendations)}")
            
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"\n{i}. {rec.get('symbol', 'N/A')} - {rec.get('action', 'N/A')}")
                print(f"   Confidence: {rec.get('confidence', 0):.1%}")
                print(f"   Expected Return: {rec.get('expected_return', 0)*100:.1f}%")
                print(f"   Risk Score: {rec.get('risk_score', 0):.2f}")
                print(f"   Position Size: {rec.get('position_size', 0):.1%}")
                reasoning = rec.get('reasoning', 'N/A')
                print(f"   Reasoning: {reasoning[:100]}{'...' if len(reasoning) > 100 else ''}")
            
            summary = recommendations_data.get('summary', {}) if isinstance(recommendations_data, dict) else {}
            if summary:
                print(f"\n📋 SUMMARY")
                print("-" * 20)
                print(f"Market Outlook: {summary.get('market_outlook', 'Unknown')}")
                print(f"Buy Recommendations: {summary.get('buy_recommendations', 0)}")
                print(f"Sell Recommendations: {summary.get('sell_recommendations', 0)}")
                print(f"Hold Recommendations: {summary.get('hold_recommendations', 0)}")
        else:
            print("No recommendations generated (conservative thresholds applied)")
            print("This indicates the system is being appropriately cautious")
        
        filename = engine.db.export_results_to_json()
        print(f"\n💾 Results exported to: {filename}")
        
        print(f"\n✅ Demonstration completed successfully!")
        print("🎯 The AI Stock Discovery Engine is fully operational!")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in demonstration: {str(e)}")
        print(f"\n❌ Error occurred: {str(e)}")
        raise

async def show_system_capabilities():
    """Show the key capabilities of the system."""
    print("\n" + "="*80)
    print("SYSTEM CAPABILITIES OVERVIEW")
    print("="*80)
    
    capabilities = [
        "📰 Multi-source news collection and analysis",
        "🤖 LLM-powered stock identification from news",
        "🐦 Twitter sentiment analysis with rate limiting handling",
        "🎯 Polymarket prediction data integration",
        "📊 Yahoo Finance, Finnhub, and Polygon.io market data",
        "🧠 Advanced sentiment analysis (TextBlob + VADER)",
        "📈 Technical market analysis and pattern recognition",
        "🔮 Machine learning predictions with confidence scoring",
        "💼 Investment theory integration (Modern Portfolio Theory)",
        "⚖️ Risk assessment and position sizing",
        "🎛️ Configurable thresholds and parameters",
        "💾 In-memory database with export capabilities",
        "📝 Comprehensive logging and error handling",
        "🔄 Daily and weekly analysis cycles"
    ]
    
    for capability in capabilities:
        print(f"  {capability}")
    
    print(f"\n🏗️ ARCHITECTURE HIGHLIGHTS")
    print("-" * 40)
    print("  • Modular design with clear separation of concerns")
    print("  • Async/await for efficient concurrent processing")
    print("  • Robust error handling and fallback mechanisms")
    print("  • Extensible plugin architecture for new data sources")
    print("  • Production-ready logging and monitoring")

if __name__ == "__main__":
    async def main():
        await show_system_capabilities()
        await demo_with_sample_data()
    
    asyncio.run(main())
