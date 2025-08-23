#!/usr/bin/env python3
"""
Test script to debug recommendation generation issues.
"""

import asyncio
import logging
from dotenv import load_dotenv
from src.decision_engine.investment_engine import InvestmentEngine
from src.utils.logger import setup_logger

# Load environment variables
load_dotenv()

async def test_recommendations():
    """Test recommendation generation with mock data."""
    logger = setup_logger(__name__)
    logger.info("Testing recommendation generation...")
    
    # Create investment engine
    engine = InvestmentEngine()
    
    # Mock data for testing
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
                },
                'MSFT': {
                    'combined_score': 0.5,
                    'confidence': 0.6,
                    'social_sentiment': 0.4,
                    'market_sentiment': 0.6
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
            },
            'MSFT': {
                'predicted_return': 0.05,
                'confidence': 0.6,
                'direction': 'bullish'
            }
        }
    }
    
    print("🧪 Testing recommendation generation with mock data...")
    print(f"   • News articles: {len(mock_news_data)}")
    print(f"   • Stocks with sentiment: {len(mock_sentiment_analysis['combined_sentiment']['stock_sentiments'])}")
    print(f"   • Stocks with ML predictions: {len(mock_ml_predictions['individual_predictions'])}")
    
    # Generate recommendations
    try:
        recommendations = await engine.generate_recommendations(
            mock_news_data,
            mock_sentiment_analysis,
            mock_market_analysis,
            mock_ml_predictions
        )
        
        print(f"\n📊 RESULTS:")
        print(f"   • Recommendations type: {type(recommendations)}")
        print(f"   • Recommendations keys: {list(recommendations.keys()) if isinstance(recommendations, dict) else 'Not a dict'}")
        
        if isinstance(recommendations, dict):
            recs = recommendations.get('recommendations', [])
            print(f"   • Number of recommendations: {len(recs)}")
            
            if recs:
                print(f"\n✅ Generated {len(recs)} recommendations:")
                for i, rec in enumerate(recs, 1):
                    print(f"   {i}. {rec.get('symbol', 'N/A')} - {rec.get('action', 'N/A')}")
                    print(f"      Confidence: {rec.get('confidence', 0):.2%}")
                    print(f"      Expected Return: {rec.get('expected_return', 0)*100:.1f}%")
            else:
                print("❌ No recommendations generated")
                summary = recommendations.get('summary', {})
                print(f"   Summary: {summary}")
        else:
            print("❌ Unexpected recommendations format")
            
    except Exception as e:
        print(f"❌ Error generating recommendations: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_recommendations())
