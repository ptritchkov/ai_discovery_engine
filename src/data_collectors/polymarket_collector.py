"""Polymarket data collector for prediction market sentiment."""

import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
import json
from src.utils.config import config

class PolymarketCollector:
    """Collects prediction market data from Polymarket."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.enabled = config.is_enabled('polymarket')
        self.base_url = "https://gamma-api.polymarket.com"
        
        if not self.enabled:
            self.logger.info("Polymarket API is disabled via configuration.")
        
    async def collect_market_sentiment(self, stocks: List[str]) -> Dict[str, Any]:
        """
        Collect prediction market sentiment for given stocks.
        
        Args:
            stocks: List of stock symbols to analyze
            
        Returns:
            Dictionary containing prediction market data for each stock
        """
        self.logger.info(f"Collecting Polymarket sentiment for {len(stocks)} stocks...")
        
        market_sentiment = {}
        
        if not self.enabled:
            self.logger.info("Polymarket API is disabled. Returning empty data.")
            return {}
        
        try:
            async with aiohttp.ClientSession() as session:
                for stock in stocks[:10]:  # Limit to avoid rate limits
                    try:
                        stock_data = await self._collect_stock_market_data(session, stock)
                        market_sentiment[stock] = stock_data
                        
                        await asyncio.sleep(0.5)
                        
                    except Exception as e:
                        self.logger.warning(f"Error collecting Polymarket data for {stock}: {str(e)}")
                        market_sentiment[stock] = self._get_default_market_data(stock)
                        
        except Exception as e:
            self.logger.error(f"Error in Polymarket collection: {str(e)}")
            return {}
            
        self.logger.info(f"Collected Polymarket sentiment for {len(market_sentiment)} stocks")
        return market_sentiment
    
    async def _collect_stock_market_data(self, session: aiohttp.ClientSession, stock: str) -> Dict[str, Any]:
        """Collect prediction market data for a specific stock."""
        try:
            search_queries = [
                f"{stock}",
                f"{stock} stock",
                f"{stock} price",
                f"{stock} earnings"
            ]
            
            all_markets = []
            
            for query in search_queries:
                try:
                    markets = await self._search_markets(session, query)
                    all_markets.extend(markets)
                except Exception as e:
                    self.logger.warning(f"Error searching markets for '{query}': {str(e)}")
                    continue
            
            market_analysis = self._analyze_market_sentiment(all_markets, stock)
            
            return {
                'stock': stock,
                'market_count': len(all_markets),
                'sentiment_score': market_analysis['sentiment_score'],
                'confidence': market_analysis['confidence'],
                'prediction_probability': market_analysis['prediction_probability'],
                'volume_weighted_sentiment': market_analysis['volume_weighted_sentiment'],
                'top_markets': market_analysis['top_markets'][:3],
                'collected_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting market data for {stock}: {str(e)}")
            return self._get_default_market_data(stock)
    
    async def _search_markets(self, session: aiohttp.ClientSession, query: str) -> List[Dict[str, Any]]:
        """Search for prediction markets related to a query."""
        try:
            return []
            
        except Exception as e:
            self.logger.warning(f"Error searching markets for '{query}': {str(e)}")
            return []
    
    def _analyze_market_sentiment(self, markets: List[Dict[str, Any]], stock: str) -> Dict[str, Any]:
        """Analyze sentiment from prediction markets."""
        if not markets:
            return {
                'sentiment_score': 0.0,
                'confidence': 0.0,
                'prediction_probability': 0.5,
                'volume_weighted_sentiment': 0.0,
                'top_markets': []
            }
        
        total_volume = 0
        weighted_sentiment = 0
        probabilities = []
        
        for market in markets:
            volume = market.get('volume', 0)
            probability = market.get('probability', 0.5)
            
            sentiment = (probability - 0.5) * 2
            
            weighted_sentiment += sentiment * volume
            total_volume += volume
            probabilities.append(probability)
        
        if total_volume > 0:
            volume_weighted_sentiment = weighted_sentiment / total_volume
        else:
            volume_weighted_sentiment = 0.0
            
        avg_probability = sum(probabilities) / len(probabilities) if probabilities else 0.5
        sentiment_score = (avg_probability - 0.5) * 2
        confidence = abs(sentiment_score)
        
        markets.sort(key=lambda x: x.get('volume', 0), reverse=True)
        
        return {
            'sentiment_score': sentiment_score,
            'confidence': confidence,
            'prediction_probability': avg_probability,
            'volume_weighted_sentiment': volume_weighted_sentiment,
            'top_markets': markets[:5]
        }
    
    def _get_default_market_data(self, stock: str) -> Dict[str, Any]:
        """Return default market data when collection fails."""
        return {
            'stock': stock,
            'market_count': 0,
            'sentiment_score': 0.0,
            'confidence': 0.0,
            'prediction_probability': 0.5,
            'volume_weighted_sentiment': 0.0,
            'top_markets': [],
            'collected_at': datetime.now().isoformat()
        }
    
    
    
    async def get_trending_markets(self) -> List[Dict[str, Any]]:
        """Get trending prediction markets that might indicate market sentiment."""
        try:
            if not self.enabled:
                return []
            return []
            
        except Exception as e:
            self.logger.error(f"Error fetching trending markets: {str(e)}")
            return []
