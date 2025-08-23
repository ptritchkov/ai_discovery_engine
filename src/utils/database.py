"""In-memory database for storing analysis results."""

import json
from datetime import datetime
from typing import Dict, List, Any
import logging

class Database:
    """Simple in-memory database for storing analysis results."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analysis_results = []
        self.news_cache = {}
        self.sentiment_cache = {}
        self.market_data_cache = {}
        
    async def store_analysis_results(self, results: Dict[str, Any]) -> None:
        """Store analysis results in memory."""
        results["stored_at"] = datetime.now().isoformat()
        self.analysis_results.append(results)
        self.logger.info(f"Stored analysis results for {results.get('timeframe', 'unknown')} timeframe")
        
    async def get_recent_analysis(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent analysis results."""
        return sorted(self.analysis_results, key=lambda x: x.get("stored_at", ""), reverse=True)[:limit]
        
    async def cache_news_data(self, key: str, data: List[Dict[str, Any]]) -> None:
        """Cache news data to avoid redundant API calls."""
        self.news_cache[key] = {
            "data": data,
            "cached_at": datetime.now().isoformat()
        }
        
    async def get_cached_news(self, key: str) -> List[Dict[str, Any]]:
        """Get cached news data if available and recent."""
        cached = self.news_cache.get(key)
        if cached:
            cached_time = datetime.fromisoformat(cached["cached_at"])
            if (datetime.now() - cached_time).seconds < 3600:
                return cached["data"]
        return []
        
    async def cache_sentiment_data(self, key: str, data: Dict[str, Any]) -> None:
        """Cache sentiment analysis data."""
        self.sentiment_cache[key] = {
            "data": data,
            "cached_at": datetime.now().isoformat()
        }
        
    async def get_cached_sentiment(self, key: str) -> Dict[str, Any]:
        """Get cached sentiment data if available and recent."""
        cached = self.sentiment_cache.get(key)
        if cached:
            cached_time = datetime.fromisoformat(cached["cached_at"])
            if (datetime.now() - cached_time).seconds < 1800:  # 30 minutes
                return cached["data"]
        return {}
        
    def export_results_to_json(self, filename: str = None) -> str:
        """Export all analysis results to JSON file."""
        if not filename:
            filename = f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        with open(filename, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
            
        self.logger.info(f"Exported {len(self.analysis_results)} analysis results to {filename}")
        return filename
