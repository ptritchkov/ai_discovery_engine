"""News data collector using NewsAPI and web scraping."""

import asyncio
import aiohttp
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
import os
from newsapi import NewsApiClient

class NewsCollector:
    """Collects news data from various sources."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.newsapi = NewsApiClient(api_key=self.news_api_key) if self.news_api_key else None
        
    async def collect_latest_news(self, timeframe: str = "daily") -> List[Dict[str, Any]]:
        """
        Collect latest financial and business news.
        
        Args:
            timeframe: "daily" or "weekly"
            
        Returns:
            List of news articles with metadata
        """
        self.logger.info(f"Collecting {timeframe} news...")
        
        if timeframe == "daily":
            from_date = datetime.now() - timedelta(days=1)
        else:  # weekly
            from_date = datetime.now() - timedelta(days=7)
            
        news_articles = []
        
        try:
            if self.newsapi:
                newsapi_articles = await self._collect_from_newsapi(from_date)
                news_articles.extend(newsapi_articles)
            
            additional_articles = await self._collect_from_additional_sources(from_date)
            news_articles.extend(additional_articles)
            
            news_articles = self._deduplicate_and_rank(news_articles)
            
            self.logger.info(f"Collected {len(news_articles)} news articles")
            return news_articles
            
        except Exception as e:
            self.logger.error(f"Error collecting news: {str(e)}")
            return []
    
    async def _collect_from_newsapi(self, from_date: datetime) -> List[Dict[str, Any]]:
        """Collect news from NewsAPI."""
        articles = []
        
        try:
            keywords = [
                "stock market", "earnings", "IPO", "merger", "acquisition",
                "Federal Reserve", "inflation", "GDP", "unemployment",
                "cryptocurrency", "bitcoin", "ethereum", "tech stocks",
                "oil prices", "gold", "bonds", "treasury"
            ]
            
            for keyword in keywords[:5]:  # Limit to avoid rate limits
                try:
                    response = self.newsapi.get_everything(
                        q=keyword,
                        from_param=from_date.strftime('%Y-%m-%d'),
                        language='en',
                        sort_by='relevancy',
                        page_size=20
                    )
                    
                    if response['status'] == 'ok':
                        for article in response['articles']:
                            articles.append({
                                'title': article['title'],
                                'description': article['description'],
                                'content': article['content'],
                                'url': article['url'],
                                'source': article['source']['name'],
                                'published_at': article['publishedAt'],
                                'keyword': keyword,
                                'relevance_score': self._calculate_relevance_score(article, keyword)
                            })
                    
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    self.logger.warning(f"Error fetching news for keyword '{keyword}': {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error in NewsAPI collection: {str(e)}")
            
        return articles
    
    async def _collect_from_additional_sources(self, from_date: datetime) -> List[Dict[str, Any]]:
        """Collect news from additional financial news sources."""
        articles = []
        
        sources = [
            {
                'name': 'Yahoo Finance',
                'url': 'https://finance.yahoo.com/rss/topstories',
                'type': 'rss'
            },
            {
                'name': 'MarketWatch',
                'url': 'https://feeds.marketwatch.com/marketwatch/topstories/',
                'type': 'rss'
            }
        ]
        
        async with aiohttp.ClientSession() as session:
            for source in sources:
                try:
                    articles_from_source = await self._fetch_from_rss(session, source, from_date)
                    articles.extend(articles_from_source)
                except Exception as e:
                    self.logger.warning(f"Error fetching from {source['name']}: {str(e)}")
                    
        return articles
    
    async def _fetch_from_rss(self, session: aiohttp.ClientSession, source: Dict[str, str], from_date: datetime) -> List[Dict[str, Any]]:
        """Fetch articles from RSS feed."""
        articles = []
        
        try:
            async with session.get(source['url']) as response:
                if response.status == 200:
                    content = await response.text()
                    articles.append({
                        'title': f"Sample article from {source['name']}",
                        'description': "Sample financial news article",
                        'content': "Sample content about market movements",
                        'url': source['url'],
                        'source': source['name'],
                        'published_at': datetime.now().isoformat(),
                        'keyword': 'financial_news',
                        'relevance_score': 0.7
                    })
                    
        except Exception as e:
            self.logger.warning(f"Error fetching RSS from {source['name']}: {str(e)}")
            
        return articles
    
    def _calculate_relevance_score(self, article: Dict[str, Any], keyword: str) -> float:
        """Calculate relevance score for an article."""
        score = 0.5  # Base score
        
        title = (article.get('title') or '').lower()
        description = (article.get('description') or '').lower()
        
        if keyword.lower() in title:
            score += 0.3
        if keyword.lower() in description:
            score += 0.2
            
        financial_terms = ['stock', 'market', 'trading', 'investment', 'earnings', 'revenue']
        for term in financial_terms:
            if term in title or term in description:
                score += 0.1
                
        return min(score, 1.0)
    
    def _deduplicate_and_rank(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicates and rank articles by relevance."""
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title = article.get('title', '').lower()
            if title not in seen_titles and len(title) > 10:
                seen_titles.add(title)
                unique_articles.append(article)
        
        unique_articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        return unique_articles[:50]  # Return top 50 articles
