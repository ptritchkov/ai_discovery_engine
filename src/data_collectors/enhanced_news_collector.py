"""Enhanced news collector that gathers real financial news from multiple sources."""

import asyncio
import aiohttp
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
import re
from bs4 import BeautifulSoup

class EnhancedNewsCollector:
    """Collects real financial news from multiple RSS feeds and web sources."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def collect_comprehensive_news(self, timeframe: str = "daily") -> List[Dict[str, Any]]:
        """
        Collect comprehensive financial news from multiple sources.
        
        Args:
            timeframe: "daily" or "weekly"
            
        Returns:
            List of news articles with metadata
        """
        self.logger.info(f"Collecting comprehensive {timeframe} news...")
        
        if timeframe == "daily":
            hours_back = 24
        else:  # weekly
            hours_back = 168
            
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        news_sources = [
            {
                'name': 'Yahoo Finance',
                'url': 'https://finance.yahoo.com/rss/topstories',
                'type': 'rss'
            },
            {
                'name': 'MarketWatch Top Stories',
                'url': 'https://feeds.marketwatch.com/marketwatch/topstories/',
                'type': 'rss'
            },
            {
                'name': 'Reuters Business',
                'url': 'https://feeds.reuters.com/reuters/businessNews',
                'type': 'rss'
            },
            {
                'name': 'CNN Business',
                'url': 'http://rss.cnn.com/rss/money_latest.rss',
                'type': 'rss'
            },
            {
                'name': 'Bloomberg Markets',
                'url': 'https://feeds.bloomberg.com/markets/news.rss',
                'type': 'rss'
            },
            {
                'name': 'Financial Times',
                'url': 'https://www.ft.com/rss/home/us',
                'type': 'rss'
            },
            {
                'name': 'Seeking Alpha',
                'url': 'https://seekingalpha.com/feed.xml',
                'type': 'rss'
            }
        ]
        
        all_articles = []
        
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={'User-Agent': 'Mozilla/5.0 (compatible; NewsBot/1.0)'}
        ) as session:
            
            tasks = []
            for source in news_sources:
                task = self._fetch_from_source(session, source, cutoff_time)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self.logger.warning(f"Error fetching from {news_sources[i]['name']}: {str(result)}")
                else:
                    if isinstance(result, list):
                        all_articles.extend(result)
        
        # Deduplicate and filter
        unique_articles = self._deduplicate_articles(all_articles)
        filtered_articles = self._filter_financial_relevance(unique_articles)
        
        self.logger.info(f"Collected {len(filtered_articles)} relevant financial articles")
        return filtered_articles[:100]  # Return top 100 most relevant
    
    async def _fetch_from_source(self, session: aiohttp.ClientSession, source: Dict[str, str], cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Fetch articles from a single news source."""
        articles = []
        
        try:
            async with session.get(source['url']) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    if source['type'] == 'rss':
                        articles = self._parse_rss_feed(content, source['name'], cutoff_time)
                    
        except Exception as e:
            self.logger.warning(f"Error fetching from {source['name']}: {str(e)}")
        
        return articles
    
    def _parse_rss_feed(self, content: str, source_name: str, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Parse RSS feed content."""
        articles = []
        
        try:
            feed = feedparser.parse(content)
            
            for entry in feed.entries:
                try:
                    # Parse publication date
                    pub_date = None
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        pub_date = datetime(*entry.updated_parsed[:6])
                    
                    # Skip old articles
                    if pub_date and pub_date < cutoff_time:
                        continue
                    
                    # Extract content
                    title = getattr(entry, 'title', '')
                    description = getattr(entry, 'description', '') or getattr(entry, 'summary', '')
                    link = getattr(entry, 'link', '')
                    
                    # Clean HTML from description
                    if description:
                        description = BeautifulSoup(description, 'html.parser').get_text()
                    
                    article = {
                        'title': title,
                        'description': description,
                        'content': description,  # Use description as content for RSS
                        'url': link,
                        'source': source_name,
                        'published_at': pub_date.isoformat() if pub_date else datetime.now().isoformat(),
                        'keyword': 'financial_news',
                        'relevance_score': self._calculate_financial_relevance(title, description)
                    }
                    
                    if article['relevance_score'] > 0.3:  # Only include relevant articles
                        articles.append(article)
                        
                except Exception as e:
                    self.logger.debug(f"Error parsing RSS entry: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Error parsing RSS feed from {source_name}: {str(e)}")
        
        return articles
    
    def _calculate_financial_relevance(self, title: str, description: str) -> float:
        """Calculate how relevant an article is to financial markets."""
        text = f"{title} {description}".lower()
        
        # Financial keywords with weights
        financial_keywords = {
            # High relevance
            'earnings': 0.3, 'revenue': 0.3, 'profit': 0.3, 'loss': 0.3,
            'stock': 0.25, 'shares': 0.25, 'market': 0.2, 'trading': 0.25,
            'ipo': 0.3, 'merger': 0.3, 'acquisition': 0.3, 'buyout': 0.3,
            'dividend': 0.25, 'split': 0.25, 'buyback': 0.25,
            
            # Medium relevance
            'investment': 0.2, 'investor': 0.2, 'fund': 0.15, 'portfolio': 0.2,
            'analyst': 0.15, 'rating': 0.15, 'upgrade': 0.2, 'downgrade': 0.2,
            'forecast': 0.15, 'outlook': 0.15, 'guidance': 0.2,
            
            # Economic indicators
            'inflation': 0.2, 'gdp': 0.2, 'unemployment': 0.15, 'fed': 0.2,
            'interest rate': 0.25, 'federal reserve': 0.2, 'economy': 0.15,
            
            # Sector specific
            'technology': 0.1, 'healthcare': 0.1, 'energy': 0.1, 'financial': 0.1,
            'retail': 0.1, 'manufacturing': 0.1, 'automotive': 0.1,
            
            # Company actions
            'ceo': 0.15, 'executive': 0.1, 'leadership': 0.1, 'partnership': 0.15,
            'contract': 0.15, 'deal': 0.15, 'agreement': 0.15, 'launch': 0.1
        }
        
        score = 0.0
        word_count = len(text.split())
        
        for keyword, weight in financial_keywords.items():
            if keyword in text:
                # Boost score based on frequency and position
                frequency = text.count(keyword)
                if keyword in title.lower():
                    score += weight * 1.5 * frequency  # Title mentions are more important
                else:
                    score += weight * frequency
        
        # Normalize by text length
        if word_count > 0:
            score = min(score / (word_count / 50), 1.0)  # Normalize to 0-1 scale
        
        return score
    
    def _deduplicate_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate articles based on title similarity."""
        unique_articles = []
        seen_titles = set()
        
        for article in articles:
            title = article.get('title', '').lower().strip()
            
            # Create a simplified version for comparison
            simplified_title = re.sub(r'[^\w\s]', '', title)
            simplified_title = ' '.join(simplified_title.split())
            
            if len(simplified_title) > 10 and simplified_title not in seen_titles:
                seen_titles.add(simplified_title)
                unique_articles.append(article)
        
        return unique_articles
    
    def _filter_financial_relevance(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter articles by financial relevance and sort by score."""
        # Filter by minimum relevance score
        relevant_articles = [
            article for article in articles 
            if article.get('relevance_score', 0) > 0.3
        ]
        
        # Sort by relevance score (highest first)
        relevant_articles.sort(
            key=lambda x: x.get('relevance_score', 0), 
            reverse=True
        )
        
        return relevant_articles
