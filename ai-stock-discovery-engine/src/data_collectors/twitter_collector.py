"""Twitter data collector for sentiment analysis."""

import asyncio
import tweepy
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
import re

class TwitterCollector:
    """Collects Twitter data for sentiment analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.bearer_token = os.getenv("TWITTER_BEARER_TOKEN")
        
        if self.bearer_token:
            self.client = tweepy.Client(bearer_token=self.bearer_token)
        else:
            self.client = None
            self.logger.warning("Twitter Bearer Token not found. Twitter data collection will be limited.")
    
    async def collect_sentiment_data(self, stocks: List[str]) -> Dict[str, Any]:
        """
        Collect Twitter sentiment data for given stocks.
        
        Args:
            stocks: List of stock symbols to analyze
            
        Returns:
            Dictionary containing sentiment data for each stock
        """
        self.logger.info(f"Collecting Twitter sentiment for {len(stocks)} stocks...")
        
        sentiment_data = {}
        
        if not self.client:
            self.logger.warning("Twitter client not available. Returning mock data.")
            return self._generate_mock_sentiment_data(stocks)
        
        try:
            for stock in stocks[:10]:  # Limit to avoid rate limits
                try:
                    stock_sentiment = await self._collect_stock_sentiment(stock)
                    sentiment_data[stock] = stock_sentiment
                    
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.warning(f"Error collecting sentiment for {stock}: {str(e)}")
                    sentiment_data[stock] = self._get_default_sentiment()
                    
        except Exception as e:
            self.logger.error(f"Error in Twitter sentiment collection: {str(e)}")
            return self._generate_mock_sentiment_data(stocks)
            
        self.logger.info(f"Collected Twitter sentiment for {len(sentiment_data)} stocks")
        return sentiment_data
    
    async def _collect_stock_sentiment(self, stock: str) -> Dict[str, Any]:
        """Collect sentiment data for a specific stock."""
        try:
            queries = [
                f"${stock}",
                f"{stock} stock",
                f"{stock} earnings",
                f"{stock} price"
            ]
            
            all_tweets = []
            
            for query in queries:
                try:
                    tweets = tweepy.Paginator(
                        self.client.search_recent_tweets,
                        query=query,
                        max_results=100,
                        tweet_fields=['created_at', 'public_metrics', 'context_annotations']
                    ).flatten(limit=100)
                    
                    for tweet in tweets:
                        all_tweets.append({
                            'id': tweet.id,
                            'text': tweet.text,
                            'created_at': tweet.created_at,
                            'retweet_count': tweet.public_metrics['retweet_count'],
                            'like_count': tweet.public_metrics['like_count'],
                            'reply_count': tweet.public_metrics['reply_count'],
                            'quote_count': tweet.public_metrics['quote_count']
                        })
                        
                except Exception as e:
                    self.logger.warning(f"Error searching for query '{query}': {str(e)}")
                    continue
            
            sentiment_analysis = self._analyze_tweets_sentiment(all_tweets)
            
            return {
                'stock': stock,
                'tweet_count': len(all_tweets),
                'sentiment_score': sentiment_analysis['sentiment_score'],
                'sentiment_label': sentiment_analysis['sentiment_label'],
                'confidence': sentiment_analysis['confidence'],
                'engagement_score': sentiment_analysis['engagement_score'],
                'top_tweets': sentiment_analysis['top_tweets'][:5],
                'collected_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error collecting sentiment for {stock}: {str(e)}")
            return self._get_default_sentiment()
    
    def _analyze_tweets_sentiment(self, tweets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment of collected tweets."""
        if not tweets:
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.0,
                'engagement_score': 0.0,
                'top_tweets': []
            }
        
        positive_keywords = ['bullish', 'buy', 'moon', 'rocket', 'pump', 'up', 'gain', 'profit', 'bull']
        negative_keywords = ['bearish', 'sell', 'dump', 'crash', 'down', 'loss', 'bear', 'short']
        
        sentiment_scores = []
        engagement_scores = []
        
        for tweet in tweets:
            text = tweet['text'].lower()
            
            positive_count = sum(1 for word in positive_keywords if word in text)
            negative_count = sum(1 for word in negative_keywords if word in text)
            
            if positive_count + negative_count > 0:
                sentiment_score = (positive_count - negative_count) / (positive_count + negative_count)
            else:
                sentiment_score = 0.0
                
            sentiment_scores.append(sentiment_score)
            
            engagement = (
                tweet['retweet_count'] * 3 +
                tweet['like_count'] * 1 +
                tweet['reply_count'] * 2 +
                tweet['quote_count'] * 2
            )
            engagement_scores.append(engagement)
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
        avg_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0.0
        
        if avg_sentiment > 0.1:
            sentiment_label = 'positive'
        elif avg_sentiment < -0.1:
            sentiment_label = 'negative'
        else:
            sentiment_label = 'neutral'
        
        tweets_with_engagement = list(zip(tweets, engagement_scores))
        tweets_with_engagement.sort(key=lambda x: x[1], reverse=True)
        top_tweets = [tweet for tweet, _ in tweets_with_engagement[:5]]
        
        return {
            'sentiment_score': avg_sentiment,
            'sentiment_label': sentiment_label,
            'confidence': abs(avg_sentiment),
            'engagement_score': avg_engagement,
            'top_tweets': top_tweets
        }
    
    def _get_default_sentiment(self) -> Dict[str, Any]:
        """Return default sentiment data when collection fails."""
        return {
            'stock': 'unknown',
            'tweet_count': 0,
            'sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'confidence': 0.0,
            'engagement_score': 0.0,
            'top_tweets': [],
            'collected_at': datetime.now().isoformat()
        }
    
    def _generate_mock_sentiment_data(self, stocks: List[str]) -> Dict[str, Any]:
        """Generate mock sentiment data when Twitter API is not available."""
        import random
        
        sentiment_data = {}
        
        for stock in stocks:
            sentiment_score = random.uniform(-0.5, 0.5)
            
            if sentiment_score > 0.1:
                sentiment_label = 'positive'
            elif sentiment_score < -0.1:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            sentiment_data[stock] = {
                'stock': stock,
                'tweet_count': random.randint(50, 500),
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'confidence': abs(sentiment_score),
                'engagement_score': random.uniform(100, 1000),
                'top_tweets': [
                    {'text': f"Mock tweet about ${stock} looking good!", 'like_count': random.randint(10, 100)},
                    {'text': f"Interesting movement in ${stock} today", 'like_count': random.randint(5, 50)}
                ],
                'collected_at': datetime.now().isoformat(),
                'mock_data': True
            }
        
        return sentiment_data
