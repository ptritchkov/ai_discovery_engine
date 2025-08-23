"""Comprehensive sentiment analyzer combining multiple sources."""

import asyncio
from datetime import datetime
from typing import List, Dict, Any
import logging
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np

class SentimentAnalyzer:
    """Analyzes sentiment from multiple data sources."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
    async def analyze_comprehensive_sentiment(
        self, 
        news_data: List[Dict[str, Any]], 
        twitter_data: Dict[str, Any], 
        polymarket_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform comprehensive sentiment analysis across all data sources.
        
        Args:
            news_data: List of news articles
            twitter_data: Twitter sentiment data by stock
            polymarket_data: Polymarket prediction data by stock
            
        Returns:
            Comprehensive sentiment analysis results
        """
        self.logger.info("Performing comprehensive sentiment analysis...")
        
        try:
            news_sentiment = await self._analyze_news_sentiment(news_data)
            
            social_sentiment = await self._analyze_social_sentiment(twitter_data)
            
            market_sentiment = await self._analyze_prediction_market_sentiment(polymarket_data)
            
            combined_sentiment = await self._combine_sentiment_sources(
                news_sentiment, social_sentiment, market_sentiment
            )
            
            overall_sentiment = await self._calculate_overall_sentiment(combined_sentiment)
            
            return {
                'news_sentiment': news_sentiment,
                'social_sentiment': social_sentiment,
                'market_sentiment': market_sentiment,
                'combined_sentiment': combined_sentiment,
                'overall_sentiment': overall_sentiment,
                'sentiment_signals': await self._generate_sentiment_signals(combined_sentiment),
                'analyzed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive sentiment analysis: {str(e)}")
            return self._get_default_sentiment_analysis()
    
    async def _analyze_news_sentiment(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment from news articles."""
        try:
            if not news_data:
                return {'overall_score': 0.0, 'article_sentiments': [], 'top_positive': [], 'top_negative': []}
            
            article_sentiments = []
            
            for article in news_data:
                title = article.get('title', '')
                description = article.get('description', '')
                content = article.get('content', '')
                
                full_text = f"{title} {description} {content}"
                
                textblob_sentiment = self._analyze_with_textblob(full_text)
                vader_sentiment = self._analyze_with_vader(full_text)
                
                combined_score = (textblob_sentiment['polarity'] + vader_sentiment['compound']) / 2
                
                article_sentiment = {
                    'title': title,
                    'url': article.get('url', ''),
                    'source': article.get('source', ''),
                    'published_at': article.get('published_at', ''),
                    'textblob_score': textblob_sentiment['polarity'],
                    'vader_score': vader_sentiment['compound'],
                    'combined_score': combined_score,
                    'sentiment_label': self._get_sentiment_label(combined_score),
                    'relevance_score': article.get('relevance_score', 0.5),
                    'weighted_score': combined_score * article.get('relevance_score', 0.5)
                }
                
                article_sentiments.append(article_sentiment)
            
            if article_sentiments:
                overall_score = sum(a['weighted_score'] for a in article_sentiments) / len(article_sentiments)
            else:
                overall_score = 0.0
            
            article_sentiments.sort(key=lambda x: x['combined_score'], reverse=True)
            
            return {
                'overall_score': overall_score,
                'article_count': len(article_sentiments),
                'article_sentiments': article_sentiments,
                'top_positive': [a for a in article_sentiments if a['combined_score'] > 0.1][:5],
                'top_negative': [a for a in article_sentiments if a['combined_score'] < -0.1][:5],
                'sentiment_distribution': self._calculate_sentiment_distribution(article_sentiments)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {str(e)}")
            return {'overall_score': 0.0, 'article_sentiments': [], 'top_positive': [], 'top_negative': []}
    
    async def _analyze_social_sentiment(self, twitter_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment from social media data."""
        try:
            if not twitter_data:
                return {'overall_score': 0.0, 'stock_sentiments': {}}
            
            stock_sentiments = {}
            overall_scores = []
            
            for stock, data in twitter_data.items():
                sentiment_score = data.get('sentiment_score', 0.0)
                confidence = data.get('confidence', 0.0)
                engagement_score = data.get('engagement_score', 0.0)
                tweet_count = data.get('tweet_count', 0)
                
                weighted_score = sentiment_score * confidence * min(engagement_score / 1000, 1.0)
                
                stock_sentiments[stock] = {
                    'sentiment_score': sentiment_score,
                    'confidence': confidence,
                    'engagement_score': engagement_score,
                    'tweet_count': tweet_count,
                    'weighted_score': weighted_score,
                    'sentiment_label': data.get('sentiment_label', 'neutral'),
                    'top_tweets': data.get('top_tweets', [])
                }
                
                if tweet_count > 10:  # Only include stocks with sufficient data
                    overall_scores.append(weighted_score)
            
            overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
            
            return {
                'overall_score': overall_score,
                'stock_count': len(stock_sentiments),
                'stock_sentiments': stock_sentiments,
                'most_positive': max(stock_sentiments.items(), key=lambda x: x[1]['weighted_score'])[0] if stock_sentiments else None,
                'most_negative': min(stock_sentiments.items(), key=lambda x: x[1]['weighted_score'])[0] if stock_sentiments else None
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing social sentiment: {str(e)}")
            return {'overall_score': 0.0, 'stock_sentiments': {}}
    
    async def _analyze_prediction_market_sentiment(self, polymarket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment from prediction markets."""
        try:
            if not polymarket_data:
                return {'overall_score': 0.0, 'stock_sentiments': {}}
            
            stock_sentiments = {}
            overall_scores = []
            
            for stock, data in polymarket_data.items():
                sentiment_score = data.get('sentiment_score', 0.0)
                confidence = data.get('confidence', 0.0)
                volume_weighted_sentiment = data.get('volume_weighted_sentiment', 0.0)
                market_count = data.get('market_count', 0)
                
                primary_score = volume_weighted_sentiment if market_count > 0 else sentiment_score
                
                stock_sentiments[stock] = {
                    'sentiment_score': sentiment_score,
                    'volume_weighted_sentiment': volume_weighted_sentiment,
                    'confidence': confidence,
                    'market_count': market_count,
                    'primary_score': primary_score,
                    'prediction_probability': data.get('prediction_probability', 0.5),
                    'top_markets': data.get('top_markets', [])
                }
                
                if market_count > 0:  # Only include stocks with market data
                    overall_scores.append(primary_score)
            
            overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else 0.0
            
            return {
                'overall_score': overall_score,
                'stock_count': len(stock_sentiments),
                'stock_sentiments': stock_sentiments,
                'highest_confidence': max(stock_sentiments.items(), key=lambda x: x[1]['confidence'])[0] if stock_sentiments else None,
                'most_bullish': max(stock_sentiments.items(), key=lambda x: x[1]['primary_score'])[0] if stock_sentiments else None,
                'most_bearish': min(stock_sentiments.items(), key=lambda x: x[1]['primary_score'])[0] if stock_sentiments else None
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing prediction market sentiment: {str(e)}")
            return {'overall_score': 0.0, 'stock_sentiments': {}}
    
    async def _combine_sentiment_sources(
        self, 
        news_sentiment: Dict[str, Any], 
        social_sentiment: Dict[str, Any], 
        market_sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Combine sentiment from all sources."""
        try:
            all_stocks = set()
            
            if 'stock_sentiments' in social_sentiment:
                all_stocks.update(social_sentiment['stock_sentiments'].keys())
            if 'stock_sentiments' in market_sentiment:
                all_stocks.update(market_sentiment['stock_sentiments'].keys())
            
            combined_stock_sentiments = {}
            
            for stock in all_stocks:
                social_data = social_sentiment.get('stock_sentiments', {}).get(stock, {})
                market_data = market_sentiment.get('stock_sentiments', {}).get(stock, {})
                
                social_score = social_data.get('weighted_score', 0.0)
                market_score = market_data.get('primary_score', 0.0)
                
                social_weight = 0.4
                market_weight = 0.6
                
                combined_score = (social_score * social_weight + market_score * market_weight)
                
                confidence = 0.0
                if social_data:
                    confidence += social_data.get('confidence', 0.0) * 0.4
                if market_data:
                    confidence += market_data.get('confidence', 0.0) * 0.6
                
                combined_stock_sentiments[stock] = {
                    'combined_score': combined_score,
                    'confidence': confidence,
                    'social_sentiment': social_score,
                    'market_sentiment': market_score,
                    'sentiment_label': self._get_sentiment_label(combined_score),
                    'data_sources': {
                        'social': bool(social_data),
                        'market': bool(market_data)
                    }
                }
            
            news_score = news_sentiment.get('overall_score', 0.0)
            social_score = social_sentiment.get('overall_score', 0.0)
            market_score = market_sentiment.get('overall_score', 0.0)
            
            overall_combined = (news_score * 0.3 + social_score * 0.3 + market_score * 0.4)
            
            return {
                'stock_sentiments': combined_stock_sentiments,
                'overall_score': overall_combined,
                'source_scores': {
                    'news': news_score,
                    'social': social_score,
                    'market': market_score
                },
                'top_bullish': sorted(combined_stock_sentiments.items(), key=lambda x: x[1]['combined_score'], reverse=True)[:5],
                'top_bearish': sorted(combined_stock_sentiments.items(), key=lambda x: x[1]['combined_score'])[:5]
            }
            
        except Exception as e:
            self.logger.error(f"Error combining sentiment sources: {str(e)}")
            return {'stock_sentiments': {}, 'overall_score': 0.0}
    
    async def _calculate_overall_sentiment(self, combined_sentiment: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall market sentiment."""
        try:
            overall_score = combined_sentiment.get('overall_score', 0.0)
            stock_sentiments = combined_sentiment.get('stock_sentiments', {})
            
            positive_count = sum(1 for s in stock_sentiments.values() if s['combined_score'] > 0.1)
            negative_count = sum(1 for s in stock_sentiments.values() if s['combined_score'] < -0.1)
            neutral_count = len(stock_sentiments) - positive_count - negative_count
            
            if overall_score > 0.2:
                market_mood = 'very_bullish'
            elif overall_score > 0.05:
                market_mood = 'bullish'
            elif overall_score > -0.05:
                market_mood = 'neutral'
            elif overall_score > -0.2:
                market_mood = 'bearish'
            else:
                market_mood = 'very_bearish'
            
            return {
                'overall_score': overall_score,
                'market_mood': market_mood,
                'sentiment_distribution': {
                    'positive': positive_count,
                    'negative': negative_count,
                    'neutral': neutral_count
                },
                'sentiment_strength': abs(overall_score),
                'market_consensus': positive_count / max(len(stock_sentiments), 1)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating overall sentiment: {str(e)}")
            return {'overall_score': 0.0, 'market_mood': 'neutral'}
    
    async def _generate_sentiment_signals(self, combined_sentiment: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable sentiment signals."""
        try:
            signals = []
            stock_sentiments = combined_sentiment.get('stock_sentiments', {})
            
            for stock, data in stock_sentiments.items():
                score = data['combined_score']
                confidence = data['confidence']
                
                if score > 0.3 and confidence > 0.6:
                    signals.append({
                        'stock': stock,
                        'signal': 'strong_buy_sentiment',
                        'strength': score * confidence,
                        'description': f"Strong positive sentiment for {stock}"
                    })
                elif score < -0.3 and confidence > 0.6:
                    signals.append({
                        'stock': stock,
                        'signal': 'strong_sell_sentiment',
                        'strength': abs(score) * confidence,
                        'description': f"Strong negative sentiment for {stock}"
                    })
                elif abs(score) > 0.15 and confidence > 0.4:
                    signal_type = 'moderate_buy_sentiment' if score > 0 else 'moderate_sell_sentiment'
                    signals.append({
                        'stock': stock,
                        'signal': signal_type,
                        'strength': abs(score) * confidence,
                        'description': f"Moderate sentiment signal for {stock}"
                    })
            
            signals.sort(key=lambda x: x['strength'], reverse=True)
            
            return signals[:10]  # Return top 10 signals
            
        except Exception as e:
            self.logger.error(f"Error generating sentiment signals: {str(e)}")
            return []
    
    def _analyze_with_textblob(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using TextBlob."""
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception:
            return {'polarity': 0.0, 'subjectivity': 0.0}
    
    def _analyze_with_vader(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using VADER."""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return scores
        except Exception:
            return {'compound': 0.0, 'pos': 0.0, 'neu': 1.0, 'neg': 0.0}
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label."""
        if score > 0.1:
            return 'positive'
        elif score < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def _calculate_sentiment_distribution(self, sentiments: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of sentiment labels."""
        distribution = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for sentiment in sentiments:
            label = sentiment.get('sentiment_label', 'neutral')
            distribution[label] = distribution.get(label, 0) + 1
        
        return distribution
    
    def _get_default_sentiment_analysis(self) -> Dict[str, Any]:
        """Return default sentiment analysis when processing fails."""
        return {
            'news_sentiment': {'overall_score': 0.0, 'article_sentiments': []},
            'social_sentiment': {'overall_score': 0.0, 'stock_sentiments': {}},
            'market_sentiment': {'overall_score': 0.0, 'stock_sentiments': {}},
            'combined_sentiment': {'stock_sentiments': {}, 'overall_score': 0.0},
            'overall_sentiment': {'overall_score': 0.0, 'market_mood': 'neutral'},
            'sentiment_signals': [],
            'analyzed_at': datetime.now().isoformat(),
            'error': 'Failed to analyze sentiment'
        }
