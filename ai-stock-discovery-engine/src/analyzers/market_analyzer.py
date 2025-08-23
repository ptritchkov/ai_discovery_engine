"""Market analyzer for understanding price movements and reactions to news."""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
from scipy import stats
import yfinance as yf

class MarketAnalyzer:
    """Analyzes market reactions and patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def analyze_market_reactions(
        self, 
        news_data: List[Dict[str, Any]], 
        market_data: Dict[str, Any], 
        sentiment_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze how markets have reacted to news and sentiment.
        
        Args:
            news_data: News articles
            market_data: Stock market data
            sentiment_analysis: Sentiment analysis results
            
        Returns:
            Market reaction analysis
        """
        self.logger.info("Analyzing market reactions to news and sentiment...")
        
        try:
            news_correlations = await self._analyze_news_price_correlations(news_data, market_data)
            
            sentiment_correlations = await self._analyze_sentiment_price_correlations(
                sentiment_analysis, market_data
            )
            
            inefficiencies = await self._identify_market_inefficiencies(
                news_data, market_data, sentiment_analysis
            )
            
            volume_analysis = await self._analyze_volume_patterns(market_data)
            
            momentum_analysis = await self._analyze_market_momentum(market_data)
            
            reversal_patterns = await self._identify_reversal_patterns(market_data)
            
            return {
                'news_correlations': news_correlations,
                'sentiment_correlations': sentiment_correlations,
                'market_inefficiencies': inefficiencies,
                'volume_analysis': volume_analysis,
                'momentum_analysis': momentum_analysis,
                'reversal_patterns': reversal_patterns,
                'market_signals': await self._generate_market_signals(market_data),
                'analyzed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in market reaction analysis: {str(e)}")
            return self._get_default_market_analysis()
    
    async def _analyze_news_price_correlations(
        self, 
        news_data: List[Dict[str, Any]], 
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze correlations between news and price movements."""
        try:
            correlations = {}
            
            news_by_date = {}
            for article in news_data:
                pub_date = article.get('published_at', '')
                if pub_date:
                    try:
                        date_key = datetime.fromisoformat(pub_date.replace('Z', '+00:00')).date()
                        if date_key not in news_by_date:
                            news_by_date[date_key] = []
                        news_by_date[date_key].append(article)
                    except:
                        continue
            
            for stock, data in market_data.items():
                historical_data = data.get('price_data', {}).get('historical_data', [])
                
                if len(historical_data) < 3:
                    continue
                
                returns = []
                dates = []
                
                for i in range(1, len(historical_data)):
                    prev_close = historical_data[i-1].get('Close', 0)
                    curr_close = historical_data[i].get('Close', 0)
                    
                    if prev_close > 0:
                        daily_return = (curr_close - prev_close) / prev_close
                        returns.append(daily_return)
                        
                        date_str = historical_data[i].get('Date', '')
                        if date_str:
                            try:
                                date_obj = pd.to_datetime(date_str).date()
                                dates.append(date_obj)
                            except:
                                dates.append(None)
                        else:
                            dates.append(None)
                
                news_impact_scores = []
                for date, return_val in zip(dates, returns):
                    if date and date in news_by_date:
                        day_articles = news_by_date[date]
                        impact_score = sum(article.get('relevance_score', 0.5) for article in day_articles)
                        news_impact_scores.append(impact_score)
                    else:
                        news_impact_scores.append(0)
                
                if len(returns) > 5 and len(news_impact_scores) > 5:
                    try:
                        correlation, p_value = stats.pearsonr(returns, news_impact_scores)
                        
                        correlations[stock] = {
                            'correlation': correlation,
                            'p_value': p_value,
                            'significance': 'significant' if p_value < 0.05 else 'not_significant',
                            'sample_size': len(returns),
                            'interpretation': self._interpret_news_correlation(correlation, p_value)
                        }
                    except:
                        correlations[stock] = {'correlation': 0, 'significance': 'insufficient_data'}
            
            return {
                'stock_correlations': correlations,
                'overall_correlation': self._calculate_overall_news_correlation(correlations),
                'strongest_correlations': sorted(
                    [(k, v) for k, v in correlations.items() if 'correlation' in v],
                    key=lambda x: abs(x[1]['correlation']),
                    reverse=True
                )[:5]
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing news-price correlations: {str(e)}")
            return {'stock_correlations': {}, 'overall_correlation': 0}
    
    async def _analyze_sentiment_price_correlations(
        self, 
        sentiment_analysis: Dict[str, Any], 
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze correlations between sentiment and price movements."""
        try:
            correlations = {}
            combined_sentiment = sentiment_analysis.get('combined_sentiment', {})
            stock_sentiments = combined_sentiment.get('stock_sentiments', {})
            
            for stock in stock_sentiments.keys():
                if stock in market_data:
                    sentiment_score = stock_sentiments[stock].get('combined_score', 0)
                    price_change = market_data[stock].get('price_data', {}).get('price_change_pct', 0)
                    
                    correlations[stock] = {
                        'sentiment_score': sentiment_score,
                        'price_change': price_change,
                        'alignment': self._calculate_sentiment_price_alignment(sentiment_score, price_change),
                        'divergence': abs(sentiment_score * 100 - price_change)  # Measure of divergence
                    }
            
            divergences = []
            for stock, data in correlations.items():
                if data['divergence'] > 5:  # Significant divergence
                    divergences.append({
                        'stock': stock,
                        'sentiment_score': data['sentiment_score'],
                        'price_change': data['price_change'],
                        'divergence': data['divergence'],
                        'opportunity_type': self._classify_divergence_opportunity(data)
                    })
            
            return {
                'stock_correlations': correlations,
                'sentiment_price_divergences': sorted(divergences, key=lambda x: x['divergence'], reverse=True),
                'average_alignment': np.mean([c['alignment'] for c in correlations.values()]) if correlations else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment-price correlations: {str(e)}")
            return {'stock_correlations': {}, 'sentiment_price_divergences': []}
    
    async def _identify_market_inefficiencies(
        self, 
        news_data: List[Dict[str, Any]], 
        market_data: Dict[str, Any], 
        sentiment_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Identify potential market inefficiencies."""
        try:
            inefficiencies = []
            
            delayed_reactions = await self._find_delayed_reactions(news_data, market_data)
            inefficiencies.extend(delayed_reactions)
            
            overreactions = await self._find_overreactions(market_data, sentiment_analysis)
            inefficiencies.extend(overreactions)
            
            underreactions = await self._find_underreactions(market_data, sentiment_analysis)
            inefficiencies.extend(underreactions)
            
            inefficiencies.sort(key=lambda x: x.get('opportunity_score', 0), reverse=True)
            
            return inefficiencies[:10]  # Return top 10 inefficiencies
            
        except Exception as e:
            self.logger.error(f"Error identifying market inefficiencies: {str(e)}")
            return []
    
    async def _find_delayed_reactions(
        self, 
        news_data: List[Dict[str, Any]], 
        market_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find stocks that may have delayed reactions to news."""
        delayed_reactions = []
        
        try:
            for stock, data in market_data.items():
                price_change = abs(data.get('price_data', {}).get('price_change_pct', 0))
                volume_ratio = data.get('volume_analysis', {}).get('volume_ratio', 1)
                
                relevant_news_count = sum(1 for article in news_data 
                                        if stock.lower() in article.get('title', '').lower() or 
                                           stock.lower() in article.get('description', '').lower())
                
                if relevant_news_count > 0 and price_change < 2 and volume_ratio < 1.2:
                    delayed_reactions.append({
                        'type': 'delayed_reaction',
                        'stock': stock,
                        'news_count': relevant_news_count,
                        'price_change': price_change,
                        'volume_ratio': volume_ratio,
                        'opportunity_score': relevant_news_count * (2 - price_change),
                        'description': f"{stock} has {relevant_news_count} news mentions but minimal price/volume reaction"
                    })
        
        except Exception as e:
            self.logger.error(f"Error finding delayed reactions: {str(e)}")
        
        return delayed_reactions
    
    async def _find_overreactions(
        self, 
        market_data: Dict[str, Any], 
        sentiment_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find stocks that may have overreacted to news/sentiment."""
        overreactions = []
        
        try:
            combined_sentiment = sentiment_analysis.get('combined_sentiment', {})
            stock_sentiments = combined_sentiment.get('stock_sentiments', {})
            
            for stock, sentiment_data in stock_sentiments.items():
                if stock in market_data:
                    sentiment_score = sentiment_data.get('combined_score', 0)
                    price_change = market_data[stock].get('price_data', {}).get('price_change_pct', 0)
                    volume_ratio = market_data[stock].get('volume_analysis', {}).get('volume_ratio', 1)
                    
                    if abs(price_change) > 5 and abs(sentiment_score) < 0.3:
                        overreactions.append({
                            'type': 'overreaction',
                            'stock': stock,
                            'price_change': price_change,
                            'sentiment_score': sentiment_score,
                            'volume_ratio': volume_ratio,
                            'opportunity_score': abs(price_change) * (0.5 - abs(sentiment_score)),
                            'description': f"{stock} moved {price_change:.1f}% with weak sentiment justification"
                        })
        
        except Exception as e:
            self.logger.error(f"Error finding overreactions: {str(e)}")
        
        return overreactions
    
    async def _find_underreactions(
        self, 
        market_data: Dict[str, Any], 
        sentiment_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find stocks that may have underreacted to strong sentiment."""
        underreactions = []
        
        try:
            combined_sentiment = sentiment_analysis.get('combined_sentiment', {})
            stock_sentiments = combined_sentiment.get('stock_sentiments', {})
            
            for stock, sentiment_data in stock_sentiments.items():
                if stock in market_data:
                    sentiment_score = sentiment_data.get('combined_score', 0)
                    confidence = sentiment_data.get('confidence', 0)
                    price_change = market_data[stock].get('price_data', {}).get('price_change_pct', 0)
                    
                    if abs(sentiment_score) > 0.4 and confidence > 0.6 and abs(price_change) < 3:
                        underreactions.append({
                            'type': 'underreaction',
                            'stock': stock,
                            'sentiment_score': sentiment_score,
                            'confidence': confidence,
                            'price_change': price_change,
                            'opportunity_score': abs(sentiment_score) * confidence * (5 - abs(price_change)),
                            'description': f"{stock} has strong sentiment ({sentiment_score:.2f}) but minimal price movement"
                        })
        
        except Exception as e:
            self.logger.error(f"Error finding underreactions: {str(e)}")
        
        return underreactions
    
    async def _analyze_volume_patterns(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze volume patterns across stocks."""
        try:
            volume_analysis = {
                'high_volume_stocks': [],
                'unusual_volume_patterns': [],
                'volume_price_divergences': []
            }
            
            for stock, data in market_data.items():
                volume_data = data.get('volume_analysis', {})
                price_data = data.get('price_data', {})
                
                volume_ratio = volume_data.get('volume_ratio', 1)
                price_change = price_data.get('price_change_pct', 0)
                
                if volume_ratio > 2:
                    volume_analysis['high_volume_stocks'].append({
                        'stock': stock,
                        'volume_ratio': volume_ratio,
                        'price_change': price_change,
                        'volume_trend': volume_data.get('volume_trend', 'normal')
                    })
                
                if volume_ratio > 1.5 and abs(price_change) < 1:
                    volume_analysis['volume_price_divergences'].append({
                        'stock': stock,
                        'volume_ratio': volume_ratio,
                        'price_change': price_change,
                        'divergence_type': 'high_volume_low_movement'
                    })
                elif volume_ratio < 0.7 and abs(price_change) > 3:
                    volume_analysis['volume_price_divergences'].append({
                        'stock': stock,
                        'volume_ratio': volume_ratio,
                        'price_change': price_change,
                        'divergence_type': 'low_volume_high_movement'
                    })
            
            return volume_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing volume patterns: {str(e)}")
            return {'high_volume_stocks': [], 'unusual_volume_patterns': [], 'volume_price_divergences': []}
    
    async def _analyze_market_momentum(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market momentum across stocks."""
        try:
            momentum_stocks = []
            
            for stock, data in market_data.items():
                price_data = data.get('price_data', {})
                technical = data.get('technical_indicators', {})
                market_metrics = data.get('market_metrics', {})
                
                momentum_score = market_metrics.get('momentum_score', 0)
                price_change = price_data.get('price_change_pct', 0)
                rsi = technical.get('rsi', 50)
                
                combined_momentum = (momentum_score + price_change / 100) / 2
                
                momentum_stocks.append({
                    'stock': stock,
                    'momentum_score': momentum_score,
                    'price_change': price_change,
                    'rsi': rsi,
                    'combined_momentum': combined_momentum,
                    'momentum_category': self._categorize_momentum(combined_momentum, rsi)
                })
            
            momentum_stocks.sort(key=lambda x: x['combined_momentum'], reverse=True)
            
            return {
                'top_momentum': momentum_stocks[:10],
                'bottom_momentum': momentum_stocks[-10:],
                'momentum_distribution': self._calculate_momentum_distribution(momentum_stocks)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market momentum: {str(e)}")
            return {'top_momentum': [], 'bottom_momentum': [], 'momentum_distribution': {}}
    
    async def _identify_reversal_patterns(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential reversal patterns."""
        try:
            reversal_candidates = []
            
            for stock, data in market_data.items():
                technical = data.get('technical_indicators', {})
                price_data = data.get('price_data', {})
                market_metrics = data.get('market_metrics', {})
                
                rsi = technical.get('rsi', 50)
                price_change = price_data.get('price_change_pct', 0)
                position_in_range = market_metrics.get('position_in_52w_range', 0.5)
                
                if rsi < 30 and price_change < -5:
                    reversal_candidates.append({
                        'stock': stock,
                        'pattern_type': 'oversold_reversal',
                        'rsi': rsi,
                        'price_change': price_change,
                        'position_in_range': position_in_range,
                        'reversal_probability': self._calculate_reversal_probability(rsi, price_change, position_in_range)
                    })
                
                elif rsi > 70 and price_change > 5:
                    reversal_candidates.append({
                        'stock': stock,
                        'pattern_type': 'overbought_reversal',
                        'rsi': rsi,
                        'price_change': price_change,
                        'position_in_range': position_in_range,
                        'reversal_probability': self._calculate_reversal_probability(100 - rsi, -price_change, 1 - position_in_range)
                    })
            
            reversal_candidates.sort(key=lambda x: x['reversal_probability'], reverse=True)
            
            return reversal_candidates[:10]
            
        except Exception as e:
            self.logger.error(f"Error identifying reversal patterns: {str(e)}")
            return []
    
    async def _generate_market_signals(self, market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate actionable market signals."""
        try:
            signals = []
            
            for stock, data in market_data.items():
                technical = data.get('technical_indicators', {})
                volume_analysis = data.get('volume_analysis', {})
                price_data = data.get('price_data', {})
                
                rsi = technical.get('rsi', 50)
                macd_signal = technical.get('macd_signal_trend', 'neutral')
                price_vs_ma20 = technical.get('price_vs_ma20', 0)
                
                volume_trend = volume_analysis.get('volume_trend', 'normal')
                volume_ratio = volume_analysis.get('volume_ratio', 1)
                
                if rsi < 30 and macd_signal == 'bullish' and volume_ratio > 1.2:
                    signals.append({
                        'stock': stock,
                        'signal': 'oversold_bounce',
                        'strength': (30 - rsi) / 30 * volume_ratio,
                        'description': f"{stock} oversold with bullish MACD and high volume"
                    })
                
                elif rsi > 70 and macd_signal == 'bearish' and volume_ratio > 1.2:
                    signals.append({
                        'stock': stock,
                        'signal': 'overbought_reversal',
                        'strength': (rsi - 70) / 30 * volume_ratio,
                        'description': f"{stock} overbought with bearish MACD and high volume"
                    })
                
                elif price_vs_ma20 > 5 and volume_trend == 'high':
                    signals.append({
                        'stock': stock,
                        'signal': 'breakout_momentum',
                        'strength': min(price_vs_ma20 / 10, 1) * volume_ratio,
                        'description': f"{stock} breaking above MA20 with high volume"
                    })
            
            signals.sort(key=lambda x: x['strength'], reverse=True)
            
            return signals[:15]
            
        except Exception as e:
            self.logger.error(f"Error generating market signals: {str(e)}")
            return []
    
    def _interpret_news_correlation(self, correlation: float, p_value: float) -> str:
        """Interpret news-price correlation."""
        if p_value >= 0.05:
            return "No significant correlation"
        elif correlation > 0.3:
            return "Strong positive correlation - news drives price up"
        elif correlation < -0.3:
            return "Strong negative correlation - news drives price down"
        elif correlation > 0.1:
            return "Moderate positive correlation"
        elif correlation < -0.1:
            return "Moderate negative correlation"
        else:
            return "Weak correlation"
    
    def _calculate_overall_news_correlation(self, correlations: Dict[str, Any]) -> float:
        """Calculate overall news-price correlation."""
        valid_correlations = [v['correlation'] for v in correlations.values() 
                            if 'correlation' in v and v.get('significance') == 'significant']
        
        return np.mean(valid_correlations) if valid_correlations else 0
    
    def _calculate_sentiment_price_alignment(self, sentiment_score: float, price_change: float) -> float:
        """Calculate alignment between sentiment and price movement."""
        normalized_sentiment = sentiment_score * 100
        
        if abs(normalized_sentiment) < 1 and abs(price_change) < 1:
            return 0  # Both neutral
        
        alignment = np.sign(normalized_sentiment) * np.sign(price_change)
        
        magnitude_weight = min(abs(normalized_sentiment), abs(price_change)) / max(abs(normalized_sentiment), abs(price_change), 1)
        
        return alignment * magnitude_weight
    
    def _classify_divergence_opportunity(self, data: Dict[str, Any]) -> str:
        """Classify the type of sentiment-price divergence opportunity."""
        sentiment_score = data['sentiment_score']
        price_change = data['price_change']
        
        if sentiment_score > 0.2 and price_change < -2:
            return "bullish_divergence"  # Positive sentiment, negative price
        elif sentiment_score < -0.2 and price_change > 2:
            return "bearish_divergence"  # Negative sentiment, positive price
        elif sentiment_score > 0.3 and abs(price_change) < 1:
            return "undervalued_positive"  # Strong positive sentiment, no price movement
        elif sentiment_score < -0.3 and abs(price_change) < 1:
            return "overvalued_negative"  # Strong negative sentiment, no price movement
        else:
            return "neutral_divergence"
    
    def _categorize_momentum(self, momentum: float, rsi: float) -> str:
        """Categorize momentum strength."""
        if momentum > 0.1 and rsi < 70:
            return "strong_bullish"
        elif momentum > 0.05:
            return "moderate_bullish"
        elif momentum < -0.1 and rsi > 30:
            return "strong_bearish"
        elif momentum < -0.05:
            return "moderate_bearish"
        else:
            return "neutral"
    
    def _calculate_momentum_distribution(self, momentum_stocks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of momentum categories."""
        distribution = {}
        for stock in momentum_stocks:
            category = stock['momentum_category']
            distribution[category] = distribution.get(category, 0) + 1
        
        return distribution
    
    def _calculate_reversal_probability(self, rsi_extreme: float, price_extreme: float, position_extreme: float) -> float:
        """Calculate probability of price reversal."""
        rsi_factor = min(abs(rsi_extreme - 50) / 50, 1)  # How extreme is RSI
        price_factor = min(abs(price_extreme) / 10, 1)   # How extreme is price change
        position_factor = abs(position_extreme - 0.5) * 2  # Position in 52-week range
        
        probability = (rsi_factor * 0.4 + price_factor * 0.4 + position_factor * 0.2)
        
        return min(probability, 0.95)  # Cap at 95%
    
    def _get_default_market_analysis(self) -> Dict[str, Any]:
        """Return default market analysis when processing fails."""
        return {
            'news_correlations': {'stock_correlations': {}, 'overall_correlation': 0},
            'sentiment_correlations': {'stock_correlations': {}, 'sentiment_price_divergences': []},
            'market_inefficiencies': [],
            'volume_analysis': {'high_volume_stocks': [], 'volume_price_divergences': []},
            'momentum_analysis': {'top_momentum': [], 'bottom_momentum': []},
            'reversal_patterns': [],
            'market_signals': [],
            'analyzed_at': datetime.now().isoformat(),
            'error': 'Failed to analyze market reactions'
        }
