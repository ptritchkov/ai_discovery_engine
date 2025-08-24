"""Investment decision engine that combines all analysis to generate recommendations."""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import logging
import numpy as np

class InvestmentEngine:
    """Generates investment recommendations based on comprehensive analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.risk_tolerance = 0.7  # Default moderate risk tolerance
        self.confidence_threshold = 0.5  # Minimum confidence for recommendations
        self.max_recommendations = 20  # Maximum number of recommendations to generate
        
    async def generate_recommendations(
        self,
        news_data: List[Dict[str, Any]],
        sentiment_analysis: Dict[str, Any],
        market_analysis: Dict[str, Any],
        ml_predictions: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate investment recommendations based on all analysis.
        
        Args:
            news_data: News articles and analysis
            sentiment_analysis: Comprehensive sentiment analysis
            market_analysis: Market reaction analysis
            ml_predictions: ML model predictions
            
        Returns:
            List of investment recommendations
        """
        self.logger.info("Generating investment recommendations...")
        
        try:
            combined_sentiment = sentiment_analysis.get('combined_sentiment', {})
            stock_sentiments = combined_sentiment.get('stock_sentiments', {})
            individual_predictions = ml_predictions.get('individual_predictions', {})
            market_inefficiencies = market_analysis.get('market_inefficiencies', [])
            
            self.logger.info(f"Stocks with sentiment data: {list(stock_sentiments.keys())}")
            self.logger.info(f"Stocks with ML predictions: {list(individual_predictions.keys())}")
            self.logger.info(f"Market inefficiencies found: {len(market_inefficiencies)}")
            
            stock_recommendations = []
            
            # Get all stocks that have either sentiment data or ML predictions
            all_stocks = set(stock_sentiments.keys()) | set(individual_predictions.keys())
            self.logger.info(f"Analyzing {len(all_stocks)} stocks: {list(all_stocks)}")
            
            for stock in all_stocks:
                try:
                    recommendation = await self._analyze_stock_opportunity(
                        stock, news_data, sentiment_analysis, market_analysis, ml_predictions
                    )
                    
                    if recommendation:
                        confidence = recommendation.get('confidence', 0)
                        if confidence >= self.confidence_threshold:
                            stock_recommendations.append(recommendation)
                            self.logger.info(f"Added {stock} recommendation: {recommendation.get('action', 'N/A')} (confidence: {confidence:.3f})")
                        else:
                            self.logger.debug(f"Filtered out {stock} due to low confidence: {confidence:.3f} < {self.confidence_threshold}")
                    else:
                        self.logger.debug(f"No recommendation generated for {stock}")
                        
                except Exception as e:
                    self.logger.warning(f"Error analyzing {stock}: {str(e)}")
                    continue
            
            inefficiency_recommendations = await self._generate_inefficiency_recommendations(market_inefficiencies)
            stock_recommendations.extend(inefficiency_recommendations)
            
            scored_recommendations = await self._score_and_rank_recommendations(stock_recommendations)
            
            final_recommendations = await self._apply_portfolio_theory(scored_recommendations)
            
            summary = await self._generate_recommendation_summary(final_recommendations, sentiment_analysis, market_analysis)
            
            self.logger.info(f"Generated {len(final_recommendations)} investment recommendations")
            
            return {
                'recommendations': final_recommendations[:self.max_recommendations],
                'summary': summary,
                'total_opportunities': len(stock_recommendations),
                'high_confidence_count': len([r for r in final_recommendations if r.get('confidence', 0) > 0.8]),
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {str(e)}")
            return {'recommendations': [], 'summary': {}, 'error': str(e)}
    
    async def _analyze_stock_opportunity(
        self,
        stock: str,
        news_data: List[Dict[str, Any]],
        sentiment_analysis: Dict[str, Any],
        market_analysis: Dict[str, Any],
        ml_predictions: Dict[str, Any]
    ) -> Dict[str, Any] | None:
        """Analyze investment opportunity for a single stock."""
        try:
            combined_sentiment = sentiment_analysis.get('combined_sentiment', {})
            stock_sentiment = combined_sentiment.get('stock_sentiments', {}).get(stock, {})
            ml_prediction = ml_predictions.get('individual_predictions', {}).get(stock, {})
            
            sentiment_score = stock_sentiment.get('combined_score', 0)
            sentiment_confidence = stock_sentiment.get('confidence', 0)
            ml_predicted_return = ml_prediction.get('predicted_return', 0)
            ml_confidence = ml_prediction.get('confidence', 0)
            ml_direction = ml_prediction.get('direction', 'neutral')
            
            investment_thesis = await self._generate_investment_thesis(
                stock, sentiment_score, ml_predicted_return, news_data
            )
            
            risk_assessment = await self._assess_investment_risk(stock, market_analysis, ml_prediction)
            
            opportunity_score = await self._calculate_opportunity_score(
                sentiment_score, sentiment_confidence, ml_predicted_return, ml_confidence, risk_assessment
            )
            
            action, position_size = await self._determine_action_and_size(
                opportunity_score, risk_assessment, ml_direction
            )
            
            self.logger.debug(f"{stock}: ml_direction={ml_direction}, opportunity_score={opportunity_score.get('total_score', 0):.3f}, action={action}")
            
            if action == 'hold':
                return None  # Skip neutral recommendations
            
            price_targets = await self._calculate_price_targets(stock, ml_predicted_return, risk_assessment)
            
            reasoning = await self._generate_reasoning(
                stock, action, sentiment_score, ml_predicted_return, investment_thesis, risk_assessment
            )
            
            return {
                'symbol': stock,
                'action': action,
                'confidence': opportunity_score['confidence'],
                'expected_return': ml_predicted_return,
                'position_size': position_size,
                'risk_level': risk_assessment['risk_level'],
                'time_horizon': self._determine_time_horizon(ml_prediction, sentiment_analysis),
                'price_targets': price_targets,
                'reasoning': reasoning,
                'investment_thesis': investment_thesis,
                'sentiment_score': sentiment_score,
                'ml_prediction': ml_predicted_return,
                'opportunity_score': opportunity_score['total_score'],
                'risk_reward_ratio': opportunity_score['risk_reward_ratio'],
                'supporting_factors': await self._identify_supporting_factors(stock, sentiment_analysis, market_analysis),
                'risk_factors': risk_assessment['risk_factors'],
                'news_catalyst': await self._identify_news_catalyst(stock, news_data),
                'technical_signals': await self._extract_technical_signals(stock, market_analysis),
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing opportunity for {stock}: {str(e)}")
            return None
    
    async def _generate_investment_thesis(
        self,
        stock: str,
        sentiment_score: float,
        predicted_return: float,
        news_data: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate investment thesis based on available data."""
        try:
            relevant_news = [
                article for article in news_data
                if stock.lower() in article.get('title', '').lower() or
                   stock.lower() in article.get('description', '').lower()
            ]
            
            if predicted_return > 0.05 and sentiment_score > 0.2:
                thesis_type = "Growth Momentum"
                thesis = f"Strong positive sentiment and ML prediction suggest {stock} has significant upward momentum."
            elif predicted_return > 0.02 and sentiment_score > 0:
                thesis_type = "Value Recovery"
                thesis = f"Moderate positive indicators suggest {stock} is undervalued with recovery potential."
            elif predicted_return < -0.05 and sentiment_score < -0.2:
                thesis_type = "Bearish Outlook"
                thesis = f"Negative sentiment and ML prediction indicate {stock} faces significant headwinds."
            elif abs(sentiment_score) > 0.3 and abs(predicted_return) < 0.02:
                thesis_type = "Sentiment Divergence"
                thesis = f"Strong sentiment not reflected in price prediction suggests market inefficiency for {stock}."
            else:
                thesis_type = "Neutral"
                thesis = f"Mixed signals for {stock} suggest a wait-and-see approach."
            
            if relevant_news:
                news_context = f"Recent news coverage includes {len(relevant_news)} relevant articles."
            else:
                news_context = "Limited recent news coverage."
            
            return {
                'type': thesis_type,
                'thesis': thesis,
                'news_context': news_context,
                'relevant_news_count': len(relevant_news),
                'confidence': min(abs(sentiment_score) + abs(predicted_return), 1.0)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating investment thesis for {stock}: {str(e)}")
            return {'type': 'Unknown', 'thesis': 'Unable to generate thesis', 'confidence': 0.1}
    
    async def _assess_investment_risk(
        self,
        stock: str,
        market_analysis: Dict[str, Any],
        ml_prediction: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Assess investment risk for a stock."""
        try:
            risk_factors = []
            risk_score = 0.5  # Base risk score
            
            inefficiencies = market_analysis.get('market_inefficiencies', [])
            stock_inefficiencies = [ineff for ineff in inefficiencies if ineff.get('stock') == stock]
            
            if stock_inefficiencies:
                risk_factors.append("Market inefficiency detected")
                risk_score += 0.1
            
            ml_confidence = ml_prediction.get('confidence', 0)
            if ml_confidence < 0.5:
                risk_factors.append("Low ML prediction confidence")
                risk_score += 0.15
            
            volume_analysis = market_analysis.get('volume_analysis', {})
            high_volume_stocks = volume_analysis.get('high_volume_stocks', [])
            
            if any(item.get('stock') == stock for item in high_volume_stocks):
                volume_item = next(item for item in high_volume_stocks if item.get('stock') == stock)
                if volume_item.get('volume_ratio', 1) > 3:
                    risk_factors.append("Unusually high trading volume")
                    risk_score += 0.1
            
            reversal_patterns = market_analysis.get('reversal_patterns', [])
            if any(pattern.get('stock') == stock for pattern in reversal_patterns):
                risk_factors.append("Potential reversal pattern detected")
                risk_score += 0.1
            
            if risk_score < 0.4:
                risk_level = "Low"
            elif risk_score < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            return {
                'risk_score': min(risk_score, 1.0),
                'risk_level': risk_level,
                'risk_factors': risk_factors,
                'volatility_assessment': self._assess_volatility_risk(stock, market_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing risk for {stock}: {str(e)}")
            return {'risk_score': 0.7, 'risk_level': 'High', 'risk_factors': ['Assessment error']}
    
    async def _calculate_opportunity_score(
        self,
        sentiment_score: float,
        sentiment_confidence: float,
        predicted_return: float,
        ml_confidence: float,
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall opportunity score."""
        try:
            # Adjusted weights to preserve LLM confidence scores
            if sentiment_score == 0 and sentiment_confidence == 0:
                # No sentiment data - rely more on ML predictions
                sentiment_weight = 0.0
                prediction_weight = 0.5
                confidence_weight = 0.4  # Increased to preserve LLM confidence
                risk_weight = 0.1
            else:
                sentiment_weight = 0.2
                prediction_weight = 0.3
                confidence_weight = 0.4  # Increased to preserve LLM confidence
                risk_weight = 0.1
            
            sentiment_component = abs(sentiment_score) * sentiment_confidence * sentiment_weight
            prediction_component = abs(predicted_return) * ml_confidence * prediction_weight
            confidence_component = max(sentiment_confidence, ml_confidence) * confidence_weight  # Use max instead of average
            risk_component = (1 - risk_assessment['risk_score']) * risk_weight
            
            total_score = sentiment_component + prediction_component + confidence_component + risk_component
            
            # Boost score when we have directional ML predictions
            if abs(predicted_return) > 0.01:
                total_score *= 1.2  # 20% boost for having directional predictions
            
            expected_return = abs(predicted_return)
            risk_score = risk_assessment['risk_score']
            risk_reward_ratio = expected_return / max(risk_score, 0.1) if expected_return > 0 else 0
            
            overall_confidence = min(total_score, 0.95)
            
            return {
                'total_score': total_score,
                'confidence': overall_confidence,
                'risk_reward_ratio': risk_reward_ratio,
                'components': {
                    'sentiment': sentiment_component,
                    'prediction': prediction_component,
                    'confidence': confidence_component,
                    'risk_adjustment': risk_component
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating opportunity score: {str(e)}")
            return {'total_score': 0.3, 'confidence': 0.3, 'risk_reward_ratio': 0.5}
    
    async def _determine_action_and_size(
        self,
        opportunity_score: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        ml_direction: str
    ) -> Tuple[str, str]:
        """Determine investment action and position size."""
        try:
            total_score = opportunity_score['total_score']
            confidence = opportunity_score['confidence']
            risk_level = risk_assessment['risk_level']
            
            # Lowered thresholds for testing
            if ml_direction == 'bullish' and total_score > 0.3:
                action = 'buy'
            elif ml_direction == 'bearish' and total_score > 0.3:
                action = 'sell'
            elif ml_direction == 'bullish' and total_score > 0.2:
                action = 'buy'
            elif ml_direction == 'bearish' and total_score > 0.2:
                action = 'sell'
            else:
                action = 'hold'
            
            if action == 'hold':
                position_size = 'none'
            elif confidence > 0.8 and risk_level == 'Low':
                position_size = 'large'
            elif confidence > 0.7 and risk_level in ['Low', 'Medium']:
                position_size = 'medium'
            elif confidence > 0.6:
                position_size = 'small'
            else:
                position_size = 'minimal'
            
            return action, position_size
            
        except Exception as e:
            self.logger.error(f"Error determining action and size: {str(e)}")
            return 'hold', 'none'
    
    async def _calculate_price_targets(
        self,
        stock: str,
        predicted_return: float,
        risk_assessment: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate price targets and stop losses."""
        try:
            
            expected_move = abs(predicted_return) * 100  # Convert to percentage
            risk_score = risk_assessment['risk_score']
            
            if predicted_return > 0:  # Bullish
                target_1 = expected_move * 0.5
                target_2 = expected_move
                stop_loss = -expected_move * risk_score
            else:  # Bearish
                target_1 = -expected_move * 0.5
                target_2 = -expected_move
                stop_loss = expected_move * risk_score
            
            return {
                'target_1': round(target_1, 2),
                'target_2': round(target_2, 2),
                'stop_loss': round(stop_loss, 2),
                'risk_reward_1': abs(target_1 / stop_loss) if stop_loss != 0 else 0,
                'risk_reward_2': abs(target_2 / stop_loss) if stop_loss != 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating price targets for {stock}: {str(e)}")
            return {'target_1': 0, 'target_2': 0, 'stop_loss': 0}
    
    async def _generate_reasoning(
        self,
        stock: str,
        action: str,
        sentiment_score: float,
        predicted_return: float,
        investment_thesis: Dict[str, Any],
        risk_assessment: Dict[str, Any]
    ) -> str:
        """Generate human-readable reasoning for the recommendation."""
        try:
            reasoning_parts = []
            
            if action == 'buy':
                reasoning_parts.append(f"Recommending BUY for {stock} based on positive indicators.")
            elif action == 'sell':
                reasoning_parts.append(f"Recommending SELL for {stock} due to negative outlook.")
            
            if sentiment_score > 0.2:
                reasoning_parts.append(f"Strong positive sentiment ({sentiment_score:.2f}) suggests market optimism.")
            elif sentiment_score < -0.2:
                reasoning_parts.append(f"Negative sentiment ({sentiment_score:.2f}) indicates market pessimism.")
            
            if abs(predicted_return) > 0.03:
                direction = "upward" if predicted_return > 0 else "downward"
                reasoning_parts.append(f"ML model predicts {direction} movement of {abs(predicted_return)*100:.1f}%.")
            
            reasoning_parts.append(f"Investment thesis: {investment_thesis.get('thesis', 'Standard analysis')}.")
            
            risk_level = risk_assessment.get('risk_level', 'Medium')
            reasoning_parts.append(f"Risk level assessed as {risk_level}.")
            
            return " ".join(reasoning_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating reasoning for {stock}: {str(e)}")
            return f"Recommendation for {stock} based on comprehensive analysis."
    
    async def _generate_inefficiency_recommendations(
        self,
        inefficiencies: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate recommendations from market inefficiencies."""
        try:
            recommendations = []
            
            for inefficiency in inefficiencies[:5]:  # Limit to top 5
                stock = inefficiency.get('stock')
                inefficiency_type = inefficiency.get('type')
                opportunity_score = inefficiency.get('opportunity_score', 0)
                
                if opportunity_score < 3:  # Minimum threshold
                    continue
                
                if inefficiency_type == 'underreaction':
                    action = 'buy'
                    reasoning = f"Market appears to have underreacted to positive sentiment for {stock}."
                elif inefficiency_type == 'overreaction':
                    action = 'sell'
                    reasoning = f"Market may have overreacted to news for {stock}, suggesting reversal opportunity."
                elif inefficiency_type == 'delayed_reaction':
                    action = 'buy'
                    reasoning = f"Delayed market reaction to news for {stock} suggests catching up potential."
                else:
                    continue
                
                recommendations.append({
                    'symbol': stock,
                    'action': action,
                    'confidence': min(opportunity_score / 10, 0.8),
                    'expected_return': 0.03 if action == 'buy' else -0.03,
                    'position_size': 'small',
                    'risk_level': 'Medium',
                    'time_horizon': 'short_term',
                    'reasoning': reasoning,
                    'opportunity_type': 'market_inefficiency',
                    'inefficiency_details': inefficiency,
                    'generated_at': datetime.now().isoformat()
                })
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating inefficiency recommendations: {str(e)}")
            return []
    
    async def _score_and_rank_recommendations(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Score and rank all recommendations."""
        try:
            for rec in recommendations:
                confidence = rec.get('confidence', 0)
                expected_return = abs(rec.get('expected_return', 0))
                risk_reward = rec.get('risk_reward_ratio', 0)
                
                composite_score = (
                    confidence * 0.4 +
                    expected_return * 10 * 0.3 +  # Scale expected return
                    min(risk_reward, 3) / 3 * 0.3  # Cap and normalize risk-reward
                )
                
                rec['composite_score'] = composite_score
            
            recommendations.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error scoring recommendations: {str(e)}")
            return recommendations
    
    async def _apply_portfolio_theory(
        self,
        recommendations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply portfolio theory to optimize recommendations."""
        try:
            final_recommendations = []
            buy_count = 0
            sell_count = 0
            sectors_covered = set()
            
            for rec in recommendations:
                action = rec.get('action')
                
                if action == 'buy' and buy_count >= 10:
                    continue
                elif action == 'sell' and sell_count >= 5:
                    continue
                
                rec['diversification_benefit'] = self._calculate_diversification_benefit(
                    rec, final_recommendations
                )
                
                final_recommendations.append(rec)
                
                if action == 'buy':
                    buy_count += 1
                elif action == 'sell':
                    sell_count += 1
            
            return final_recommendations
            
        except Exception as e:
            self.logger.error(f"Error applying portfolio theory: {str(e)}")
            return recommendations
    
    async def _generate_recommendation_summary(
        self,
        recommendations: List[Dict[str, Any]],
        sentiment_analysis: Dict[str, Any],
        market_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary of recommendations and market outlook."""
        try:
            if not recommendations:
                return {'market_outlook': 'neutral', 'recommendation_count': 0}
            
            buy_count = sum(1 for r in recommendations if r.get('action') == 'buy')
            sell_count = sum(1 for r in recommendations if r.get('action') == 'sell')
            
            avg_confidence = np.mean([r.get('confidence', 0) for r in recommendations])
            
            overall_sentiment = sentiment_analysis.get('overall_sentiment', {})
            market_mood = overall_sentiment.get('market_mood', 'neutral')
            
            if buy_count > sell_count * 2:
                outlook = 'bullish'
            elif sell_count > buy_count * 2:
                outlook = 'bearish'
            else:
                outlook = 'mixed'
            
            top_opportunities = recommendations[:3]
            
            return {
                'market_outlook': outlook,
                'market_mood': market_mood,
                'recommendation_count': len(recommendations),
                'buy_recommendations': buy_count,
                'sell_recommendations': sell_count,
                'average_confidence': avg_confidence,
                'top_opportunities': [
                    {
                        'symbol': opp.get('symbol'),
                        'action': opp.get('action'),
                        'confidence': opp.get('confidence'),
                        'reasoning': opp.get('reasoning', '')[:100] + '...'
                    }
                    for opp in top_opportunities
                ],
                'risk_distribution': self._calculate_risk_distribution(recommendations),
                'time_horizon_distribution': self._calculate_time_horizon_distribution(recommendations)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return {'market_outlook': 'neutral', 'error': str(e)}
    
    def _assess_volatility_risk(self, stock: str, market_analysis: Dict[str, Any]) -> str:
        """Assess volatility risk for a stock."""
        return "moderate"
    
    def _determine_time_horizon(self, ml_prediction: Dict[str, Any], sentiment_analysis: Dict[str, Any]) -> str:
        """Determine appropriate time horizon for the recommendation."""
        confidence = ml_prediction.get('confidence', 0)
        
        if confidence > 0.8:
            return 'short_term'  # 1-2 weeks
        elif confidence > 0.6:
            return 'medium_term'  # 1-3 months
        else:
            return 'long_term'  # 3+ months
    
    async def _identify_supporting_factors(
        self,
        stock: str,
        sentiment_analysis: Dict[str, Any],
        market_analysis: Dict[str, Any]
    ) -> List[str]:
        """Identify supporting factors for the recommendation."""
        factors = []
        
        combined_sentiment = sentiment_analysis.get('combined_sentiment', {})
        stock_sentiment = combined_sentiment.get('stock_sentiments', {}).get(stock, {})
        
        if stock_sentiment.get('combined_score', 0) > 0.2:
            factors.append("Strong positive sentiment across multiple sources")
        
        market_signals = market_analysis.get('market_signals', [])
        stock_signals = [signal for signal in market_signals if signal.get('stock') == stock]
        
        if stock_signals:
            factors.append(f"Technical signals: {', '.join([s.get('signal', '') for s in stock_signals[:2]])}")
        
        return factors
    
    async def _identify_news_catalyst(self, stock: str, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify news catalyst for the stock."""
        relevant_news = [
            article for article in news_data
            if stock.lower() in article.get('title', '').lower() or
               stock.lower() in article.get('description', '').lower()
        ]
        
        if relevant_news:
            latest_news = max(relevant_news, key=lambda x: x.get('published_at', ''))
            return {
                'has_catalyst': True,
                'title': latest_news.get('title', ''),
                'source': latest_news.get('source', ''),
                'relevance_score': latest_news.get('relevance_score', 0.5)
            }
        
        return {'has_catalyst': False}
    
    async def _extract_technical_signals(self, stock: str, market_analysis: Dict[str, Any]) -> List[str]:
        """Extract technical signals for the stock."""
        signals = []
        
        market_signals = market_analysis.get('market_signals', [])
        stock_signals = [signal for signal in market_signals if signal.get('stock') == stock]
        
        for signal in stock_signals:
            signals.append(signal.get('signal', ''))
        
        return signals
    
    def _calculate_diversification_benefit(
        self,
        recommendation: Dict[str, Any],
        existing_recommendations: List[Dict[str, Any]]
    ) -> float:
        """Calculate diversification benefit of adding this recommendation."""
        
        if not existing_recommendations:
            return 1.0
        
        similar_count = sum(
            1 for rec in existing_recommendations
            if rec.get('action') == recommendation.get('action')
        )
        
        return max(0.1, 1.0 - (similar_count * 0.1))
    
    def _calculate_risk_distribution(self, recommendations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of risk levels."""
        distribution = {'Low': 0, 'Medium': 0, 'High': 0}
        
        for rec in recommendations:
            risk_level = rec.get('risk_level', 'Medium')
            distribution[risk_level] = distribution.get(risk_level, 0) + 1
        
        return distribution
    
    def _calculate_time_horizon_distribution(self, recommendations: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of time horizons."""
        distribution = {'short_term': 0, 'medium_term': 0, 'long_term': 0}
        
        for rec in recommendations:
            horizon = rec.get('time_horizon', 'medium_term')
            distribution[horizon] = distribution.get(horizon, 0) + 1
        
        return distribution
