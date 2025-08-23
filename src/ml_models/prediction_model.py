"""Machine learning model for stock movement prediction."""

import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

class PredictionModel:
    """Machine learning model for predicting stock movements."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_performance = {}
        
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.lr_model = LinearRegression()
        self.scaler = StandardScaler()
        
    async def predict_stock_movements(
        self, 
        market_analysis: Dict[str, Any], 
        sentiment_analysis: Dict[str, Any], 
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict stock movements using machine learning.
        
        Args:
            market_analysis: Market reaction analysis
            sentiment_analysis: Sentiment analysis results
            market_data: Stock market data
            
        Returns:
            ML predictions for stock movements
        """
        self.logger.info("Generating ML predictions for stock movements...")
        
        try:
            features_data = await self._prepare_features(market_analysis, sentiment_analysis, market_data)
            
            if not features_data:
                return self._get_default_predictions()
            
            predictions = {}
            
            for stock, features in features_data.items():
                try:
                    stock_prediction = await self._predict_single_stock(stock, features)
                    predictions[stock] = stock_prediction
                except Exception as e:
                    self.logger.warning(f"Error predicting {stock}: {str(e)}")
                    predictions[stock] = self._get_default_stock_prediction(stock)
            
            ensemble_predictions = await self._generate_ensemble_predictions(predictions)
            
            prediction_confidence = await self._calculate_prediction_confidence(predictions, features_data)
            
            return {
                'individual_predictions': predictions,
                'ensemble_predictions': ensemble_predictions,
                'prediction_confidence': prediction_confidence,
                'feature_importance': self.feature_importance,
                'model_performance': self.model_performance,
                'predicted_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in ML prediction: {str(e)}")
            return self._get_default_predictions()
    
    async def _prepare_features(
        self, 
        market_analysis: Dict[str, Any], 
        sentiment_analysis: Dict[str, Any], 
        market_data: Dict[str, Any]
    ) -> Dict[str, List[float]]:
        """Prepare feature vectors for ML models."""
        try:
            features_data = {}
            
            combined_sentiment = sentiment_analysis.get('combined_sentiment', {})
            stock_sentiments = combined_sentiment.get('stock_sentiments', {})
            
            sentiment_correlations = market_analysis.get('sentiment_correlations', {})
            volume_analysis = market_analysis.get('volume_analysis', {})
            momentum_analysis = market_analysis.get('momentum_analysis', {})
            
            for stock in market_data.keys():
                try:
                    features = await self._extract_stock_features(
                        stock, market_data, stock_sentiments, sentiment_correlations,
                        volume_analysis, momentum_analysis
                    )
                    
                    if features and len(features) > 0:
                        features_data[stock] = features
                        
                except Exception as e:
                    self.logger.warning(f"Error extracting features for {stock}: {str(e)}")
                    continue
            
            return features_data
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {str(e)}")
            return {}
    
    async def _extract_stock_features(
        self, 
        stock: str, 
        market_data: Dict[str, Any], 
        stock_sentiments: Dict[str, Any],
        sentiment_correlations: Dict[str, Any],
        volume_analysis: Dict[str, Any],
        momentum_analysis: Dict[str, Any]
    ) -> List[float]:
        """Extract feature vector for a single stock."""
        try:
            features = []
            stock_data = market_data.get(stock, {})
            
            price_data = stock_data.get('price_data', {})
            features.extend([
                price_data.get('price_change_pct', 0) / 100,  # Normalize to 0-1 scale
                price_data.get('volume', 0) / max(price_data.get('avg_volume', 1), 1),  # Volume ratio
                price_data.get('beta', 1),
                price_data.get('pe_ratio', 15) / 30,  # Normalize PE ratio
            ])
            
            technical = stock_data.get('technical_indicators', {})
            features.extend([
                technical.get('rsi', 50) / 100,  # Normalize RSI to 0-1
                technical.get('price_vs_ma5', 0) / 100,  # Normalize percentage
                technical.get('price_vs_ma20', 0) / 100,
                1 if technical.get('macd_signal_trend') == 'bullish' else -1 if technical.get('macd_signal_trend') == 'bearish' else 0,
                1 if technical.get('rsi_signal') == 'overbought' else -1 if technical.get('rsi_signal') == 'oversold' else 0
            ])
            
            market_metrics = stock_data.get('market_metrics', {})
            features.extend([
                market_metrics.get('position_in_52w_range', 0.5),
                market_metrics.get('momentum_score', 0),
                market_metrics.get('distance_from_52w_high', 50) / 100,
                market_metrics.get('distance_from_52w_low', 50) / 100
            ])
            
            volatility = stock_data.get('volatility_analysis', {})
            features.extend([
                volatility.get('volatility_score', 0.02) * 100,  # Scale volatility
                volatility.get('average_return', 0) * 100,
                volatility.get('sharpe_ratio', 0),
                1 if volatility.get('volatility_trend') == 'high' else -1 if volatility.get('volatility_trend') == 'low' else 0
            ])
            
            volume_data = stock_data.get('volume_analysis', {})
            features.extend([
                volume_data.get('volume_ratio', 1),
                volume_data.get('volume_score', 0.5),
                1 if volume_data.get('volume_trend') == 'high' else -1 if volume_data.get('volume_trend') == 'low' else 0
            ])
            
            sentiment_data = stock_sentiments.get(stock, {})
            features.extend([
                sentiment_data.get('combined_score', 0),
                sentiment_data.get('confidence', 0),
                sentiment_data.get('social_sentiment', 0),
                sentiment_data.get('market_sentiment', 0)
            ])
            
            fundamentals = stock_data.get('fundamentals', {})
            analyst_rating = fundamentals.get('analyst_rating', {})
            rating_score = self._convert_rating_to_score(analyst_rating.get('rating', 'neutral'))
            features.extend([
                rating_score,
                analyst_rating.get('score', 0)
            ])
            
            correlation_data = sentiment_correlations.get('stock_correlations', {}).get(stock, {})
            features.extend([
                correlation_data.get('alignment', 0),
                correlation_data.get('divergence', 0) / 100  # Normalize divergence
            ])
            
            features = [float(f) if not np.isnan(float(f)) else 0.0 for f in features]
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error extracting features for {stock}: {str(e)}")
            return []
    
    async def _predict_single_stock(self, stock: str, features: List[float]) -> Dict[str, Any]:
        """Generate prediction for a single stock."""
        try:
            if len(features) < 10:  # Minimum feature requirement
                return self._get_default_stock_prediction(stock)
            
            X = np.array(features).reshape(1, -1)
            
            predictions = {}
            
            momentum_pred = self._momentum_prediction(features)
            predictions['momentum'] = momentum_pred
            
            sentiment_pred = self._sentiment_prediction(features)
            predictions['sentiment'] = sentiment_pred
            
            technical_pred = self._technical_prediction(features)
            predictions['technical'] = technical_pred
            
            ensemble_pred = (
                momentum_pred * 0.3 + 
                sentiment_pred * 0.4 + 
                technical_pred * 0.3
            )
            
            model_agreement = 1 - np.std([momentum_pred, sentiment_pred, technical_pred])
            confidence = max(0.1, min(0.9, model_agreement))
            
            # Lowered thresholds for testing - more sensitive to small movements
            direction = 'bullish' if ensemble_pred > 0.01 else 'bearish' if ensemble_pred < -0.01 else 'neutral'
            strength = min(abs(ensemble_pred) * 10, 1.0)  # Scale to 0-1
            
            return {
                'stock': stock,
                'predicted_return': ensemble_pred,
                'direction': direction,
                'strength': strength,
                'confidence': confidence,
                'model_predictions': predictions,
                'features_used': len(features),
                'prediction_horizon': '1-5 days'
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting {stock}: {str(e)}")
            return self._get_default_stock_prediction(stock)
    
    def _momentum_prediction(self, features: List[float]) -> float:
        """Simple momentum-based prediction."""
        try:
            price_change = features[0] if len(features) > 0 else 0
            volume_ratio = features[1] if len(features) > 1 else 1
            momentum_score = features[11] if len(features) > 11 else 0
            
            momentum_pred = (price_change * 0.5 + momentum_score * 0.5) * min(volume_ratio, 2.0)
            
            return max(-0.2, min(0.2, momentum_pred))  # Cap prediction
            
        except Exception:
            return 0.0
    
    def _sentiment_prediction(self, features: List[float]) -> float:
        """Sentiment-based prediction."""
        try:
            combined_sentiment = features[18] if len(features) > 18 else 0
            confidence = features[19] if len(features) > 19 else 0
            social_sentiment = features[20] if len(features) > 20 else 0
            market_sentiment = features[21] if len(features) > 21 else 0
            
            weighted_sentiment = (
                combined_sentiment * 0.4 + 
                social_sentiment * 0.3 + 
                market_sentiment * 0.3
            ) * confidence
            
            return max(-0.15, min(0.15, weighted_sentiment))
            
        except Exception:
            return 0.0
    
    def _technical_prediction(self, features: List[float]) -> float:
        """Technical analysis-based prediction."""
        try:
            rsi = features[4] if len(features) > 4 else 0.5
            price_vs_ma20 = features[6] if len(features) > 6 else 0
            macd_signal = features[7] if len(features) > 7 else 0
            position_in_range = features[9] if len(features) > 9 else 0.5
            
            technical_score = 0
            
            if rsi < 0.3:  # Oversold
                technical_score += 0.05
            elif rsi > 0.7:  # Overbought
                technical_score -= 0.05
            
            technical_score += price_vs_ma20 * 0.5
            
            technical_score += macd_signal * 0.03
            
            if position_in_range < 0.2:  # Near 52-week low
                technical_score += 0.02
            elif position_in_range > 0.8:  # Near 52-week high
                technical_score -= 0.02
            
            return max(-0.1, min(0.1, technical_score))
            
        except Exception:
            return 0.0
    
    async def _generate_ensemble_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ensemble predictions across all stocks."""
        try:
            if not predictions:
                return {}
            
            bullish_stocks = []
            bearish_stocks = []
            neutral_stocks = []
            
            for stock, pred in predictions.items():
                direction = pred.get('direction', 'neutral')
                strength = pred.get('strength', 0)
                confidence = pred.get('confidence', 0)
                
                stock_info = {
                    'stock': stock,
                    'predicted_return': pred.get('predicted_return', 0),
                    'strength': strength,
                    'confidence': confidence,
                    'score': strength * confidence
                }
                
                if direction == 'bullish':
                    bullish_stocks.append(stock_info)
                elif direction == 'bearish':
                    bearish_stocks.append(stock_info)
                else:
                    neutral_stocks.append(stock_info)
            
            bullish_stocks.sort(key=lambda x: x['score'], reverse=True)
            bearish_stocks.sort(key=lambda x: x['score'], reverse=True)
            
            return {
                'top_bullish': bullish_stocks[:10],
                'top_bearish': bearish_stocks[:10],
                'neutral': neutral_stocks,
                'market_sentiment': self._calculate_overall_market_sentiment(predictions)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating ensemble predictions: {str(e)}")
            return {}
    
    async def _calculate_prediction_confidence(
        self, 
        predictions: Dict[str, Any], 
        features_data: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Calculate overall prediction confidence."""
        try:
            if not predictions:
                return {'overall_confidence': 0.0}
            
            confidences = [pred.get('confidence', 0) for pred in predictions.values()]
            feature_completeness = [len(features) for features in features_data.values()]
            
            overall_confidence = np.mean(confidences) if confidences else 0.0
            avg_feature_completeness = np.mean(feature_completeness) if feature_completeness else 0.0
            
            data_quality_factor = min(avg_feature_completeness / 25, 1.0)  # Assume 25 features is ideal
            adjusted_confidence = overall_confidence * data_quality_factor
            
            return {
                'overall_confidence': adjusted_confidence,
                'average_individual_confidence': overall_confidence,
                'data_quality_factor': data_quality_factor,
                'stocks_predicted': len(predictions),
                'confidence_distribution': {
                    'high_confidence': sum(1 for c in confidences if c > 0.7),
                    'medium_confidence': sum(1 for c in confidences if 0.4 <= c <= 0.7),
                    'low_confidence': sum(1 for c in confidences if c < 0.4)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction confidence: {str(e)}")
            return {'overall_confidence': 0.0}
    
    def _calculate_overall_market_sentiment(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall market sentiment from predictions."""
        try:
            if not predictions:
                return {'sentiment': 'neutral', 'score': 0.0}
            
            total_score = 0
            total_weight = 0
            
            for pred in predictions.values():
                predicted_return = pred.get('predicted_return', 0)
                confidence = pred.get('confidence', 0)
                
                total_score += predicted_return * confidence
                total_weight += confidence
            
            if total_weight > 0:
                weighted_sentiment = total_score / total_weight
            else:
                weighted_sentiment = 0
            
            if weighted_sentiment > 0.02:
                sentiment = 'bullish'
            elif weighted_sentiment < -0.02:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
            
            return {
                'sentiment': sentiment,
                'score': weighted_sentiment,
                'strength': abs(weighted_sentiment) * 10  # Scale to 0-1
            }
            
        except Exception:
            return {'sentiment': 'neutral', 'score': 0.0}
    
    def _convert_rating_to_score(self, rating: str) -> float:
        """Convert analyst rating to numeric score."""
        rating_map = {
            'strong_buy': 1.0,
            'buy': 0.5,
            'hold': 0.0,
            'sell': -0.5,
            'strong_sell': -1.0,
            'neutral': 0.0
        }
        return rating_map.get(rating.lower(), 0.0)
    
    def _get_default_stock_prediction(self, stock: str) -> Dict[str, Any]:
        """Return default prediction for a stock."""
        return {
            'stock': stock,
            'predicted_return': 0.0,
            'direction': 'neutral',
            'strength': 0.0,
            'confidence': 0.1,
            'model_predictions': {'momentum': 0.0, 'sentiment': 0.0, 'technical': 0.0},
            'features_used': 0,
            'prediction_horizon': '1-5 days',
            'error': 'Insufficient data for prediction'
        }
    
    def _get_default_predictions(self) -> Dict[str, Any]:
        """Return default predictions when processing fails."""
        return {
            'individual_predictions': {},
            'ensemble_predictions': {'top_bullish': [], 'top_bearish': [], 'neutral': []},
            'prediction_confidence': {'overall_confidence': 0.0},
            'feature_importance': {},
            'model_performance': {},
            'predicted_at': datetime.now().isoformat(),
            'error': 'Failed to generate predictions'
        }
    
    async def train_model_with_historical_data(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train the ML model with historical data (for future enhancement)."""
        self.logger.info("Model training with historical data not yet implemented")
        return {'status': 'not_implemented'}
    
    async def save_model(self, filepath: str) -> bool:
        """Save trained model to file."""
        try:
            model_data = {
                'models': self.models,
                'scalers': self.scalers,
                'feature_importance': self.feature_importance,
                'model_performance': self.model_performance
            }
            joblib.dump(model_data, filepath)
            return True
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return False
    
    async def load_model(self, filepath: str) -> bool:
        """Load trained model from file."""
        try:
            if os.path.exists(filepath):
                model_data = joblib.load(filepath)
                self.models = model_data.get('models', {})
                self.scalers = model_data.get('scalers', {})
                self.feature_importance = model_data.get('feature_importance', {})
                self.model_performance = model_data.get('model_performance', {})
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
