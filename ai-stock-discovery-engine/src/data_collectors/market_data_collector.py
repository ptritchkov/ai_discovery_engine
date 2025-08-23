"""Market data collector using various financial APIs."""

import asyncio
import yfinance as yf
import requests
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
import pandas as pd
import finnhub
from alpha_vantage.timeseries import TimeSeries
from polygon import RESTClient

class MarketDataCollector:
    """Collects market data from various financial APIs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.finnhub_key = os.getenv("FINNHUB_API_KEY")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.polygon_key = os.getenv("POLYGON_API_KEY")
        
        if self.finnhub_key:
            self.finnhub_client = finnhub.Client(api_key=self.finnhub_key)
        else:
            self.finnhub_client = None
            
        if self.alpha_vantage_key:
            self.alpha_vantage = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
        else:
            self.alpha_vantage = None
            
        if self.polygon_key:
            self.polygon_client = RESTClient(self.polygon_key)
        else:
            self.polygon_client = None
    
    async def collect_stock_data(self, stocks: List[str], timeframe: str = "daily") -> Dict[str, Any]:
        """
        Collect comprehensive stock data for given symbols.
        
        Args:
            stocks: List of stock symbols to analyze
            timeframe: "daily" or "weekly"
            
        Returns:
            Dictionary containing market data for each stock
        """
        self.logger.info(f"Collecting market data for {len(stocks)} stocks...")
        
        stock_data = {}
        
        try:
            for stock in stocks[:20]:  # Limit to avoid rate limits
                try:
                    stock_info = await self._collect_comprehensive_stock_data(stock, timeframe)
                    stock_data[stock] = stock_info
                    
                    await asyncio.sleep(0.2)
                    
                except Exception as e:
                    self.logger.warning(f"Error collecting data for {stock}: {str(e)}")
                    stock_data[stock] = self._get_default_stock_data(stock)
                    
        except Exception as e:
            self.logger.error(f"Error in market data collection: {str(e)}")
            
        self.logger.info(f"Collected market data for {len(stock_data)} stocks")
        return stock_data
    
    async def _collect_comprehensive_stock_data(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Collect comprehensive data for a single stock."""
        try:
            if timeframe == "daily":
                start_date = datetime.now() - timedelta(days=7)
                period = "7d"
            else:  # weekly
                start_date = datetime.now() - timedelta(days=30)
                period = "1mo"
            
            yfinance_data = await self._collect_yfinance_data(symbol, period)
            finnhub_data = await self._collect_finnhub_data(symbol)
            technical_indicators = await self._calculate_technical_indicators(symbol, period)
            
            comprehensive_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'price_data': yfinance_data,
                'fundamentals': finnhub_data,
                'technical_indicators': technical_indicators,
                'market_metrics': await self._calculate_market_metrics(yfinance_data),
                'volatility_analysis': await self._analyze_volatility(yfinance_data),
                'volume_analysis': await self._analyze_volume(yfinance_data),
                'collected_at': datetime.now().isoformat()
            }
            
            return comprehensive_data
            
        except Exception as e:
            self.logger.error(f"Error collecting comprehensive data for {symbol}: {str(e)}")
            return self._get_default_stock_data(symbol)
    
    async def _collect_yfinance_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """Collect data using yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            
            hist = ticker.history(period=period)
            
            if hist.empty:
                return self._get_default_price_data()
            
            info = ticker.info
            
            current_price = hist['Close'].iloc[-1]
            previous_price = hist['Close'].iloc[0]
            price_change = current_price - previous_price
            price_change_pct = (price_change / previous_price) * 100
            
            return {
                'current_price': float(current_price),
                'previous_price': float(previous_price),
                'price_change': float(price_change),
                'price_change_pct': float(price_change_pct),
                'volume': int(hist['Volume'].iloc[-1]) if not hist['Volume'].empty else 0,
                'avg_volume': float(hist['Volume'].mean()),
                'high_52w': float(info.get('fiftyTwoWeekHigh', 0)),
                'low_52w': float(info.get('fiftyTwoWeekLow', 0)),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 1.0),
                'historical_data': hist.to_dict('records')[-10:]  # Last 10 days
            }
            
        except Exception as e:
            self.logger.warning(f"Error collecting yfinance data for {symbol}: {str(e)}")
            return self._get_default_price_data()
    
    async def _collect_finnhub_data(self, symbol: str) -> Dict[str, Any]:
        """Collect fundamental data using Finnhub."""
        try:
            if not self.finnhub_client:
                return self._get_default_fundamentals()
            
            profile = self.finnhub_client.company_profile2(symbol=symbol)
            
            financials = self.finnhub_client.company_basic_financials(symbol, 'all')
            
            recommendations = self.finnhub_client.recommendation_trends(symbol)
            
            return {
                'company_name': profile.get('name', ''),
                'industry': profile.get('finnhubIndustry', ''),
                'sector': profile.get('ggroup', ''),
                'country': profile.get('country', ''),
                'market_cap_finnhub': profile.get('marketCapitalization', 0),
                'shares_outstanding': profile.get('shareOutstanding', 0),
                'financials': financials.get('metric', {}),
                'recommendations': recommendations[:5] if recommendations else [],
                'analyst_rating': self._calculate_analyst_rating(recommendations)
            }
            
        except Exception as e:
            self.logger.warning(f"Error collecting Finnhub data for {symbol}: {str(e)}")
            return self._get_default_fundamentals()
    
    async def _calculate_technical_indicators(self, symbol: str, period: str) -> Dict[str, Any]:
        """Calculate technical indicators."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return {}
            
            hist['MA_5'] = hist['Close'].rolling(window=5).mean()
            hist['MA_20'] = hist['Close'].rolling(window=20).mean()
            
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            exp1 = hist['Close'].ewm(span=12).mean()
            exp2 = hist['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            
            current_price = hist['Close'].iloc[-1]
            ma_5 = hist['MA_5'].iloc[-1] if not hist['MA_5'].isna().iloc[-1] else current_price
            ma_20 = hist['MA_20'].iloc[-1] if not hist['MA_20'].isna().iloc[-1] else current_price
            current_rsi = rsi.iloc[-1] if not rsi.isna().iloc[-1] else 50
            current_macd = macd.iloc[-1] if not macd.isna().iloc[-1] else 0
            current_signal = signal.iloc[-1] if not signal.isna().iloc[-1] else 0
            
            return {
                'ma_5': float(ma_5),
                'ma_20': float(ma_20),
                'rsi': float(current_rsi),
                'macd': float(current_macd),
                'macd_signal': float(current_signal),
                'price_vs_ma5': float((current_price - ma_5) / ma_5 * 100),
                'price_vs_ma20': float((current_price - ma_20) / ma_20 * 100),
                'rsi_signal': 'overbought' if current_rsi > 70 else 'oversold' if current_rsi < 30 else 'neutral',
                'macd_signal_trend': 'bullish' if current_macd > current_signal else 'bearish'
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating technical indicators for {symbol}: {str(e)}")
            return {}
    
    async def _calculate_market_metrics(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate additional market metrics."""
        try:
            current_price = price_data.get('current_price', 0)
            high_52w = price_data.get('high_52w', current_price)
            low_52w = price_data.get('low_52w', current_price)
            
            if high_52w != low_52w:
                position_in_range = (current_price - low_52w) / (high_52w - low_52w)
            else:
                position_in_range = 0.5
            
            return {
                'position_in_52w_range': float(position_in_range),
                'distance_from_52w_high': float((high_52w - current_price) / high_52w * 100),
                'distance_from_52w_low': float((current_price - low_52w) / low_52w * 100),
                'momentum_score': self._calculate_momentum_score(price_data)
            }
            
        except Exception as e:
            self.logger.warning(f"Error calculating market metrics: {str(e)}")
            return {}
    
    async def _analyze_volatility(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze price volatility."""
        try:
            historical_data = price_data.get('historical_data', [])
            
            if len(historical_data) < 2:
                return {'volatility_score': 0.5, 'volatility_trend': 'stable'}
            
            prices = [day['Close'] for day in historical_data]
            returns = []
            
            for i in range(1, len(prices)):
                daily_return = (prices[i] - prices[i-1]) / prices[i-1]
                returns.append(daily_return)
            
            if returns:
                volatility = pd.Series(returns).std()
                avg_return = pd.Series(returns).mean()
                
                return {
                    'volatility_score': float(volatility),
                    'average_return': float(avg_return),
                    'volatility_trend': 'high' if volatility > 0.03 else 'low' if volatility < 0.01 else 'moderate',
                    'sharpe_ratio': float(avg_return / volatility) if volatility > 0 else 0
                }
            
            return {'volatility_score': 0.5, 'volatility_trend': 'stable'}
            
        except Exception as e:
            self.logger.warning(f"Error analyzing volatility: {str(e)}")
            return {}
    
    async def _analyze_volume(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze trading volume patterns."""
        try:
            current_volume = price_data.get('volume', 0)
            avg_volume = price_data.get('avg_volume', current_volume)
            
            if avg_volume > 0:
                volume_ratio = current_volume / avg_volume
                volume_trend = 'high' if volume_ratio > 1.5 else 'low' if volume_ratio < 0.5 else 'normal'
            else:
                volume_ratio = 1.0
                volume_trend = 'normal'
            
            return {
                'current_volume': current_volume,
                'average_volume': avg_volume,
                'volume_ratio': float(volume_ratio),
                'volume_trend': volume_trend,
                'volume_score': min(volume_ratio, 3.0) / 3.0  # Normalize to 0-1
            }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing volume: {str(e)}")
            return {}
    
    def _calculate_momentum_score(self, price_data: Dict[str, Any]) -> float:
        """Calculate momentum score based on price movement."""
        try:
            price_change_pct = price_data.get('price_change_pct', 0)
            volume_ratio = price_data.get('volume', 1) / max(price_data.get('avg_volume', 1), 1)
            
            momentum = (price_change_pct / 100) * min(volume_ratio, 2.0)
            
            return max(-1.0, min(1.0, momentum))
            
        except Exception as e:
            return 0.0
    
    def _calculate_analyst_rating(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall analyst rating from recommendations."""
        if not recommendations:
            return {'rating': 'neutral', 'score': 0.0}
        
        try:
            latest = recommendations[0]
            
            buy = latest.get('buy', 0)
            hold = latest.get('hold', 0)
            sell = latest.get('sell', 0)
            strong_buy = latest.get('strongBuy', 0)
            strong_sell = latest.get('strongSell', 0)
            
            total = buy + hold + sell + strong_buy + strong_sell
            
            if total == 0:
                return {'rating': 'neutral', 'score': 0.0}
            
            score = (strong_buy * 2 + buy * 1 + hold * 0 + sell * (-1) + strong_sell * (-2)) / total
            
            if score > 0.5:
                rating = 'strong_buy'
            elif score > 0:
                rating = 'buy'
            elif score > -0.5:
                rating = 'hold'
            elif score > -1:
                rating = 'sell'
            else:
                rating = 'strong_sell'
            
            return {'rating': rating, 'score': float(score)}
            
        except Exception as e:
            return {'rating': 'neutral', 'score': 0.0}
    
    def _get_default_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Return default stock data when collection fails."""
        return {
            'symbol': symbol,
            'timeframe': 'daily',
            'price_data': self._get_default_price_data(),
            'fundamentals': self._get_default_fundamentals(),
            'technical_indicators': {},
            'market_metrics': {},
            'volatility_analysis': {},
            'volume_analysis': {},
            'collected_at': datetime.now().isoformat(),
            'error': 'Failed to collect data'
        }
    
    def _get_default_price_data(self) -> Dict[str, Any]:
        """Return default price data."""
        return {
            'current_price': 0.0,
            'previous_price': 0.0,
            'price_change': 0.0,
            'price_change_pct': 0.0,
            'volume': 0,
            'avg_volume': 0,
            'high_52w': 0.0,
            'low_52w': 0.0,
            'market_cap': 0,
            'pe_ratio': 0.0,
            'dividend_yield': 0.0,
            'beta': 1.0,
            'historical_data': []
        }
    
    def _get_default_fundamentals(self) -> Dict[str, Any]:
        """Return default fundamental data."""
        return {
            'company_name': '',
            'industry': '',
            'sector': '',
            'country': '',
            'market_cap_finnhub': 0,
            'shares_outstanding': 0,
            'financials': {},
            'recommendations': [],
            'analyst_rating': {'rating': 'neutral', 'score': 0.0}
        }
