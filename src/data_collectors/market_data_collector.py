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
import json
from src.utils.config import config

class MarketDataCollector:
    """Collects market data from various financial APIs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.polygon_enabled = config.is_enabled('polygon')
        self.yfinance_enabled = config.is_enabled('yfinance')
        self.prioritize_polygon = config.PRIORITIZE_POLYGON
        
        self.finnhub_key = os.getenv("FINNHUB_API_KEY")
        self.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.polygon_key = os.getenv("POLYGON_API_KEY")
        
        # Rate limiting counters
        self.alpha_vantage_calls = 0
        self.alpha_vantage_reset_time = datetime.now()
        self.max_alpha_vantage_calls = 5  # 5 calls per minute
        
        if self.finnhub_key:
            self.finnhub_client = finnhub.Client(api_key=self.finnhub_key)
        else:
            self.finnhub_client = None
            
        if self.alpha_vantage_key:
            self.alpha_vantage = TimeSeries(key=self.alpha_vantage_key, output_format='pandas')
        else:
            self.alpha_vantage = None
            
        if self.polygon_enabled and self.polygon_key:
            self.polygon_client = RESTClient(self.polygon_key)
        else:
            self.polygon_client = None
            
        if not (self.polygon_enabled or self.yfinance_enabled):
            self.logger.warning("No market data APIs are enabled. Market data collection will be unavailable.")
    
    async def _check_alpha_vantage_rate_limit(self):
        """Check and enforce Alpha Vantage rate limits (5 calls per minute)."""
        current_time = datetime.now()
        
        # Reset counter if a minute has passed
        if (current_time - self.alpha_vantage_reset_time).total_seconds() >= 60:
            self.alpha_vantage_calls = 0
            self.alpha_vantage_reset_time = current_time
        
        # If we've hit the limit, wait until the minute resets
        if self.alpha_vantage_calls >= self.max_alpha_vantage_calls:
            wait_time = 60 - (current_time - self.alpha_vantage_reset_time).total_seconds()
            if wait_time > 0:
                self.logger.info(f"Alpha Vantage rate limit reached. Waiting {wait_time:.1f} seconds...")
                await asyncio.sleep(wait_time)
                self.alpha_vantage_calls = 0
                self.alpha_vantage_reset_time = datetime.now()
        
        self.alpha_vantage_calls += 1
    
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
        
        if not (self.polygon_enabled or self.yfinance_enabled):
            self.logger.info("Market data APIs are disabled. Returning empty data.")
            return {}
        
        # Prioritize stocks - put major stocks first
        prioritized_stocks = self._prioritize_stocks(stocks)
        
        # Limit to 5 stocks for testing purposes
        max_stocks = 5
        if len(prioritized_stocks) > max_stocks:
            self.logger.info(f"Limiting analysis to top {max_stocks} stocks for testing")
            prioritized_stocks = prioritized_stocks[:max_stocks]
        
        try:
            for i, stock in enumerate(prioritized_stocks):
                try:
                    self.logger.info(f"Processing stock {i+1}/{len(prioritized_stocks)}: {stock}")
                    stock_info = await self._collect_comprehensive_stock_data(stock, timeframe)
                    stock_data[stock] = stock_info
                    
                    # Add delay between requests to be respectful to APIs
                    if i < len(prioritized_stocks) - 1:  # Don't wait after the last one
                        await asyncio.sleep(0.5)  # Increased delay slightly
                    
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
            
            price_data = await self._collect_price_data(symbol, period)
            finnhub_data = await self._collect_finnhub_data(symbol)
            technical_indicators = await self._calculate_technical_indicators(symbol, period)
            
            comprehensive_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'price_data': price_data,
                'fundamentals': finnhub_data,
                'technical_indicators': technical_indicators,
                'market_metrics': await self._calculate_market_metrics(price_data),
                'volatility_analysis': await self._analyze_volatility(price_data),
                'volume_analysis': await self._analyze_volume(price_data),
                'collected_at': datetime.now().isoformat()
            }
            
            return comprehensive_data
            
        except Exception as e:
            self.logger.error(f"Error collecting comprehensive data for {symbol}: {str(e)}")
            return self._get_default_stock_data(symbol)
    
    async def _collect_price_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """Collect price data using preferred API (Polygon.io first, then yfinance, then Alpha Vantage)."""
        # Only try Polygon if it's actually enabled and we have a client
        if self.prioritize_polygon and self.polygon_enabled and self.polygon_client:
            try:
                polygon_data = await self._collect_polygon_data(symbol, period)
                if polygon_data and polygon_data.get('current_price', 0) > 0:
                    self.logger.info(f"Successfully collected data for {symbol} using Polygon.io")
                    return polygon_data
            except Exception as e:
                self.logger.warning(f"Polygon.io failed for {symbol}: {str(e)}, falling back to yfinance")
        
        # Use YFinance if it's enabled
        if self.yfinance_enabled:
            try:
                yfinance_data = await self._collect_yfinance_data(symbol, period)
                if yfinance_data and yfinance_data.get('current_price', 0) > 0:
                    self.logger.info(f"Successfully collected data for {symbol} using YFinance")
                    return yfinance_data
            except Exception as e:
                self.logger.warning(f"YFinance failed for {symbol}: {str(e)}, trying Alpha Vantage")
        
        # Skip Alpha Vantage due to rate limits (25/day is too restrictive)
        # self.logger.info("Skipping Alpha Vantage due to rate limit constraints")
        
        # If we get here, all APIs failed or are disabled
        self.logger.error(f"No available market data APIs for {symbol}")
        return self._get_default_stock_data(symbol)
    
    async def _collect_polygon_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """Collect data using Polygon.io API."""
        try:
            if not self.polygon_client:
                return self._get_default_stock_data(symbol)
            
            end_date = datetime.now().date()
            if period == "7d":
                start_date = end_date - timedelta(days=7)
            else:
                start_date = end_date - timedelta(days=30)
            
            aggs = self.polygon_client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan="day",
                from_=start_date,
                to=end_date
            )
            
            if not aggs or len(aggs) == 0:
                return self._get_default_stock_data(symbol)
            
            latest = aggs[-1]
            first = aggs[0]
            
            current_price = latest.close
            previous_price = first.close
            price_change = current_price - previous_price
            price_change_pct = (price_change / previous_price) * 100 if previous_price > 0 else 0
            
            historical_data = []
            for agg in aggs[-10:]:
                historical_data.append({
                    'Date': datetime.fromtimestamp(agg.timestamp / 1000).strftime('%Y-%m-%d'),
                    'Open': agg.open,
                    'High': agg.high,
                    'Low': agg.low,
                    'Close': agg.close,
                    'Volume': agg.volume
                })
            
            avg_volume = sum(agg.volume for agg in aggs) / len(aggs) if aggs else 0
            
            return {
                'current_price': float(current_price),
                'previous_price': float(previous_price),
                'price_change': float(price_change),
                'price_change_pct': float(price_change_pct),
                'volume': int(latest.volume),
                'avg_volume': float(avg_volume),
                'high_52w': 0.0,
                'low_52w': 0.0,
                'market_cap': 0,
                'pe_ratio': 0.0,
                'dividend_yield': 0.0,
                'beta': 1.0,
                'historical_data': historical_data,
                'data_source': 'polygon'
            }
            
        except Exception as e:
            self.logger.warning(f"Error collecting Polygon data for {symbol}: {str(e)}")
            return self._get_default_stock_data(symbol)
    
    async def _collect_yfinance_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """Collect data using direct Yahoo Finance API to bypass rate limits."""
        try:
            # First try direct Yahoo Finance API
            direct_data = await self._collect_direct_yahoo_data(symbol, period)
            if direct_data and direct_data.get('current_price', 0) > 0:
                return direct_data
            
            # Fallback to yfinance library if direct API fails
            return await self._collect_yfinance_fallback(symbol, period)
            
        except Exception as e:
            self.logger.warning(f"Failed to get ticker '{symbol}' reason: {str(e)}")
            return self._get_default_price_data()
    
    async def _collect_direct_yahoo_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """Collect data using direct Yahoo Finance API calls."""
        try:
            # Add delay to avoid rate limiting
            await asyncio.sleep(0.2)
            
            # Get current quote data
            quote_url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Determine the range parameter based on period - get more data for analysis
            if period == "7d":
                range_param = "1mo"  # Get more historical context
            else:
                range_param = "3mo"  # Get even more for weekly analysis
            
            params = {
                'range': range_param,
                'interval': '1d'
            }
            
            response = requests.get(quote_url, headers=headers, params=params, timeout=10)
            
            if response.status_code != 200:
                self.logger.warning(f"Yahoo API returned status {response.status_code} for {symbol}")
                return self._get_default_price_data()
            
            data = response.json()
            
            if not data.get('chart') or not data['chart'].get('result'):
                self.logger.warning(f"No chart data in Yahoo response for {symbol}")
                return self._get_default_price_data()
            
            result = data['chart']['result'][0]
            meta = result.get('meta', {})
            timestamps = result.get('timestamp', [])
            indicators = result.get('indicators', {})
            quote = indicators.get('quote', [{}])[0] if indicators.get('quote') else {}
            
            if not timestamps or not quote:
                self.logger.warning(f"Missing price data in Yahoo response for {symbol}")
                return self._get_default_price_data()
            
            # Extract price data
            closes = quote.get('close', [])
            opens = quote.get('open', [])
            highs = quote.get('high', [])
            lows = quote.get('low', [])
            volumes = quote.get('volume', [])
            
            # Filter out None values and get valid data points
            valid_data = []
            for i, (ts, close, open_price, high, low, vol) in enumerate(zip(timestamps, closes, opens, highs, lows, volumes)):
                if close is not None and open_price is not None:
                    valid_data.append({
                        'timestamp': ts,
                        'close': close,
                        'open': open_price,
                        'high': high or close,
                        'low': low or close,
                        'volume': vol or 0
                    })
            
            if len(valid_data) < 1:
                return self._get_default_price_data()
            
            # Get current and previous prices
            current_price = float(valid_data[-1]['close'])
            previous_price = float(valid_data[0]['close']) if len(valid_data) > 1 else current_price
            price_change = current_price - previous_price
            price_change_pct = (price_change / previous_price) * 100 if previous_price > 0 else 0
            
            # Calculate volume metrics
            current_volume = int(valid_data[-1]['volume'])
            avg_volume = sum(d['volume'] for d in valid_data) / len(valid_data) if valid_data else 0
            
            # Create historical data - keep all available data for analysis
            historical_data = []
            for d in valid_data:  # Keep all historical data, not just last 10 days
                date_str = datetime.fromtimestamp(d['timestamp']).strftime('%Y-%m-%d')
                historical_data.append({
                    'Date': date_str,
                    'Open': float(d['open']),
                    'High': float(d['high']),
                    'Low': float(d['low']),
                    'Close': float(d['close']),
                    'Volume': int(d['volume'])
                })
            
            return {
                'current_price': current_price,
                'previous_price': previous_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'volume': current_volume,
                'avg_volume': float(avg_volume),
                'high_52w': float(meta.get('fiftyTwoWeekHigh', 0) or 0),
                'low_52w': float(meta.get('fiftyTwoWeekLow', 0) or 0),
                'market_cap': 0,  # Not available in this API
                'pe_ratio': 0,    # Not available in this API
                'dividend_yield': 0,  # Not available in this API
                'beta': 1.0,      # Not available in this API
                'historical_data': historical_data,
                'data_source': 'yahoo_direct'
            }
            
        except Exception as e:
            self.logger.warning(f"Direct Yahoo API failed for {symbol}: {str(e)}")
            return self._get_default_price_data()
    
    async def _collect_yfinance_fallback(self, symbol: str, period: str) -> Dict[str, Any]:
        """Fallback to yfinance library with improved error handling."""
        try:
            # Add delay to avoid rate limiting
            await asyncio.sleep(0.1)
            
            ticker = yf.Ticker(symbol)
            
            # Try to get historical data with retries
            hist = None
            for attempt in range(2):  # Reduced attempts
                try:
                    hist = ticker.history(period=period, timeout=10)
                    if not hist.empty:
                        break
                    await asyncio.sleep(1)
                except Exception as e:
                    self.logger.warning(f"YFinance attempt {attempt + 1} failed for {symbol}: {str(e)}")
                    if attempt < 1:
                        await asyncio.sleep(2)
            
            if hist is None or hist.empty:
                self.logger.warning(f"YFinance failed to get ticker '{symbol}' reason: No price data found")
                return self._get_default_price_data()
            
            # Ensure we have valid price data
            if len(hist) < 1:
                return self._get_default_price_data()
            
            current_price = float(hist['Close'].iloc[-1])
            previous_price = float(hist['Close'].iloc[0]) if len(hist) > 1 else current_price
            price_change = current_price - previous_price
            price_change_pct = (price_change / previous_price) * 100 if previous_price > 0 else 0
            
            # Handle volume data safely
            volume = 0
            avg_volume = 0
            if 'Volume' in hist.columns and not hist['Volume'].empty:
                try:
                    volume = int(hist['Volume'].iloc[-1])
                    avg_volume = float(hist['Volume'].mean())
                except (ValueError, TypeError):
                    volume = 0
                    avg_volume = 0
            
            # Convert historical data safely
            historical_data = []
            try:
                for i, (date, row) in enumerate(hist.tail(10).iterrows()):
                    historical_data.append({
                        'Date': date.strftime('%Y-%m-%d'),
                        'Open': float(row.get('Open', 0)),
                        'High': float(row.get('High', 0)),
                        'Low': float(row.get('Low', 0)),
                        'Close': float(row.get('Close', 0)),
                        'Volume': int(row.get('Volume', 0))
                    })
            except Exception as e:
                self.logger.warning(f"Error converting historical data for {symbol}: {str(e)}")
                historical_data = []
            
            return {
                'current_price': current_price,
                'previous_price': previous_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'volume': volume,
                'avg_volume': avg_volume,
                'high_52w': 0,  # Skip info calls to avoid rate limits
                'low_52w': 0,
                'market_cap': 0,
                'pe_ratio': 0,
                'dividend_yield': 0,
                'beta': 1.0,
                'historical_data': historical_data,
                'data_source': 'yfinance_fallback'
            }
            
        except Exception as e:
            self.logger.warning(f"YFinance fallback failed for {symbol}: {str(e)}")
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
        """Calculate technical indicators using direct API data."""
        try:
            # Get historical data using our direct Yahoo API method
            price_data = await self._collect_direct_yahoo_data(symbol, period)
            
            if not price_data or not price_data.get('historical_data'):
                return {}
            
            historical_data = price_data['historical_data']
            if len(historical_data) < 20:  # Need at least 20 days for MA_20
                return {}
            
            # Convert to pandas DataFrame for calculations
            df = pd.DataFrame(historical_data)
            df['Close'] = pd.to_numeric(df['Close'])
            df = df.sort_values('Date')
            
            # Calculate moving averages
            df['MA_5'] = df['Close'].rolling(window=5).mean()
            df['MA_20'] = df['Close'].rolling(window=20).mean()
            
            # Calculate RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Calculate MACD
            exp1 = df['Close'].ewm(span=12).mean()
            exp2 = df['Close'].ewm(span=26).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9).mean()
            
            # Get current values
            current_price = df['Close'].iloc[-1]
            ma_5 = df['MA_5'].iloc[-1] if not pd.isna(df['MA_5'].iloc[-1]) else current_price
            ma_20 = df['MA_20'].iloc[-1] if not pd.isna(df['MA_20'].iloc[-1]) else current_price
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            current_macd = macd.iloc[-1] if not pd.isna(macd.iloc[-1]) else 0
            current_signal = signal.iloc[-1] if not pd.isna(signal.iloc[-1]) else 0
            
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
                returns_series = pd.Series(returns, dtype=float)
                volatility = float(returns_series.std())
                avg_return = float(returns_series.mean())
                
                return {
                    'volatility_score': volatility,
                    'average_return': avg_return,
                    'volatility_trend': 'high' if volatility > 0.03 else 'low' if volatility < 0.01 else 'moderate',
                    'sharpe_ratio': avg_return / volatility if volatility > 0 else 0
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

    async def _collect_alpha_vantage_data(self, symbol: str, period: str) -> Dict[str, Any]:
        """Collect data using Alpha Vantage API with rate limiting."""
        try:
            if not self.alpha_vantage:
                return self._get_default_stock_data(symbol)
            
            # Check rate limits before making the call
            await self._check_alpha_vantage_rate_limit()
            
            # Alpha Vantage has different period formats
            if period == "7d":
                av_period = "daily"
                outputsize = "compact"  # Last 100 data points
            else:
                av_period = "daily"
                outputsize = "full"     # Full history
            
            # Get daily time series
            data, meta_data = self.alpha_vantage.get_daily(symbol=symbol, outputsize=outputsize)
            
            if data.empty or len(data) < 2:
                return self._get_default_stock_data(symbol)
            
            # Convert to numeric and sort by date
            data['4. close'] = pd.to_numeric(data['4. close'], errors='coerce')
            data = data.sort_index()
            
            # Get latest and previous prices
            current_price = data['4. close'].iloc[-1]
            previous_price = data['4. close'].iloc[0]
            
            if pd.isna(current_price) or pd.isna(previous_price):
                return self._get_default_stock_data(symbol)
            
            price_change = current_price - previous_price
            price_change_pct = (price_change / previous_price) * 100 if previous_price > 0 else 0
            
            # Get volume data if available
            volume = 0
            if '5. volume' in data.columns:
                volume_data = pd.to_numeric(data['5. volume'], errors='coerce')
                volume = int(volume_data.iloc[-1]) if not volume_data.empty else 0
            
            # Create historical data structure
            historical_data = []
            for i, (date, row) in enumerate(data.tail(10).iterrows()):
                historical_data.append({
                    'Date': date.strftime('%Y-%m-%d'),
                    'Open': float(row.get('1. open', 0)),
                    'High': float(row.get('2. high', 0)),
                    'Low': float(row.get('3. low', 0)),
                    'Close': float(row['4. close']),
                    'Volume': int(row.get('5. volume', 0))
                })
            
            return {
                'current_price': float(current_price),
                'previous_price': float(previous_price),
                'price_change': float(price_change),
                'price_change_pct': float(price_change_pct),
                'volume': volume,
                'avg_volume': float(volume),  # Alpha Vantage doesn't provide historical volume easily
                'high_52w': 0.0,  # Alpha Vantage free tier doesn't provide this
                'low_52w': 0.0,
                'market_cap': 0,
                'pe_ratio': 0.0,
                'dividend_yield': 0.0,
                'beta': 1.0,
                'historical_data': historical_data,
                'data_source': 'alpha_vantage'
            }
            
        except Exception as e:
            self.logger.warning(f"Error collecting Alpha Vantage data for {symbol}: {str(e)}")
            return self._get_default_stock_data(symbol)

    def _prioritize_stocks(self, stocks: List[str]) -> List[str]:
        """Prioritize stocks by importance - major stocks first."""
        # Major tech and popular stocks that are more likely to be relevant
        high_priority = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 
            'BRK.B', 'JPM', 'JNJ', 'V', 'WMT', 'PG', 'MA', 'HD', 'UNH'
        ]
        
        # ETFs and sector funds
        medium_priority = [
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'XLF', 'XLE', 'XLK', 'XLV'
        ]
        
        # Split stocks into priority groups
        high_pri_stocks = [s for s in stocks if s in high_priority]
        medium_pri_stocks = [s for s in stocks if s in medium_priority and s not in high_pri_stocks]
        low_pri_stocks = [s for s in stocks if s not in high_priority and s not in medium_priority]
        
        # Return prioritized list
        return high_pri_stocks + medium_pri_stocks + low_pri_stocks
