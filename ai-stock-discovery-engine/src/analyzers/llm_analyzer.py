"""LLM analyzer for identifying stocks affected by news and pattern recognition."""

import asyncio
import openai
import os
from datetime import datetime
from typing import List, Dict, Any
import logging
import json
import re

class LLMAnalyzer:
    """Uses LLM for advanced pattern recognition and stock identification."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            self.client = openai.OpenAI(api_key=self.openai_api_key)
        else:
            self.client = None
            self.logger.warning("OpenAI API key not found. LLM analysis will be limited.")
    
    async def identify_affected_stocks(self, news_data: List[Dict[str, Any]]) -> List[str]:
        """
        Use LLM to identify stocks that might be affected by news.
        
        Args:
            news_data: List of news articles
            
        Returns:
            List of stock symbols that might be affected
        """
        self.logger.info(f"Analyzing {len(news_data)} news articles to identify affected stocks...")
        
        if not self.client:
            self.logger.warning("OpenAI client not available. Using fallback method.")
            return self._fallback_stock_identification(news_data)
        
        try:
            all_stocks = set()
            batch_size = 5
            
            for i in range(0, len(news_data), batch_size):
                batch = news_data[i:i + batch_size]
                batch_stocks = await self._analyze_news_batch(batch)
                all_stocks.update(batch_stocks)
                
                await asyncio.sleep(1)
            
            validated_stocks = self._validate_stock_symbols(list(all_stocks))
            
            self.logger.info(f"Identified {len(validated_stocks)} potentially affected stocks")
            return validated_stocks[:50]  # Limit to top 50 stocks
            
        except Exception as e:
            self.logger.error(f"Error in LLM stock identification: {str(e)}")
            return self._fallback_stock_identification(news_data)
    
    async def _analyze_news_batch(self, news_batch: List[Dict[str, Any]]) -> List[str]:
        """Analyze a batch of news articles to identify affected stocks."""
        try:
            news_summaries = []
            for article in news_batch:
                summary = {
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'source': article.get('source', ''),
                    'keyword': article.get('keyword', '')
                }
                news_summaries.append(summary)
            
            prompt = self._create_stock_identification_prompt(news_summaries)
            
            response = self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a financial analyst expert at identifying which stocks might be affected by news events. You understand market dynamics, sector relationships, and supply chain impacts."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=1,
                
            )
            
            stocks = self._parse_stock_response(response.choices[0].message.content)
            return stocks
            
        except Exception as e:
            self.logger.error(f"Error analyzing news batch: {str(e)}")
            return []
    
    def _create_stock_identification_prompt(self, news_summaries: List[Dict[str, Any]]) -> str:
        """Create a prompt for identifying affected stocks."""
        news_text = ""
        for i, news in enumerate(news_summaries, 1):
            news_text += f"\n{i}. Title: {news['title']}\n   Description: {news['description']}\n   Source: {news['source']}\n"
        
        prompt = f"""
Analyze the following news articles and identify publicly traded stocks (with their ticker symbols) that might be affected by these events. Consider:

1. Direct mentions of companies
2. Industry/sector impacts
3. Supply chain effects
4. Regulatory impacts
5. Economic indicators affecting specific sectors
6. Competitive dynamics

News Articles:
{news_text}

Please provide your analysis in the following JSON format:
{{
    "directly_mentioned": ["AAPL", "MSFT"],
    "sector_affected": ["XLF", "XLE"],
    "supply_chain_impact": ["NVDA", "TSM"],
    "competitive_impact": ["GOOGL", "META"],
    "reasoning": {{
        "AAPL": "Mentioned in article about new product launch",
        "XLF": "Banking sector affected by interest rate news"
    }}
}}

Focus on major publicly traded companies with liquid markets. Use standard ticker symbols (e.g., AAPL for Apple, MSFT for Microsoft).
"""
        return prompt
    
    def _parse_stock_response(self, response_text: str) -> List[str]:
        """Parse LLM response to extract stock symbols."""
        try:
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
                
                try:
                    parsed = json.loads(json_text)
                    stocks = []
                    
                    for category in ['directly_mentioned', 'sector_affected', 'supply_chain_impact', 'competitive_impact']:
                        if category in parsed and isinstance(parsed[category], list):
                            stocks.extend(parsed[category])
                    
                    return stocks
                except json.JSONDecodeError:
                    pass
            
            ticker_pattern = r'\b[A-Z]{1,5}\b'
            potential_tickers = re.findall(ticker_pattern, response_text)
            
            common_words = {'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN', 'HER', 'WAS', 'ONE', 'OUR', 'HAD', 'BY', 'UP', 'DO', 'NO', 'IF', 'MY', 'ON', 'AS', 'WE', 'TO', 'BE', 'AT', 'OR', 'IN', 'IS', 'IT', 'OF', 'SO', 'HE', 'HIS', 'SHE', 'HAS', 'AN'}
            
            filtered_tickers = [ticker for ticker in potential_tickers if ticker not in common_words and len(ticker) <= 5]
            
            return filtered_tickers[:20]  # Limit results
            
        except Exception as e:
            self.logger.error(f"Error parsing stock response: {str(e)}")
            return []
    
    def _validate_stock_symbols(self, stocks: List[str]) -> List[str]:
        """Validate and filter stock symbols."""
        known_stocks = {
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'BRK.B', 'UNH', 'JNJ',
            'V', 'WMT', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'ABBV', 'BAC', 'PFE', 'KO', 'AVGO',
            'PEP', 'TMO', 'COST', 'DIS', 'ABT', 'DHR', 'VZ', 'ADBE', 'CMCSA', 'NKE', 'LIN',
            'NFLX', 'CRM', 'XOM', 'T', 'ORCL', 'AMD', 'INTC', 'IBM', 'QCOM', 'TXN', 'HON',
            'UPS', 'LOW', 'SBUX', 'MDT', 'CAT', 'GS', 'AXP', 'BLK', 'GILD', 'AMT', 'ISRG',
            'SPGI', 'BKNG', 'MU', 'ZTS', 'TJX', 'ADP', 'MMM', 'CVS', 'MDLZ', 'TMUS', 'SYK',
            'CI', 'SO', 'DUK', 'PLD', 'CCI', 'NSC', 'D', 'USB', 'WM', 'ITW', 'GE', 'F', 'GM'
        }
        
        etf_symbols = {
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'AGG', 'BND', 'VNQ',
            'XLF', 'XLE', 'XLK', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLRE', 'XLB'
        }
        
        validated = []
        for stock in stocks:
            stock = stock.upper().strip()
            
            if len(stock) >= 1 and len(stock) <= 5 and stock.isalpha():
                validated.append(stock)
            elif stock in known_stocks or stock in etf_symbols:
                validated.append(stock)
        
        seen = set()
        result = []
        for stock in validated:
            if stock not in seen:
                seen.add(stock)
                result.append(stock)
        
        return result
    
    def _fallback_stock_identification(self, news_data: List[Dict[str, Any]]) -> List[str]:
        """Fallback method when LLM is not available."""
        stocks = set()
        
        stock_keywords = {
            'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL', 'amazon': 'AMZN',
            'tesla': 'TSLA', 'meta': 'META', 'facebook': 'META', 'nvidia': 'NVDA',
            'berkshire': 'BRK.B', 'johnson': 'JNJ', 'visa': 'V', 'walmart': 'WMT',
            'jpmorgan': 'JPM', 'procter': 'PG', 'mastercard': 'MA', 'depot': 'HD',
            'chevron': 'CVX', 'pfizer': 'PFE', 'coca': 'KO', 'pepsi': 'PEP',
            'disney': 'DIS', 'netflix': 'NFLX', 'oracle': 'ORCL', 'intel': 'INTC',
            'amd': 'AMD', 'qualcomm': 'QCOM', 'starbucks': 'SBUX', 'boeing': 'BA'
        }
        
        sector_keywords = {
            'bank': ['JPM', 'BAC', 'WFC', 'C'],
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
            'oil': ['XOM', 'CVX', 'COP'],
            'pharma': ['PFE', 'JNJ', 'MRK'],
            'retail': ['WMT', 'TGT', 'COST']
        }
        
        for article in news_data:
            text = f"{article.get('title', '')} {article.get('description', '')}".lower()
            
            for keyword, symbol in stock_keywords.items():
                if keyword in text:
                    stocks.add(symbol)
            
            for sector, symbols in sector_keywords.items():
                if sector in text:
                    stocks.update(symbols)
        
        return list(stocks)[:30]  # Limit results
    
    async def analyze_market_patterns(self, market_data: Dict[str, Any], news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Use LLM to identify patterns that humans might miss.
        
        Args:
            market_data: Market data for stocks
            news_data: News articles
            
        Returns:
            Pattern analysis results
        """
        self.logger.info("Analyzing market patterns with LLM...")
        
        if not self.client:
            return self._fallback_pattern_analysis(market_data, news_data)
        
        try:
            pattern_prompt = self._create_pattern_analysis_prompt(market_data, news_data)
            
            response = self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert quantitative analyst with deep knowledge of market patterns, behavioral finance, and hidden correlations. You excel at finding non-obvious patterns that human analysts might miss."
                    },
                    {
                        "role": "user",
                        "content": pattern_prompt
                    }
                ],
                temperature=1,
                
            )
            
            patterns = self._parse_pattern_response(response.choices[0].message.content)
            
            return {
                'patterns_identified': patterns,
                'analysis_confidence': 0.8,
                'analyzed_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in pattern analysis: {str(e)}")
            return self._fallback_pattern_analysis(market_data, news_data)
    
    def _create_pattern_analysis_prompt(self, market_data: Dict[str, Any], news_data: List[Dict[str, Any]]) -> str:
        """Create prompt for pattern analysis."""
        market_summary = ""
        for stock, data in list(market_data.items())[:10]:  # Limit to avoid token limits
            price_data = data.get('price_data', {})
            technical = data.get('technical_indicators', {})
            
            market_summary += f"\n{stock}: Price change: {price_data.get('price_change_pct', 0):.2f}%, "
            market_summary += f"Volume ratio: {data.get('volume_analysis', {}).get('volume_ratio', 1):.2f}, "
            market_summary += f"RSI: {technical.get('rsi', 50):.1f}"
        
        news_themes = []
        for article in news_data[:5]:
            news_themes.append(f"- {article.get('title', '')}")
        
        prompt = f"""
Analyze the following market data and news to identify hidden patterns, correlations, or opportunities that might not be obvious:

Market Data Summary:
{market_summary}

Recent News Themes:
{chr(10).join(news_themes)}

Please identify:
1. Non-obvious correlations between stocks
2. Unusual volume or price patterns
3. Potential sector rotation signals
4. Hidden opportunities based on news sentiment vs. price action
5. Contrarian indicators
6. Technical pattern divergences

Provide your analysis in JSON format:
{{
    "hidden_correlations": [
        {{"stocks": ["AAPL", "MSFT"], "pattern": "inverse correlation", "confidence": 0.7}}
    ],
    "unusual_patterns": [
        {{"stock": "TSLA", "pattern": "high volume with minimal price movement", "implication": "potential breakout"}}
    ],
    "opportunities": [
        {{"type": "contrarian", "stock": "XYZ", "reasoning": "negative sentiment but strong technicals"}}
    ],
    "sector_signals": [
        {{"from_sector": "tech", "to_sector": "energy", "strength": 0.6}}
    ]
}}
"""
        return prompt
    
    def _parse_pattern_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM pattern analysis response."""
        try:
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
                
                return json.loads(json_text)
            
            return {'error': 'Could not parse pattern analysis'}
            
        except Exception as e:
            self.logger.error(f"Error parsing pattern response: {str(e)}")
            return {'error': str(e)}
    
    def _fallback_pattern_analysis(self, market_data: Dict[str, Any], news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback pattern analysis when LLM is not available."""
        return {
            'patterns_identified': {
                'hidden_correlations': [],
                'unusual_patterns': [],
                'opportunities': [],
                'sector_signals': []
            },
            'analysis_confidence': 0.3,
            'analyzed_at': datetime.now().isoformat(),
            'fallback_mode': True
        }
    
    async def generate_investment_thesis(self, stock: str, all_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive investment thesis for a stock using LLM.
        
        Args:
            stock: Stock symbol
            all_data: Combined data from all sources
            
        Returns:
            Investment thesis and reasoning
        """
        if not self.client:
            return self._fallback_investment_thesis(stock)
        
        try:
            thesis_prompt = self._create_thesis_prompt(stock, all_data)
            
            response = self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a senior investment analyst who creates comprehensive, well-reasoned investment theses. You consider fundamental analysis, technical indicators, sentiment, and market dynamics."
                    },
                    {
                        "role": "user",
                        "content": thesis_prompt
                    }
                ],
                temperature=1,
                
            )
            
            thesis = self._parse_thesis_response(response.choices[0].message.content)
            
            return {
                'stock': stock,
                'thesis': thesis,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error generating investment thesis for {stock}: {str(e)}")
            return self._fallback_investment_thesis(stock)
    
    def _create_thesis_prompt(self, stock: str, all_data: Dict[str, Any]) -> str:
        """Create prompt for investment thesis generation."""
        market_data = all_data.get('market_data', {}).get(stock, {})
        sentiment_data = all_data.get('sentiment_analysis', {}).get('combined_sentiment', {}).get('stock_sentiments', {}).get(stock, {})
        
        prompt = f"""
Create a comprehensive investment thesis for {stock} based on the following data:

Market Data:
- Current Price: ${market_data.get('price_data', {}).get('current_price', 0)}
- Price Change: {market_data.get('price_data', {}).get('price_change_pct', 0):.2f}%
- Volume Trend: {market_data.get('volume_analysis', {}).get('volume_trend', 'normal')}
- RSI: {market_data.get('technical_indicators', {}).get('rsi', 50)}
- Analyst Rating: {market_data.get('fundamentals', {}).get('analyst_rating', {}).get('rating', 'neutral')}

Sentiment Analysis:
- Combined Sentiment Score: {sentiment_data.get('combined_score', 0):.3f}
- Confidence: {sentiment_data.get('confidence', 0):.3f}
- Social Sentiment: {sentiment_data.get('social_sentiment', 0):.3f}
- Market Sentiment: {sentiment_data.get('market_sentiment', 0):.3f}

Please provide:
1. Investment recommendation (Strong Buy/Buy/Hold/Sell/Strong Sell)
2. Price target (if applicable)
3. Key bullish factors
4. Key bearish factors
5. Risk assessment
6. Time horizon
7. Confidence level (0-1)

Format as JSON:
{{
    "recommendation": "Buy",
    "price_target": 150.00,
    "confidence": 0.75,
    "time_horizon": "3-6 months",
    "bullish_factors": ["Strong earnings growth", "Positive sentiment"],
    "bearish_factors": ["High valuation", "Market volatility"],
    "risk_level": "Medium",
    "thesis_summary": "Brief summary of investment thesis"
}}
"""
        return prompt
    
    def _parse_thesis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse investment thesis response."""
        try:
            if '{' in response_text and '}' in response_text:
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                json_text = response_text[json_start:json_end]
                
                return json.loads(json_text)
            
            return {'error': 'Could not parse thesis'}
            
        except Exception as e:
            return {'error': str(e)}
    
    def _fallback_investment_thesis(self, stock: str) -> Dict[str, Any]:
        """Fallback thesis when LLM is not available."""
        return {
            'stock': stock,
            'thesis': {
                'recommendation': 'Hold',
                'confidence': 0.5,
                'thesis_summary': 'Limited analysis available',
                'fallback_mode': True
            },
            'generated_at': datetime.now().isoformat()
        }
