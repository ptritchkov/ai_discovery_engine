"""Enhanced LLM analyzer for deep stock analysis and news impact assessment."""

import asyncio
import openai
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging
import json
import re
from dotenv import load_dotenv
import os

load_dotenv()

class EnhancedLLMAnalyzer:
    """Advanced LLM analyzer for comprehensive stock analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv('OPENAI_API_KEY')
        
        if self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
            self.enabled = True
        else:
            self.client = None
            self.enabled = False
            self.logger.warning("OpenAI API key not found. Deep analysis will be unavailable.")
    
    async def analyze_news_and_identify_stocks(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comprehensive news analysis to identify affected stocks with impact assessment.
        
        This is the core AI analysis that:
        1. Analyzes news for market implications
        2. Identifies affected companies (including niche/lesser-known ones)
        3. Assesses potential impact magnitude and direction
        4. Provides reasoning for each stock selection
        """
        if not self.enabled or not news_data:
            return {'stocks': [], 'analysis': {}}
        
        self.logger.info(f"Performing deep LLM analysis on {len(news_data)} news articles...")
        
        try:
            # Process news in batches for comprehensive analysis
            batch_size = 5
            all_stock_analyses = {}
            
            for i in range(0, len(news_data), batch_size):
                batch = news_data[i:i + batch_size]
                batch_analysis = await self._analyze_news_batch_comprehensive(batch)
                
                # Merge results
                for stock, analysis in batch_analysis.items():
                    if stock in all_stock_analyses:
                        # Combine multiple mentions
                        all_stock_analyses[stock]['impact_score'] = max(
                            all_stock_analyses[stock]['impact_score'],
                            analysis['impact_score']
                        )
                        all_stock_analyses[stock]['news_mentions'].extend(analysis['news_mentions'])
                        all_stock_analyses[stock]['reasoning'] += f" {analysis['reasoning']}"
                    else:
                        all_stock_analyses[stock] = analysis
                
                # Rate limiting
                await asyncio.sleep(2)
            
            # Rank stocks by impact potential
            ranked_stocks = self._rank_stocks_by_impact(all_stock_analyses)
            
            self.logger.info(f"Deep analysis identified {len(ranked_stocks)} potentially affected stocks")
            
            return {
                'stocks': ranked_stocks,
                'analysis': all_stock_analyses,
                'total_news_analyzed': len(news_data),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive news analysis: {str(e)}")
            return {'stocks': [], 'analysis': {}}
    
    async def _analyze_news_batch_comprehensive(self, news_batch: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Perform deep analysis on a batch of news articles."""
        
        news_summaries = []
        for article in news_batch:
            summary = {
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'content': article.get('content', ''),
                'source': article.get('source', ''),
                'published_at': article.get('published_at', ''),
                'url': article.get('url', '')
            }
            news_summaries.append(summary)
        
        prompt = self._create_comprehensive_analysis_prompt(news_summaries)
        
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are an expert financial analyst with deep knowledge of:
- Market dynamics and sector relationships
- Supply chain impacts and business dependencies  
- Regulatory and economic policy effects
- Emerging companies and niche market players
- Technical analysis and price movement patterns

Your goal is to identify ALL potentially affected stocks, including lesser-known companies that others might miss. Think beyond obvious connections to find hidden opportunities."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            analysis_text = response.choices[0].message.content
            return self._parse_comprehensive_analysis(analysis_text)
            
        except Exception as e:
            self.logger.error(f"Error in LLM analysis: {str(e)}")
            return {}
    
    def _create_comprehensive_analysis_prompt(self, news_summaries: List[Dict[str, Any]]) -> str:
        """Create a detailed prompt for comprehensive stock analysis."""
        
        news_text = ""
        for i, article in enumerate(news_summaries, 1):
            news_text += f"""
Article {i}:
Title: {article['title']}
Content: {article['description']} {article['content']}
Source: {article['source']}
Published: {article['published_at']}
---
"""
        
        prompt = f"""
Analyze the following news articles and identify ALL stocks that could be affected, including:

1. DIRECTLY MENTIONED companies
2. SUPPLY CHAIN partners and dependencies  
3. COMPETITORS in the same sector
4. REGULATORY/POLICY impacts on industries
5. NICHE PLAYERS that others might overlook
6. EMERGING COMPANIES in affected sectors

{news_text}

For each stock you identify, provide:
- Stock symbol (if publicly traded)
- Company name
- Impact direction (POSITIVE/NEGATIVE/MIXED)
- Impact magnitude (1-10 scale)
- Reasoning for why this stock is affected
- Time horizon (IMMEDIATE/SHORT_TERM/LONG_TERM)
- Confidence level (1-10)

Focus on finding opportunities that others might miss. Think about:
- Second and third-order effects
- Supply chain implications
- Regulatory ripple effects
- Sector rotation possibilities
- Emerging market trends

Return your analysis in this JSON format:
{{
  "STOCK_SYMBOL": {{
    "company_name": "Company Name",
    "impact_direction": "POSITIVE/NEGATIVE/MIXED",
    "impact_magnitude": 7,
    "confidence": 8,
    "time_horizon": "SHORT_TERM",
    "reasoning": "Detailed explanation of why this stock is affected",
    "news_articles": [1, 2],
    "category": "DIRECT/SUPPLY_CHAIN/COMPETITOR/REGULATORY/NICHE"
  }}
}}

Prioritize lesser-known companies and hidden opportunities over obvious large-cap stocks.
"""
        
        return prompt
    
    def _parse_comprehensive_analysis(self, analysis_text: str) -> Dict[str, Dict[str, Any]]:
        """Parse the LLM's comprehensive analysis response."""
        try:
            # Extract JSON from the response
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                json_text = json_match.group()
                analysis_data = json.loads(json_text)
                
                # Convert to our internal format
                stock_analyses = {}
                for symbol, data in analysis_data.items():
                    stock_analyses[symbol] = {
                        'company_name': data.get('company_name', ''),
                        'impact_direction': data.get('impact_direction', 'MIXED'),
                        'impact_score': data.get('impact_magnitude', 5) * data.get('confidence', 5) / 10,
                        'confidence': data.get('confidence', 5),
                        'time_horizon': data.get('time_horizon', 'SHORT_TERM'),
                        'reasoning': data.get('reasoning', ''),
                        'category': data.get('category', 'DIRECT'),
                        'news_mentions': data.get('news_articles', [])
                    }
                
                return stock_analyses
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM analysis: {str(e)}")
        
        # Fallback parsing
        return self._fallback_parse_analysis(analysis_text)
    
    def _fallback_parse_analysis(self, analysis_text: str) -> Dict[str, Dict[str, Any]]:
        """Fallback parsing when JSON parsing fails."""
        stock_analyses = {}
        
        # Look for stock symbols in the text
        stock_pattern = r'\b([A-Z]{1,5})\b'
        potential_symbols = re.findall(stock_pattern, analysis_text)
        
        for symbol in potential_symbols:
            if len(symbol) >= 2 and symbol not in ['THE', 'AND', 'FOR', 'ARE', 'BUT']:
                stock_analyses[symbol] = {
                    'company_name': f'Company {symbol}',
                    'impact_direction': 'MIXED',
                    'impact_score': 5.0,
                    'confidence': 6,
                    'time_horizon': 'SHORT_TERM',
                    'reasoning': f'Mentioned in news analysis for {symbol}',
                    'category': 'DIRECT',
                    'news_mentions': [1]
                }
        
        return stock_analyses
    
    def _rank_stocks_by_impact(self, stock_analyses: Dict[str, Dict[str, Any]]) -> List[str]:
        """Rank stocks by their potential impact score."""
        
        # Calculate composite score
        scored_stocks = []
        for symbol, analysis in stock_analyses.items():
            composite_score = (
                analysis['impact_score'] * 0.4 +
                analysis['confidence'] * 0.3 +
                len(analysis['news_mentions']) * 0.2 +
                (10 if analysis['category'] == 'NICHE' else 5) * 0.1  # Boost niche stocks
            )
            
            scored_stocks.append((symbol, composite_score))
        
        # Sort by score and return top stocks
        scored_stocks.sort(key=lambda x: x[1], reverse=True)
        return [symbol for symbol, score in scored_stocks[:20]]  # Return top 20
    
    async def analyze_stock_price_impact(self, stock: str, news_data: List[Dict[str, Any]], 
                                       market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze how news has affected stock price movement and predict future impact.
        
        This performs the deep analysis of:
        1. Historical price movement correlation with news
        2. Market reaction patterns
        3. Future price movement prediction
        4. Risk assessment
        """
        if not self.enabled:
            return {}
        
        try:
            stock_market_data = market_data.get(stock, {})
            price_data = stock_market_data.get('price_data', {})
            historical_data = price_data.get('historical_data', [])
            
            # Create comprehensive analysis prompt
            prompt = self._create_price_impact_prompt(stock, news_data, historical_data)
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a quantitative analyst expert in:
- Price action analysis and technical patterns
- News sentiment impact on stock prices
- Market microstructure and trading behavior
- Risk assessment and volatility analysis
- Institutional vs retail investor behavior

Analyze the correlation between news events and price movements to predict future impact."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=1500
            )
            
            analysis = response.choices[0].message.content
            return self._parse_price_impact_analysis(analysis)
            
        except Exception as e:
            self.logger.error(f"Error in price impact analysis for {stock}: {str(e)}")
            return {}
    
    def _create_price_impact_prompt(self, stock: str, news_data: List[Dict[str, Any]], 
                                  historical_data: List[Dict[str, Any]]) -> str:
        """Create prompt for price impact analysis."""
        
        # Summarize recent news about this stock
        relevant_news = []
        for article in news_data:
            content = f"{article.get('title', '')} {article.get('description', '')}".lower()
            if stock.lower() in content:
                relevant_news.append(article)
        
        news_summary = ""
        for article in relevant_news[:5]:
            news_summary += f"- {article.get('title', '')}\n"
        
        # Summarize price data
        price_summary = ""
        if historical_data:
            recent_prices = historical_data[-10:]  # Last 10 days
            for day in recent_prices:
                price_summary += f"Date: {day.get('date', '')}, Close: ${day.get('close', 0):.2f}, Volume: {day.get('volume', 0):,}\n"
        
        prompt = f"""
Analyze the price impact for {stock} based on recent news and price movements:

RECENT NEWS ABOUT {stock}:
{news_summary}

RECENT PRICE DATA:
{price_summary}

Provide analysis on:
1. How has recent news affected the stock price?
2. What patterns do you see in price/volume reaction to news?
3. What is the likely future price impact (next 1-4 weeks)?
4. What are the key risk factors?
5. What price levels should investors watch?

Return your analysis in JSON format:
{{
  "news_impact_assessment": "How news has affected price",
  "price_reaction_pattern": "Pattern analysis",
  "future_price_prediction": {{
    "direction": "UP/DOWN/SIDEWAYS",
    "magnitude": "percentage expected move",
    "timeframe": "1-4 weeks",
    "confidence": "1-10 scale"
  }},
  "key_price_levels": {{
    "support": "price level",
    "resistance": "price level",
    "target": "price target"
  }},
  "risk_factors": ["list of risks"],
  "opportunity_score": "1-10 scale"
}}
"""
        
        return prompt
    
    def _parse_price_impact_analysis(self, analysis_text: str) -> Dict[str, Any]:
        """Parse price impact analysis from LLM response."""
        try:
            json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback
        return {
            'news_impact_assessment': 'Analysis unavailable',
            'future_price_prediction': {
                'direction': 'SIDEWAYS',
                'magnitude': '0%',
                'confidence': 5
            },
            'opportunity_score': 5
        }
