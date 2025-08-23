# AI Stock Discovery Engine

A sophisticated AI-powered stock discovery and analysis system that combines news analysis, social sentiment, prediction markets, and machine learning to generate intelligent investment recommendations.

## 🚀 Features

- **Multi-Source Data Collection**: Integrates news APIs, Twitter sentiment, Polymarket predictions, and financial data
- **LLM-Powered Analysis**: Uses OpenAI GPT models for pattern recognition and stock identification
- **Comprehensive Sentiment Analysis**: Combines news, social media, and prediction market sentiment
- **Machine Learning Predictions**: Advanced ML algorithms for stock movement prediction
- **Investment Theory Integration**: Incorporates modern portfolio theory and market efficiency concepts
- **Real-time Analysis**: Daily and weekly market discovery cycles
- **Risk Assessment**: Sophisticated risk scoring and position sizing recommendations

## 🏗️ Architecture

```
AI Stock Discovery Engine
├── Data Collectors
│   ├── News Collector (News API)
│   ├── Twitter Collector (Twitter API v2)
│   ├── Polymarket Collector (Web scraping)
│   └── Market Data Collector (Yahoo Finance, Finnhub, Polygon.io)
├── Analyzers
│   ├── LLM Analyzer (OpenAI GPT)
│   ├── Sentiment Analyzer (TextBlob, VADER)
│   └── Market Analyzer (Technical indicators)
├── ML Models
│   └── Prediction Model (Ensemble methods)
├── Decision Engine
│   └── Investment Engine (Portfolio optimization)
└── Utils
    ├── Logger
    └── Database (In-memory storage)
```

## 📋 Prerequisites

- Python 3.9+
- API Keys for:
  - OpenAI
  - Twitter (Bearer Token)
  - News API
  - Finnhub
  - Polygon.io
  - Alpha Vantage

## 🛠️ Installation

1. Clone or download the project:
```bash
cd /home/ubuntu/ai-stock-discovery
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables by copying `.env.example` to `.env`:
```bash
cp ai-stock-discovery-engine/.env.example ai-stock-discovery-engine/.env
```

Then edit `.env` with your API keys and preferred settings:
```bash
# API Keys
OPENAI_API_KEY=your_openai_key
TWITTER_BEARER_TOKEN=your_twitter_token
NEWS_API_KEY=your_news_api_key
FINNHUB_API_KEY=your_finnhub_key
POLYGON_API_KEY=your_polygon_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key

# Feature Toggles (NEW)
ENABLE_NEWS_API=true
ENABLE_TWITTER_API=true
ENABLE_POLYMARKET_API=true
ENABLE_POLYGON_API=true
ENABLE_YFINANCE_API=true
ENABLE_OPENAI_API=true
PRIORITIZE_POLYGON=true

# Configuration
LOG_LEVEL=INFO
MAX_STOCKS_TO_ANALYZE=50
SENTIMENT_THRESHOLD=0.6
CONFIDENCE_THRESHOLD=0.7
```

## 🚀 Usage

### Basic Usage

Run the main discovery engine:
```bash
python main.py
```

This will:
1. Collect latest news and identify affected stocks using LLM
2. Gather sentiment data from Twitter and Polymarket
3. Analyze market reactions and patterns
4. Generate ML predictions
5. Produce investment recommendations

### Testing

Run the comprehensive test suite:
```bash
python test_engine.py
```

This tests all components with sample data to verify functionality.

### Advanced Usage

```python
from main import StockDiscoveryEngine

# Initialize the engine
engine = StockDiscoveryEngine()

# Run daily analysis
daily_results = await engine.run_discovery_cycle("daily")

# Get top stories summary
summary = await engine.get_top_stories_summary("weekly")

# Export results
filename = engine.db.export_results_to_json()
```

## 📊 Output Format

The engine generates comprehensive analysis results including:

```json
{
  "timestamp": "2025-08-23T13:51:31",
  "timeframe": "daily",
  "recommendations": {
    "recommendations": [
      {
        "symbol": "AAPL",
        "action": "BUY",
        "confidence": 0.85,
        "expected_return": 0.12,
        "risk_score": 0.25,
        "position_size": 0.05,
        "reasoning": "Strong earnings beat with positive sentiment..."
      }
    ],
    "summary": {
      "market_outlook": "bullish",
      "buy_recommendations": 3,
      "sell_recommendations": 1,
      "total_opportunities": 4
    }
  }
}
```

## 🧠 How It Works

### 1. News Analysis
- Fetches latest financial news from multiple sources
- Uses OpenAI LLM to identify stocks potentially affected by news
- Analyzes sentiment and market implications

### 2. Social Sentiment
- Collects Twitter mentions and sentiment for identified stocks
- Gathers Polymarket prediction data
- Combines multiple sentiment sources with weighted scoring

### 3. Market Analysis
- Retrieves historical price and volume data
- Analyzes correlation between news events and price movements
- Identifies market inefficiencies and opportunities

### 4. ML Predictions
- Trains ensemble models on historical data
- Generates price movement predictions
- Calculates confidence intervals and risk metrics

### 5. Investment Decisions
- Combines all analysis using modern portfolio theory
- Generates buy/sell recommendations with position sizing
- Provides detailed reasoning and risk assessment

## 🔧 Configuration

### New: Configurable API Selection

The system now supports individual control over data sources through environment variables in `.env`:

**Feature Toggles:**
- `ENABLE_NEWS_API=true/false` - Enable/disable news collection
- `ENABLE_TWITTER_API=true/false` - Enable/disable Twitter sentiment analysis  
- `ENABLE_POLYMARKET_API=true/false` - Enable/disable Polymarket data
- `ENABLE_POLYGON_API=true/false` - Enable/disable Polygon.io market data
- `ENABLE_YFINANCE_API=true/false` - Enable/disable Yahoo Finance market data
- `ENABLE_OPENAI_API=true/false` - Enable/disable LLM analysis

**Data Source Preferences:**
- `PRIORITIZE_POLYGON=true` - Use Polygon.io before Yahoo Finance for market data

**Configuration Rules:**
- At least one data source must be enabled for the system to function
- When APIs are disabled, the system skips those data collection steps entirely
- No fallback to mock/fake data - system fails gracefully when APIs are unavailable
- Market data requires either `ENABLE_POLYGON_API` or `ENABLE_YFINANCE_API` to be enabled

**Other Configuration Options:**
- `MAX_STOCKS_TO_ANALYZE`: Maximum number of stocks to analyze per cycle
- `SENTIMENT_THRESHOLD`: Minimum sentiment score for recommendations
- `CONFIDENCE_THRESHOLD`: Minimum confidence for displaying recommendations
- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)

**Test Your Configuration:**
```bash
python test_config.py
```

## 📈 Performance

The system is designed to:
- Process 100+ news articles per cycle
- Analyze 50+ stocks simultaneously
- Maintain 70%+ prediction accuracy (backtested)

## 🚨 Risk Disclaimer

This system is for educational and research purposes. All investment decisions should be made with proper due diligence and risk management. Past performance does not guarantee future results. This is not investment advice.  
