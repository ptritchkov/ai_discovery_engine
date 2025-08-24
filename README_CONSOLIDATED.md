# Consolidated AI Stock Discovery Engine

This repository contains a consolidated and enhanced version of the AI Stock Discovery Engine, combining scattered scripts into a unified, production-ready system with comprehensive backtesting capabilities.

## 🚀 Features

### Core Functionality
- **Enhanced News Collection**: Multi-source RSS feed aggregation from Yahoo Finance, MarketWatch, Reuters, CNN Business, Bloomberg, Financial Times, and Seeking Alpha
- **Advanced LLM Analysis**: Deep stock identification and impact analysis using OpenAI GPT-4
- **Comprehensive Market Data**: Multi-API data collection with Polygon.io, yfinance, and Alpha Vantage
- **ML-Enhanced Predictions**: Ensemble machine learning models for stock movement prediction
- **Investment Recommendations**: Risk-assessed recommendations with position sizing and price targets

### New Consolidated Features
- **Unified Entry Point**: Single `consolidated_main.py` combining best features from scattered scripts
- **Comprehensive Test Suite**: `test_suite.py` with systematic component testing
- **Historical Backtesting**: `backtesting_engine.py` for performance analysis over 1-12 weeks
- **Enhanced Visualizations**: ASCII charts and performance metrics
- **Clean Architecture**: Modular design with proper error handling and logging

## 📁 Project Structure

```
ai_discovery_engine/
├── consolidated_main.py          # Main unified entry point
├── backtesting_engine.py         # Historical performance analysis
├── test_suite.py                 # Comprehensive testing framework
├── src/
│   ├── data_collectors/
│   │   ├── enhanced_news_collector.py    # Multi-source news aggregation
│   │   └── market_data_collector.py      # Multi-API market data
│   ├── analyzers/
│   │   ├── enhanced_llm_analyzer.py      # Advanced LLM analysis
│   │   └── market_analyzer.py            # Market pattern analysis
│   ├── ml_models/
│   │   └── prediction_model.py           # ML prediction ensemble
│   ├── decision_engine/
│   │   └── investment_engine.py          # Investment recommendations
│   └── utils/
│       ├── price_visualizer.py           # Enhanced visualization tools
│       └── config.py                     # Configuration management
└── legacy/                               # Original scattered scripts
    ├── main.py                          # Original main script
    ├── enhanced_main.py                 # Enhanced version
    └── test_*.py                        # Original test scripts
```

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ptritchkov/ai_discovery_engine.git
   cd ai_discovery_engine
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## 🔧 Configuration

### Required API Keys
- `OPENAI_API_KEY`: For LLM analysis
- `POLYGON_API_KEY`: Primary market data source
- `ALPHA_VANTAGE_API_KEY`: Backup market data
- `NEWSAPI_KEY`: News data (optional, RSS feeds used as primary)

### Feature Toggles
- `ENABLE_OPENAI_API=true`: Enable/disable LLM analysis
- `ENABLE_POLYGON_API=true`: Enable/disable Polygon.io
- `ENABLE_YFINANCE_API=true`: Enable/disable yfinance backup
- `PRIORITIZE_POLYGON=true`: Prefer Polygon.io over other sources

## 🚀 Usage

### Run Live Analysis
```bash
python consolidated_main.py
```

### Run Comprehensive Tests
```bash
python test_suite.py
```

### Run Historical Backtesting
```bash
python backtesting_engine.py
```

## 📊 Backtesting Features

The backtesting engine provides comprehensive historical analysis:

- **Historical Simulation**: Simulates running the discovery engine at past dates
- **Performance Metrics**: Success rates, returns, volatility, Sharpe ratio
- **Visual Analysis**: ASCII charts showing performance over time
- **Configurable Periods**: 1 week to 3 months historical analysis
- **Results Export**: JSON export of detailed backtesting results

### Backtesting Example
```python
from backtesting_engine import BacktestingEngine

engine = BacktestingEngine()

# Run 3-month backtest with weekly intervals
results = await engine.run_historical_backtest(
    start_weeks_ago=12,  # 3 months back
    end_weeks_ago=1,     # 1 week back
    interval_days=7      # Weekly intervals
)
```

## 🧪 Testing

The consolidated test suite includes:

1. **Market Data Collection Test**: Validates API connectivity and data quality
2. **News Collection Test**: Verifies multi-source news aggregation
3. **LLM Analysis Test**: Tests stock identification and reasoning
4. **Recommendation Generation Test**: Validates investment logic
5. **End-to-End Pipeline Test**: Complete system integration test

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

- **Recommendation Accuracy**: Success rate of buy/sell recommendations
- **Risk-Adjusted Returns**: Sharpe ratio and volatility analysis
- **Consistency Score**: Percentage of profitable periods
- **Market Correlation**: Performance vs market benchmarks

## 🔄 Migration from Legacy Scripts

The consolidation process merged:

- `main.py` → Core pipeline structure
- `enhanced_main.py` → Advanced LLM analysis and news collection
- `test_*.py` scripts → Unified test suite
- Scattered functionality → Clean modular architecture

### Key Improvements
- ✅ Fixed type annotation errors
- ✅ Unified configuration management
- ✅ Enhanced error handling and logging
- ✅ Comprehensive backtesting capabilities
- ✅ Production-ready architecture

## 🚫 Disabled Features

As requested, the following features are disabled:
- Twitter API integration (due to API issues)
- Polymarket integration (due to API issues)

These can be re-enabled by updating the configuration when API issues are resolved.

## 📝 Example Output

### Live Analysis Results
```
🎯 CONSOLIDATED AI STOCK DISCOVERY ENGINE - ANALYSIS RESULTS
================================================================================

📰 NEWS ANALYSIS SUMMARY
   • Total articles analyzed: 47
   • Stocks identified by AI: 15
   • Deep price analysis performed: 8

🧠 AI-IDENTIFIED INVESTMENT OPPORTUNITIES:
   1. AAPL - Apple Inc.
      Impact: POSITIVE | Score: 8.2/10
      Category: DIRECT | Confidence: 9/10
      Reasoning: Strong quarterly earnings beat expectations with record iPhone sales...

🎯 FINAL INVESTMENT RECOMMENDATIONS:
   1. AAPL - BUY
      Confidence: 87.3%
      Expected Return: +12.4%
      Position Size: medium
      Risk Level: Low
```

### Backtesting Results
```
📊 BACKTESTING RESULTS SUMMARY
================================================================================

📈 PERFORMANCE OVERVIEW:
   • Total Backtest Periods: 12
   • Successful Backtests: 10
   • Success Rate: 83.3%
   • Date Range: 2024-05-24 to 2024-08-16

💰 FINANCIAL PERFORMANCE:
   • Total Recommendations: 156
   • Overall Success Rate: 64.1%
   • Average Daily Return: +2.34%
   • Volatility: 4.12%
   • Sharpe Ratio: 0.57
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Run the test suite: `python test_suite.py`
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- **Repository**: https://github.com/ptritchkov/ai_discovery_engine
- **Issues**: https://github.com/ptritchkov/ai_discovery_engine/issues
- **Documentation**: See README_CONSOLIDATED.md (this file)
