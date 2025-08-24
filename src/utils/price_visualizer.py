"""
Price visualization utilities for displaying historical stock movements.
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta

def create_price_chart(historical_data: List[Dict[str, Any]], width: int = 50) -> str:
    """
    Create a simple ASCII chart of price movements.
    
    Args:
        historical_data: List of historical price data
        width: Width of the chart in characters
        
    Returns:
        ASCII chart as string
    """
    if not historical_data or len(historical_data) < 2:
        return "Insufficient data for chart"
    
    # Extract prices and dates
    prices = [float(day.get('Close', 0)) for day in historical_data]
    dates = [day.get('Date', '') for day in historical_data]
    
    if not prices or min(prices) == max(prices):
        return "No price variation to display"
    
    # Normalize prices to chart height (10 rows)
    min_price = min(prices)
    max_price = max(prices)
    price_range = max_price - min_price
    
    chart_height = 8
    chart = []
    
    # Create chart rows from top to bottom
    for row in range(chart_height):
        line = ""
        threshold = max_price - (row * price_range / (chart_height - 1))
        
        for i, price in enumerate(prices):
            if abs(price - threshold) <= (price_range / (chart_height * 2)):
                line += "●"
            elif price > threshold:
                line += "│"
            else:
                line += " "
        
        # Add price label
        price_label = f"${threshold:.2f}"
        line = f"{price_label:>8} │{line}"
        chart.append(line)
    
    # Add bottom border and dates
    chart.append("         " + "─" * len(prices))
    
    # Add date labels (show first, middle, last)
    if len(dates) >= 3:
        date_line = "         "
        date_line += dates[0][:5]  # First date (MM-DD)
        
        # Add spacing
        middle_pos = len(prices) // 2
        spaces_needed = middle_pos - len(dates[0][:5])
        date_line += " " * max(0, spaces_needed)
        date_line += dates[middle_pos][:5]  # Middle date
        
        # Add spacing for last date
        spaces_needed = len(prices) - len(date_line) + 9 - len(dates[-1][:5])
        date_line += " " * max(0, spaces_needed)
        date_line += dates[-1][:5]  # Last date
        
        chart.append(date_line)
    
    return "\n".join(chart)

def analyze_price_trend(historical_data: List[Dict[str, Any]], days: int = 7) -> Dict[str, Any]:
    """
    Analyze price trends over the specified number of days.
    
    Args:
        historical_data: List of historical price data
        days: Number of recent days to analyze
        
    Returns:
        Dictionary with trend analysis
    """
    if not historical_data or len(historical_data) < 2:
        return {"trend": "insufficient_data", "change": 0, "volatility": 0}
    
    # Get recent data
    recent_data = historical_data[-days:] if len(historical_data) >= days else historical_data
    prices = [float(day.get('Close', 0)) for day in recent_data]
    
    if len(prices) < 2:
        return {"trend": "insufficient_data", "change": 0, "volatility": 0}
    
    # Calculate overall change
    start_price = prices[0]
    end_price = prices[-1]
    total_change = ((end_price - start_price) / start_price * 100) if start_price > 0 else 0
    
    # Calculate volatility (standard deviation of daily changes)
    daily_changes = []
    for i in range(1, len(prices)):
        change = ((prices[i] - prices[i-1]) / prices[i-1] * 100) if prices[i-1] > 0 else 0
        daily_changes.append(change)
    
    if daily_changes:
        avg_change = sum(daily_changes) / len(daily_changes)
        variance = sum((x - avg_change) ** 2 for x in daily_changes) / len(daily_changes)
        volatility = variance ** 0.5
    else:
        volatility = 0
    
    # Determine trend
    if total_change > 2:
        trend = "bullish"
    elif total_change < -2:
        trend = "bearish"
    else:
        trend = "sideways"
    
    # Determine volatility level
    if volatility > 3:
        volatility_level = "high"
    elif volatility > 1:
        volatility_level = "moderate"
    else:
        volatility_level = "low"
    
    return {
        "trend": trend,
        "change": total_change,
        "volatility": volatility,
        "volatility_level": volatility_level,
        "days_analyzed": len(recent_data)
    }

def format_price_summary(historical_data: List[Dict[str, Any]]) -> str:
    """
    Create a formatted summary of recent price movements.
    
    Args:
        historical_data: List of historical price data
        
    Returns:
        Formatted price summary string
    """
    if not historical_data:
        return "No price data available"
    
    trend_analysis = analyze_price_trend(historical_data, 7)
    chart = create_price_chart(historical_data[-14:])  # Last 2 weeks
    
    summary = f"""
   📊 Price Analysis ({trend_analysis['days_analyzed']} days):
   • Trend: {trend_analysis['trend'].title()} ({trend_analysis['change']:+.1f}%)
   • Volatility: {trend_analysis['volatility_level'].title()} ({trend_analysis['volatility']:.1f}%)
   
   📈 Price Chart (Last 2 Weeks):
{chart}
"""
    
    return summary

def create_backtest_performance_chart(backtest_results: List[Dict[str, Any]], width: int = 60) -> str:
    """
    Create a performance chart for backtesting results.
    
    Args:
        backtest_results: List of backtest results with performance data
        width: Width of the chart in characters
        
    Returns:
        ASCII chart showing backtesting performance
    """
    if not backtest_results:
        return "No backtest data available"
    
    # Extract performance data
    chart_data = []
    for result in backtest_results:
        performance = result.get('actual_performance', {})
        backtest_date = result.get('backtest_date', '')
        
        try:
            date_obj = datetime.fromisoformat(backtest_date)
            return_pct = performance.get('average_return', 0) * 100
            
            chart_data.append({
                'Date': date_obj.strftime('%m-%d'),
                'Close': return_pct
            })
        except:
            continue
    
    if not chart_data:
        return "No valid backtest data for charting"
    
    return create_price_chart(chart_data, width)

def create_success_rate_chart(backtest_results: List[Dict[str, Any]], width: int = 60) -> str:
    """
    Create a success rate chart for backtesting results.
    
    Args:
        backtest_results: List of backtest results with performance data
        width: Width of the chart in characters
        
    Returns:
        ASCII chart showing success rates over time
    """
    if not backtest_results:
        return "No backtest data available"
    
    # Extract success rate data
    chart_data = []
    for result in backtest_results:
        performance = result.get('actual_performance', {})
        backtest_date = result.get('backtest_date', '')
        
        try:
            date_obj = datetime.fromisoformat(backtest_date)
            success_rate = performance.get('success_rate', 0) * 100
            
            chart_data.append({
                'Date': date_obj.strftime('%m-%d'),
                'Close': success_rate
            })
        except:
            continue
    
    if not chart_data:
        return "No valid success rate data for charting"
    
    return create_price_chart(chart_data, width)
