"""Backtesting visualization utilities using matplotlib."""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import os

class BacktestVisualizer:
    """Creates comprehensive visualizations for backtesting results."""
    
    def __init__(self):
        plt.style.use('default')
        
    def create_portfolio_performance_chart(self, portfolio_history: List[Dict], 
                                         save_path: Optional[str] = None) -> str:
        """Create portfolio value over time chart."""
        if not portfolio_history:
            return ""
            
        dates = [datetime.fromisoformat(p['date']) for p in portfolio_history]
        values = [p['total_value'] for p in portfolio_history]
        initial_value = values[0] if values else 10000
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        ax1.plot(dates, values, linewidth=2, color='#2E86AB', label='Portfolio Value')
        ax1.axhline(y=initial_value, color='gray', linestyle='--', alpha=0.7, label='Initial Value')
        ax1.set_title('Portfolio Performance Over Time', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        returns = [(v - initial_value) / initial_value * 100 for v in values]
        colors = ['green' if r >= 0 else 'red' for r in returns]
        ax2.bar(dates, returns, color=colors, alpha=0.7)
        ax2.set_title('Portfolio Returns (%)', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Return (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            save_path = f'backtest_portfolio_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
    
    def create_trade_analysis_chart(self, trades: List[Dict], save_path: Optional[str] = None) -> str:
        """Create trade success analysis charts."""
        if not trades:
            return ""
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        outcomes = [t.get('profit_loss', 0) for t in trades if 'profit_loss' in t]
        if outcomes:
            wins = [o for o in outcomes if o > 0]
            losses = [o for o in outcomes if o < 0]
            
            ax1.hist([wins, losses], bins=20, label=['Wins', 'Losses'], 
                    color=['green', 'red'], alpha=0.7)
            ax1.set_title('Trade Outcome Distribution', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Profit/Loss ($)')
            ax1.set_ylabel('Number of Trades')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        confidence_ranges = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
        success_rates = []
        range_labels = []
        
        for min_conf, max_conf in confidence_ranges:
            range_trades = [t for t in trades if min_conf <= t.get('confidence', 0) < max_conf]
            if range_trades:
                successful = sum(1 for t in range_trades if t.get('profit_loss', 0) > 0)
                success_rate = successful / len(range_trades) * 100
                success_rates.append(success_rate)
                range_labels.append(f'{min_conf:.1f}-{max_conf:.1f}')
            else:
                success_rates.append(0)
                range_labels.append(f'{min_conf:.1f}-{max_conf:.1f}')
        
        ax2.bar(range_labels, success_rates, color='skyblue', alpha=0.8)
        ax2.set_title('Success Rate by Confidence Level', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Confidence Range')
        ax2.set_ylabel('Success Rate (%)')
        ax2.grid(True, alpha=0.3)
        
        monthly_data = {}
        for trade in trades:
            if 'entry_date' in trade:
                try:
                    date = datetime.fromisoformat(trade['entry_date'])
                    month_key = date.strftime('%Y-%m')
                    if month_key not in monthly_data:
                        monthly_data[month_key] = []
                    monthly_data[month_key].append(trade.get('profit_loss', 0))
                except:
                    continue
        
        if monthly_data:
            months = sorted(monthly_data.keys())
            monthly_returns = [sum(monthly_data[month]) for month in months]
            
            colors = ['green' if r >= 0 else 'red' for r in monthly_returns]
            ax3.bar(months, monthly_returns, color=colors, alpha=0.7)
            ax3.set_title('Monthly Trading Performance', fontsize=14, fontweight='bold')
            ax3.set_xlabel('Month')
            ax3.set_ylabel('Total P&L ($)')
            ax3.grid(True, alpha=0.3)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        stock_performance = {}
        for trade in trades:
            symbol = trade.get('symbol', 'Unknown')
            profit_loss = trade.get('profit_loss', 0)
            if symbol not in stock_performance:
                stock_performance[symbol] = []
            stock_performance[symbol].append(profit_loss)
        
        if stock_performance:
            stock_totals = {symbol: sum(profits) for symbol, profits in stock_performance.items()}
            top_stocks = sorted(stock_totals.items(), key=lambda x: x[1], reverse=True)[:10]
            
            if top_stocks:
                symbols, profits = zip(*top_stocks)
                colors = ['green' if p >= 0 else 'red' for p in profits]
                ax4.barh(symbols, profits, color=colors, alpha=0.7)
                ax4.set_title('Top 10 Stock Performance', fontsize=14, fontweight='bold')
                ax4.set_xlabel('Total P&L ($)')
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            save_path = f'backtest_trade_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
    
    def create_summary_dashboard(self, backtest_results: Dict, save_path: Optional[str] = None) -> str:
        """Create comprehensive summary dashboard."""
        fig = plt.figure(figsize=(16, 12))
        
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        ax_metrics = fig.add_subplot(gs[0, :])
        ax_metrics.axis('off')
        
        performance = backtest_results.get('performance_analysis', {})
        portfolio_perf = performance.get('portfolio_performance', {})
        
        metrics_text = f"""
        BACKTESTING SUMMARY DASHBOARD
        
        Total Return: {portfolio_perf.get('total_return_pct', 0):+.2f}%
        Final Portfolio Value: ${portfolio_perf.get('final_value', 0):,.2f}
        Total Trades: {portfolio_perf.get('total_trades', 0)}
        Success Rate: {portfolio_perf.get('success_rate', 0):.1%}
        Sharpe Ratio: {portfolio_perf.get('sharpe_ratio', 0):.2f}
        Max Drawdown: {portfolio_perf.get('max_drawdown', 0):.2f}%
        """
        
        ax_metrics.text(0.5, 0.5, metrics_text, transform=ax_metrics.transAxes,
                       fontsize=14, ha='center', va='center',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.suptitle('AI Stock Discovery Engine - Backtesting Results', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            save_path = f'backtest_summary_dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
