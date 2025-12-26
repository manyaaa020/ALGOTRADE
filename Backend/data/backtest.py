"""
Backtesting Engine Module
Simulates trades based on signals and tracks portfolio performance.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a completed trade"""
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    shares: int
    side: str
    pnl: float
    pnl_pct: float
    exit_reason: str
    
    def to_dict(self) -> Dict:
        return {
            "entry_date": self.entry_date.isoformat() if hasattr(self.entry_date, 'isoformat') else str(self.entry_date),
            "exit_date": self.exit_date.isoformat() if hasattr(self.exit_date, 'isoformat') else str(self.exit_date),
            "entry_price": round(self.entry_price, 2),
            "exit_price": round(self.exit_price, 2),
            "shares": self.shares,
            "side": self.side,
            "pnl": round(self.pnl, 2),
            "pnl_pct": round(self.pnl_pct * 100, 2),
            "exit_reason": self.exit_reason,
        }


@dataclass 
class BacktestResult:
    """Contains all backtest results"""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)
    metrics: Dict = field(default_factory=dict)


class BacktestEngine:
    """
    Backtesting engine that simulates trading based on signals.
    
    Features:
    - Portfolio simulation with realistic costs
    - Trade logging
    - Equity curve generation
    - Performance metrics calculation
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        position_size_pct: float = 0.1,
        stop_loss_pct: float = 0.02
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.position_size_pct = position_size_pct
        self.stop_loss_pct = stop_loss_pct
        
        # State
        self.cash = initial_capital
        self.position_shares = 0
        self.position_entry_price = 0.0
        self.position_entry_date = None
        self.stop_loss_price = 0.0
        
    def run_backtest(
        self, 
        df: pd.DataFrame, 
        signal_column: str = 'signal_rule'
    ) -> BacktestResult:
        """
        Run backtest on historical data with signals.
        
        Args:
            df: DataFrame with price data and signals
            signal_column: Name of the signal column to use
            
        Returns:
            BacktestResult with trades, equity curve, and metrics
        """
        # Reset state
        self.cash = self.initial_capital
        self.position_shares = 0
        self.position_entry_price = 0.0
        self.position_entry_date = None
        
        trades = []
        equity_curve = []
        
        df = df.copy()
        
        for idx, row in df.iterrows():
            date = row['date']
            close = row['close']
            signal = row.get(signal_column, 0)
            
            # Calculate current portfolio value
            position_value = self.position_shares * close
            portfolio_value = self.cash + position_value
            
            # Check stop-loss if in position
            if self.position_shares > 0 and close <= self.stop_loss_price:
                trade = self._close_position(date, close, "stop_loss")
                if trade:
                    trades.append(trade)
            
            # Process signals
            if signal == 1 and self.position_shares == 0:  # BUY signal
                self._open_position(date, close, portfolio_value)
                
            elif signal == -1 and self.position_shares > 0:  # SELL signal
                trade = self._close_position(date, close, "signal")
                if trade:
                    trades.append(trade)
            
            # Record equity curve
            position_value = self.position_shares * close
            portfolio_value = self.cash + position_value
            
            equity_curve.append({
                "date": date.isoformat() if hasattr(date, 'isoformat') else str(date),
                "portfolio_value": round(portfolio_value, 2),
                "cash": round(self.cash, 2),
                "position_value": round(position_value, 2),
                "close": round(close, 2),
                "signal": signal,
            })
        
        # Close any remaining position at end
        if self.position_shares > 0:
            final_close = df.iloc[-1]['close']
            final_date = df.iloc[-1]['date']
            trade = self._close_position(final_date, final_close, "end_of_backtest")
            if trade:
                trades.append(trade)
        
        # Calculate metrics
        metrics = self._calculate_metrics(df, equity_curve, trades)
        
        result = BacktestResult(
            trades=trades,
            equity_curve=equity_curve,
            metrics=metrics
        )
        
        logger.info(f"Backtest complete: {len(trades)} trades, Final value: ${metrics.get('final_value', 0):,.2f}")
        
        return result
    
    def _open_position(self, date, price: float, portfolio_value: float) -> None:
        """Open a long position"""
        # Apply slippage (buy at slightly higher price)
        execution_price = price * (1 + self.slippage_pct)
        
        # Calculate position size
        max_capital = portfolio_value * self.position_size_pct
        shares = int(max_capital / execution_price)
        
        if shares <= 0:
            return
        
        # Calculate commission
        commission = shares * execution_price * self.commission_pct
        
        # Execute
        cost = shares * execution_price + commission
        self.cash -= cost
        self.position_shares = shares
        self.position_entry_price = execution_price
        self.position_entry_date = date
        self.stop_loss_price = execution_price * (1 - self.stop_loss_pct)
        
        logger.debug(f"BUY {shares} @ ${execution_price:.2f} (commission: ${commission:.2f})")
    
    def _close_position(self, date, price: float, reason: str) -> Optional[Trade]:
        """Close the current position"""
        if self.position_shares <= 0:
            return None
        
        # Apply slippage (sell at slightly lower price)
        execution_price = price * (1 - self.slippage_pct)
        
        # Calculate commission
        commission = self.position_shares * execution_price * self.commission_pct
        
        # Calculate P&L
        gross_proceeds = self.position_shares * execution_price
        net_proceeds = gross_proceeds - commission
        
        entry_cost = self.position_shares * self.position_entry_price
        pnl = net_proceeds - entry_cost
        pnl_pct = (execution_price / self.position_entry_price) - 1
        
        # Create trade record
        trade = Trade(
            entry_date=self.position_entry_date,
            exit_date=date,
            entry_price=self.position_entry_price,
            exit_price=execution_price,
            shares=self.position_shares,
            side='long',
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason
        )
        
        # Update state
        self.cash += net_proceeds
        self.position_shares = 0
        self.position_entry_price = 0.0
        self.position_entry_date = None
        
        logger.debug(f"SELL {trade.shares} @ ${execution_price:.2f} - P&L: ${pnl:.2f} ({reason})")
        
        return trade
    
    def _calculate_metrics(
        self, 
        df: pd.DataFrame,
        equity_curve: List[Dict],
        trades: List[Trade]
    ) -> Dict:
        """Calculate performance metrics"""
        if not equity_curve:
            return {}
        
        # Convert equity curve to series
        values = pd.Series([e['portfolio_value'] for e in equity_curve])
        dates = pd.Series([e['date'] for e in equity_curve])
        
        # Basic metrics
        initial_value = self.initial_capital
        final_value = values.iloc[-1]
        total_return = (final_value / initial_value) - 1
        
        # Calculate daily returns for Sharpe
        daily_returns = values.pct_change().dropna()
        
        # Sharpe ratio (assuming 252 trading days, 0% risk-free rate)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0.0
        
        # Maximum drawdown
        rolling_max = values.expanding().max()
        drawdowns = (values - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
        
        # Find max drawdown dates
        max_dd_idx = drawdowns.idxmin()
        max_dd_end_date = dates.iloc[max_dd_idx] if max_dd_idx < len(dates) else None
        
        # Trade statistics
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Buy and hold comparison
        buy_hold_return = (df.iloc[-1]['close'] / df.iloc[0]['close']) - 1
        
        # Annualized return
        num_days = len(equity_curve)
        years = num_days / 252
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        metrics = {
            "initial_value": initial_value,
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return * 100, 2),
            "annualized_return_pct": round(annualized_return * 100, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown_pct": round(max_drawdown * 100, 2),
            "total_trades": len(trades),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate_pct": round(win_rate * 100, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "buy_hold_return_pct": round(buy_hold_return * 100, 2),
            "alpha_pct": round((total_return - buy_hold_return) * 100, 2),
            "num_trading_days": num_days,
        }
        
        return metrics
    
    def run_comparison_backtest(
        self, 
        df: pd.DataFrame
    ) -> Dict:
        """
        Run backtests for all signal types and compare.
        
        Returns:
            Dictionary with results for each strategy
        """
        results = {}
        
        # Rule-based strategy
        if 'signal_rule' in df.columns:
            rule_result = self.run_backtest(df, 'signal_rule')
            results['rule_based'] = {
                "metrics": rule_result.metrics,
                "trades": [t.to_dict() for t in rule_result.trades],
                "equity_curve": rule_result.equity_curve
            }
        
        # ML strategy
        if 'signal_ml' in df.columns:
            # Reset state for fair comparison
            self.cash = self.initial_capital
            self.position_shares = 0
            
            ml_result = self.run_backtest(df, 'signal_ml')
            results['ml_strategy'] = {
                "metrics": ml_result.metrics,
                "trades": [t.to_dict() for t in ml_result.trades],
                "equity_curve": ml_result.equity_curve
            }
        
        # Buy and hold benchmark
        results['buy_hold'] = self._calculate_buy_hold(df)
        
        return results
    
    def _calculate_buy_hold(self, df: pd.DataFrame) -> Dict:
        """Calculate buy-and-hold benchmark performance"""
        initial_price = df.iloc[0]['close']
        
        # Buy as many shares as possible with initial capital
        shares = int(self.initial_capital / initial_price)
        remaining_cash = self.initial_capital - (shares * initial_price)
        
        equity_curve = []
        for idx, row in df.iterrows():
            portfolio_value = shares * row['close'] + remaining_cash
            equity_curve.append({
                "date": row['date'].isoformat() if hasattr(row['date'], 'isoformat') else str(row['date']),
                "portfolio_value": round(portfolio_value, 2),
                "close": round(row['close'], 2),
            })
        
        final_value = equity_curve[-1]['portfolio_value']
        total_return = (final_value / self.initial_capital) - 1
        
        # Calculate max drawdown
        values = pd.Series([e['portfolio_value'] for e in equity_curve])
        rolling_max = values.expanding().max()
        drawdowns = (values - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
        
        # Sharpe
        daily_returns = values.pct_change().dropna()
        sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
        
        return {
            "metrics": {
                "initial_value": self.initial_capital,
                "final_value": round(final_value, 2),
                "total_return_pct": round(total_return * 100, 2),
                "max_drawdown_pct": round(max_drawdown * 100, 2),
                "sharpe_ratio": round(sharpe, 2),
                "total_trades": 1,
            },
            "equity_curve": equity_curve
        }
