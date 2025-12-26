"""
Risk Management Module
Handles position sizing, stop-loss, and drawdown monitoring.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents an open trading position"""
    symbol: str
    entry_price: float
    entry_date: datetime
    shares: int
    side: str  # 'long' or 'short'
    stop_loss_price: float
    take_profit_price: Optional[float] = None
    
    def current_pnl(self, current_price: float) -> float:
        """Calculate current unrealized P&L"""
        if self.side == 'long':
            return (current_price - self.entry_price) * self.shares
        else:  # short
            return (self.entry_price - current_price) * self.shares
    
    def current_pnl_pct(self, current_price: float) -> float:
        """Calculate current unrealized P&L percentage"""
        if self.side == 'long':
            return (current_price / self.entry_price) - 1
        else:
            return (self.entry_price / current_price) - 1
    
    def should_stop_loss(self, current_price: float) -> bool:
        """Check if stop-loss should trigger"""
        if self.side == 'long':
            return current_price <= self.stop_loss_price
        else:
            return current_price >= self.stop_loss_price
    
    def should_take_profit(self, current_price: float) -> bool:
        """Check if take-profit should trigger"""
        if self.take_profit_price is None:
            return False
        if self.side == 'long':
            return current_price >= self.take_profit_price
        else:
            return current_price <= self.take_profit_price
    
    def to_dict(self) -> Dict:
        return {
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "entry_date": self.entry_date.isoformat() if isinstance(self.entry_date, datetime) else str(self.entry_date),
            "shares": self.shares,
            "side": self.side,
            "stop_loss_price": self.stop_loss_price,
            "take_profit_price": self.take_profit_price,
        }


@dataclass
class RiskManager:
    """
    Manages trading risk including:
    - Position sizing
    - Stop-loss enforcement
    - Maximum drawdown monitoring
    - Trade rejection if risk limits violated
    """
    
    max_position_size_pct: float = 0.1  # Max 10% of portfolio per trade
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    max_drawdown_pct: float = 0.15  # 15% max drawdown
    max_open_positions: int = 1  # Only 1 position at a time
    
    # State tracking
    open_positions: List[Position] = field(default_factory=list)
    peak_portfolio_value: float = 0.0
    current_drawdown: float = 0.0
    kill_switch_triggered: bool = False
    
    def can_open_position(self, portfolio_value: float) -> tuple:
        """
        Check if a new position can be opened.
        
        Returns:
            (allowed: bool, reason: str)
        """
        # Check kill switch
        if self.kill_switch_triggered:
            return False, "Kill switch is active - trading halted"
        
        # Check max positions
        if len(self.open_positions) >= self.max_open_positions:
            return False, f"Maximum open positions ({self.max_open_positions}) reached"
        
        # Check drawdown
        if self.current_drawdown >= self.max_drawdown_pct:
            return False, f"Maximum drawdown ({self.max_drawdown_pct:.1%}) exceeded"
        
        return True, "OK"
    
    def calculate_position_size(
        self, 
        portfolio_value: float,
        entry_price: float,
        volatility: Optional[float] = None
    ) -> int:
        """
        Calculate the number of shares to trade.
        
        Uses fixed position sizing with optional volatility adjustment.
        """
        # Maximum capital to risk
        max_capital = portfolio_value * self.max_position_size_pct
        
        # Adjust for volatility if provided (reduce size for high volatility)
        if volatility is not None and volatility > 0:
            vol_scalar = min(1.0, 0.2 / volatility)  # Target 20% annualized vol
            max_capital *= vol_scalar
        
        # Calculate shares
        shares = int(max_capital / entry_price)
        
        return max(0, shares)
    
    def create_position(
        self,
        symbol: str,
        entry_price: float,
        entry_date: datetime,
        shares: int,
        side: str = 'long'
    ) -> Position:
        """
        Create a new position with stop-loss and take-profit levels.
        """
        if side == 'long':
            stop_loss_price = entry_price * (1 - self.stop_loss_pct)
            take_profit_price = entry_price * (1 + self.take_profit_pct)
        else:  # short
            stop_loss_price = entry_price * (1 + self.stop_loss_pct)
            take_profit_price = entry_price * (1 - self.take_profit_pct)
        
        position = Position(
            symbol=symbol,
            entry_price=entry_price,
            entry_date=entry_date,
            shares=shares,
            side=side,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )
        
        self.open_positions.append(position)
        logger.info(f"Opened {side} position: {shares} shares of {symbol} at ${entry_price:.2f}")
        
        return position
    
    def close_position(self, position: Position, exit_price: float, reason: str) -> Dict:
        """
        Close a position and calculate realized P&L.
        """
        pnl = position.current_pnl(exit_price)
        pnl_pct = position.current_pnl_pct(exit_price)
        
        # Remove from open positions
        if position in self.open_positions:
            self.open_positions.remove(position)
        
        result = {
            "symbol": position.symbol,
            "side": position.side,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "shares": position.shares,
            "pnl": pnl,
            "pnl_pct": pnl_pct,
            "reason": reason,
        }
        
        logger.info(f"Closed position: {reason} - P&L: ${pnl:.2f} ({pnl_pct:.2%})")
        
        return result
    
    def check_stop_loss(self, current_price: float) -> List[Dict]:
        """
        Check all open positions for stop-loss triggers.
        
        Returns:
            List of positions that were stopped out
        """
        stopped_out = []
        
        for position in self.open_positions.copy():
            if position.should_stop_loss(current_price):
                result = self.close_position(position, current_price, "stop_loss")
                stopped_out.append(result)
        
        return stopped_out
    
    def check_take_profit(self, current_price: float) -> List[Dict]:
        """
        Check all open positions for take-profit triggers.
        """
        profits_taken = []
        
        for position in self.open_positions.copy():
            if position.should_take_profit(current_price):
                result = self.close_position(position, current_price, "take_profit")
                profits_taken.append(result)
        
        return profits_taken
    
    def update_drawdown(self, portfolio_value: float) -> None:
        """
        Update peak value and current drawdown.
        Triggers kill switch if max drawdown exceeded.
        """
        # Update peak
        if portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = portfolio_value
        
        # Calculate drawdown
        if self.peak_portfolio_value > 0:
            self.current_drawdown = (self.peak_portfolio_value - portfolio_value) / self.peak_portfolio_value
        
        # Check kill switch
        if self.current_drawdown >= self.max_drawdown_pct:
            self.kill_switch_triggered = True
            logger.critical(f"KILL SWITCH TRIGGERED: Drawdown {self.current_drawdown:.2%} exceeded max {self.max_drawdown_pct:.2%}")
    
    def reset_kill_switch(self) -> None:
        """Manually reset the kill switch (use with caution)"""
        self.kill_switch_triggered = False
        logger.warning("Kill switch manually reset")
    
    def get_open_positions_value(self, current_price: float) -> float:
        """Calculate total value of open positions"""
        total = 0.0
        for pos in self.open_positions:
            if pos.side == 'long':
                total += pos.shares * current_price
            else:
                # For shorts, value is 2 * entry - current (simplified)
                total += pos.shares * (2 * pos.entry_price - current_price)
        return total
    
    def get_status(self, portfolio_value: float = 0.0) -> Dict:
        """Get current risk status"""
        return {
            "open_positions_count": len(self.open_positions),
            "open_positions": [p.to_dict() for p in self.open_positions],
            "current_drawdown_pct": round(self.current_drawdown * 100, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct * 100, 2),
            "peak_portfolio_value": self.peak_portfolio_value,
            "kill_switch_triggered": self.kill_switch_triggered,
            "can_trade": not self.kill_switch_triggered and len(self.open_positions) < self.max_open_positions,
        }
