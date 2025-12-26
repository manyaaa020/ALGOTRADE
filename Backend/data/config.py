"""
Configuration module for the algorithmic trading system.
All settings are centralized here for easy modification.
"""
import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class TradingMode(Enum):
    BACKTEST = "BACKTEST"
    PAPER = "PAPER"
    LIVE = "LIVE"


@dataclass
class RiskConfig:
    """Risk management parameters"""
    max_position_size: float = 0.1  # 10% of portfolio per trade
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit (optional)
    max_drawdown_pct: float = 0.15  # 15% max drawdown before kill switch
    max_open_positions: int = 1  # Only 1 position at a time
    

@dataclass
class BacktestConfig:
    """Backtesting parameters"""
    initial_capital: float = 100000.0
    commission_pct: float = 0.001  # 0.1% commission
    slippage_pct: float = 0.0005  # 0.05% slippage


@dataclass
class StrategyConfig:
    """Strategy parameters"""
    short_ma_window: int = 20  # Short moving average
    long_ma_window: int = 50  # Long moving average
    rsi_period: int = 14
    rsi_oversold: int = 30
    rsi_overbought: int = 70
    volatility_window: int = 20


class Config:
    """Main configuration class"""
    
    def __init__(self):
        # Trading mode - defaults to BACKTEST for safety
        mode_str = os.environ.get('TRADING_MODE', 'BACKTEST')
        self.mode = TradingMode(mode_str)
        
        # KILL SWITCH - CRITICAL SAFETY FEATURE
        # When True, all trading is halted
        self.kill_switch: bool = os.environ.get('KILL_SWITCH', 'False').lower() == 'true'
        
        # Asset configuration
        self.symbol: str = os.environ.get('TRADING_SYMBOL', 'SPY')
        self.data_years: int = int(os.environ.get('DATA_YEARS', '10'))
        
        # Sub-configurations
        self.risk = RiskConfig()
        self.backtest = BacktestConfig()
        self.strategy = StrategyConfig()
        
        # Alpaca credentials (only for PAPER/LIVE modes)
        self.alpaca_api_key: Optional[str] = os.environ.get('ALPACA_API_KEY')
        self.alpaca_secret_key: Optional[str] = os.environ.get('ALPACA_SECRET_KEY')
        self.alpaca_base_url: str = os.environ.get(
            'ALPACA_BASE_URL', 
            'https://paper-api.alpaca.markets'  # Paper trading by default
        )
    
    def is_trading_enabled(self) -> bool:
        """Check if trading is allowed"""
        return not self.kill_switch
    
    def is_live_mode(self) -> bool:
        """Check if in live trading mode"""
        return self.mode == TradingMode.LIVE
    
    def is_paper_mode(self) -> bool:
        """Check if in paper trading mode"""
        return self.mode == TradingMode.PAPER
    
    def is_backtest_mode(self) -> bool:
        """Check if in backtest mode"""
        return self.mode == TradingMode.BACKTEST
    
    def to_dict(self) -> dict:
        """Export config as dictionary"""
        return {
            "mode": self.mode.value,
            "kill_switch": self.kill_switch,
            "symbol": self.symbol,
            "data_years": self.data_years,
            "risk": {
                "max_position_size": self.risk.max_position_size,
                "stop_loss_pct": self.risk.stop_loss_pct,
                "take_profit_pct": self.risk.take_profit_pct,
                "max_drawdown_pct": self.risk.max_drawdown_pct,
                "max_open_positions": self.risk.max_open_positions,
            },
            "backtest": {
                "initial_capital": self.backtest.initial_capital,
                "commission_pct": self.backtest.commission_pct,
                "slippage_pct": self.backtest.slippage_pct,
            },
            "strategy": {
                "short_ma_window": self.strategy.short_ma_window,
                "long_ma_window": self.strategy.long_ma_window,
                "rsi_period": self.strategy.rsi_period,
                "rsi_oversold": self.strategy.rsi_oversold,
                "rsi_overbought": self.strategy.rsi_overbought,
                "volatility_window": self.strategy.volatility_window,
            }
        }


# Global config instance
config = Config()
