"""
Execution Module
Handles order execution based on trading mode.
Routes orders to appropriate handler (backtest, paper, or live).
NO strategy logic here - only execution.
"""
import logging
from typing import Optional, Dict
from datetime import datetime
from enum import Enum

from .config import Config, TradingMode
from .broker_alpaca import AlpacaBroker

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class ExecutionEngine:
    """
    Execution engine that routes orders based on trading mode.
    
    IMPORTANT: This module contains NO strategy logic.
    It only executes orders given to it.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.broker: Optional[AlpacaBroker] = None
        
        # Initialize broker for paper/live modes
        if config.mode in [TradingMode.PAPER, TradingMode.LIVE]:
            self._initialize_broker()
    
    def _initialize_broker(self) -> None:
        """Initialize broker connection"""
        if not self.config.alpaca_api_key or not self.config.alpaca_secret_key:
            logger.warning("Alpaca credentials not configured. Paper/Live trading disabled.")
            return
        
        try:
            self.broker = AlpacaBroker(
                api_key=self.config.alpaca_api_key,
                secret_key=self.config.alpaca_secret_key,
                paper=self.config.is_paper_mode()
            )
            logger.info(f"Broker initialized in {'PAPER' if self.config.is_paper_mode() else 'LIVE'} mode")
        except Exception as e:
            logger.error(f"Failed to initialize broker: {e}")
            self.broker = None
    
    def execute_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Dict:
        """
        Execute an order based on current trading mode.
        
        Args:
            symbol: Trading symbol (e.g., 'SPY')
            side: BUY or SELL
            quantity: Number of shares
            order_type: Type of order
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            
        Returns:
            Order result dictionary
        """
        # Check kill switch
        if not self.config.is_trading_enabled():
            return {
                "status": "rejected",
                "reason": "Kill switch is active",
                "mode": self.config.mode.value
            }
        
        # Validate quantity
        if quantity <= 0:
            return {
                "status": "rejected",
                "reason": "Invalid quantity",
                "mode": self.config.mode.value
            }
        
        # Route based on mode
        if self.config.is_backtest_mode():
            return self._execute_backtest_order(symbol, side, quantity, order_type)
        
        elif self.config.is_paper_mode() or self.config.is_live_mode():
            return self._execute_broker_order(
                symbol, side, quantity, order_type, limit_price, stop_price
            )
        
        return {
            "status": "error",
            "reason": "Unknown trading mode",
            "mode": self.config.mode.value
        }
    
    def _execute_backtest_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType
    ) -> Dict:
        """
        Simulate order execution for backtesting.
        In backtest mode, orders are recorded but not actually executed.
        """
        return {
            "status": "simulated",
            "mode": "BACKTEST",
            "symbol": symbol,
            "side": side.value,
            "quantity": quantity,
            "order_type": order_type.value,
            "timestamp": datetime.now().isoformat(),
            "message": "Order simulated (backtest mode)"
        }
    
    def _execute_broker_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: int,
        order_type: OrderType,
        limit_price: Optional[float],
        stop_price: Optional[float]
    ) -> Dict:
        """
        Execute order through broker API.
        """
        if self.broker is None:
            return {
                "status": "error",
                "reason": "Broker not initialized",
                "mode": self.config.mode.value
            }
        
        try:
            # Execute through broker
            result = self.broker.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side.value,
                order_type=order_type.value,
                limit_price=limit_price,
                stop_price=stop_price
            )
            
            return {
                "status": "submitted",
                "mode": self.config.mode.value,
                "order_id": result.get("id"),
                "symbol": symbol,
                "side": side.value,
                "quantity": quantity,
                "order_type": order_type.value,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return {
                "status": "error",
                "reason": str(e),
                "mode": self.config.mode.value
            }
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel an open order"""
        if self.config.is_backtest_mode():
            return {
                "status": "simulated",
                "mode": "BACKTEST",
                "message": "Order cancellation simulated"
            }
        
        if self.broker is None:
            return {
                "status": "error",
                "reason": "Broker not initialized"
            }
        
        try:
            result = self.broker.cancel_order(order_id)
            return {
                "status": "cancelled",
                "order_id": order_id,
                "result": result
            }
        except Exception as e:
            return {
                "status": "error",
                "reason": str(e)
            }
    
    def cancel_all_orders(self) -> Dict:
        """Cancel all open orders"""
        if self.config.is_backtest_mode():
            return {
                "status": "simulated",
                "mode": "BACKTEST",
                "message": "All orders cancellation simulated"
            }
        
        if self.broker is None:
            return {
                "status": "error",
                "reason": "Broker not initialized"
            }
        
        try:
            result = self.broker.cancel_all_orders()
            return {
                "status": "all_cancelled",
                "result": result
            }
        except Exception as e:
            return {
                "status": "error",
                "reason": str(e)
            }
    
    def get_open_orders(self) -> Dict:
        """Get all open orders"""
        if self.config.is_backtest_mode():
            return {
                "mode": "BACKTEST",
                "orders": []
            }
        
        if self.broker is None:
            return {
                "status": "error",
                "reason": "Broker not initialized"
            }
        
        try:
            orders = self.broker.get_open_orders()
            return {
                "mode": self.config.mode.value,
                "orders": orders
            }
        except Exception as e:
            return {
                "status": "error",
                "reason": str(e)
            }
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        if self.config.is_backtest_mode():
            return {
                "mode": "BACKTEST",
                "message": "Backtest mode - no real account"
            }
        
        if self.broker is None:
            return {
                "status": "error",
                "reason": "Broker not initialized"
            }
        
        try:
            return self.broker.get_account()
        except Exception as e:
            return {
                "status": "error",
                "reason": str(e)
            }
    
    def get_positions(self) -> Dict:
        """Get current positions"""
        if self.config.is_backtest_mode():
            return {
                "mode": "BACKTEST",
                "positions": []
            }
        
        if self.broker is None:
            return {
                "status": "error",
                "reason": "Broker not initialized"
            }
        
        try:
            return self.broker.get_positions()
        except Exception as e:
            return {
                "status": "error",
                "reason": str(e)
            }
    
    def get_status(self) -> Dict:
        """Get execution engine status"""
        return {
            "mode": self.config.mode.value,
            "kill_switch": not self.config.is_trading_enabled(),
            "broker_connected": self.broker is not None,
            "trading_enabled": self.config.is_trading_enabled() and (
                self.config.is_backtest_mode() or self.broker is not None
            )
        }
