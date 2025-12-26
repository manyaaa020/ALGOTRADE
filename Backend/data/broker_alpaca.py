"""
Alpaca Broker Interface Module
Handles communication with Alpaca API for paper and live trading.
ALL credentials from environment variables - NEVER hardcoded.
"""
import os
import logging
from typing import Optional, Dict, List
from datetime import datetime

logger = logging.getLogger(__name__)


class AlpacaBroker:
    """
    Alpaca broker interface for paper and live trading.
    
    IMPORTANT NOTES:
    - All API keys MUST come from environment variables
    - Paper trading uses: https://paper-api.alpaca.markets
    - Live trading uses: https://api.alpaca.markets
    - Default is ALWAYS paper trading for safety
    """
    
    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        paper: bool = True
    ):
        """
        Initialize Alpaca broker connection.
        
        Args:
            api_key: Alpaca API key (from env var)
            secret_key: Alpaca secret key (from env var)
            paper: If True, use paper trading (default)
        """
        # Get credentials from args or environment
        self.api_key = api_key or os.environ.get('ALPACA_API_KEY')
        self.secret_key = secret_key or os.environ.get('ALPACA_SECRET_KEY')
        self.paper = paper
        
        # Set base URL
        self.base_url = self.PAPER_URL if paper else self.LIVE_URL
        
        # Track initialization status
        self._initialized = False
        self._api = None
        
        # Validate credentials exist
        if not self.api_key or not self.secret_key:
            logger.warning("Alpaca credentials not provided. Broker will not be functional.")
            return
        
        # Initialize API connection
        self._initialize_api()
    
    def _initialize_api(self) -> None:
        """Initialize the Alpaca API connection"""
        try:
            # Import here to make dependency optional
            from alpaca.trading.client import TradingClient
            from alpaca.trading.requests import (
                MarketOrderRequest,
                LimitOrderRequest,
                StopOrderRequest,
                StopLimitOrderRequest
            )
            from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
            
            self._api = TradingClient(
                api_key=self.api_key,
                secret_key=self.secret_key,
                paper=self.paper
            )
            
            # Store request classes for order creation
            self._MarketOrderRequest = MarketOrderRequest
            self._LimitOrderRequest = LimitOrderRequest
            self._StopOrderRequest = StopOrderRequest
            self._StopLimitOrderRequest = StopLimitOrderRequest
            self._OrderSide = OrderSide
            self._TimeInForce = TimeInForce
            self._OrderType = OrderType
            
            self._initialized = True
            mode = "PAPER" if self.paper else "LIVE"
            logger.info(f"Alpaca API initialized in {mode} mode")
            
        except ImportError:
            logger.error("alpaca-py library not installed. Run: pip install alpaca-py")
            self._initialized = False
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca API: {e}")
            self._initialized = False
    
    def is_connected(self) -> bool:
        """Check if broker is connected"""
        return self._initialized and self._api is not None
    
    def get_account(self) -> Dict:
        """
        Get account information.
        
        Returns:
            Account details including buying power, equity, etc.
        """
        if not self.is_connected():
            return {"error": "Broker not connected"}
        
        try:
            account = self._api.get_account()
            return {
                "id": account.id,
                "status": account.status.value if hasattr(account.status, 'value') else str(account.status),
                "currency": account.currency,
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "last_equity": float(account.last_equity),
                "long_market_value": float(account.long_market_value),
                "short_market_value": float(account.short_market_value),
                "pattern_day_trader": account.pattern_day_trader,
                "trading_blocked": account.trading_blocked,
                "transfers_blocked": account.transfers_blocked,
                "account_blocked": account.account_blocked,
                "mode": "PAPER" if self.paper else "LIVE"
            }
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return {"error": str(e)}
    
    def get_positions(self) -> Dict:
        """
        Get all current positions.
        
        Returns:
            List of positions with P&L information
        """
        if not self.is_connected():
            return {"error": "Broker not connected"}
        
        try:
            positions = self._api.get_all_positions()
            
            position_list = []
            for pos in positions:
                position_list.append({
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "side": pos.side.value if hasattr(pos.side, 'value') else str(pos.side),
                    "market_value": float(pos.market_value),
                    "cost_basis": float(pos.cost_basis),
                    "unrealized_pl": float(pos.unrealized_pl),
                    "unrealized_plpc": float(pos.unrealized_plpc),
                    "current_price": float(pos.current_price),
                    "avg_entry_price": float(pos.avg_entry_price),
                })
            
            return {
                "positions": position_list,
                "count": len(position_list),
                "mode": "PAPER" if self.paper else "LIVE"
            }
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {"error": str(e)}
    
    def submit_order(
        self,
        symbol: str,
        qty: int,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "day"
    ) -> Dict:
        """
        Submit a trading order.
        
        Args:
            symbol: Stock symbol
            qty: Number of shares
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            limit_price: Limit price for limit/stop_limit orders
            stop_price: Stop price for stop/stop_limit orders
            time_in_force: 'day', 'gtc', 'ioc', etc.
            
        Returns:
            Order details
        """
        if not self.is_connected():
            return {"error": "Broker not connected"}
        
        try:
            # Convert side string to enum
            order_side = self._OrderSide.BUY if side.lower() == 'buy' else self._OrderSide.SELL
            
            # Convert time in force
            tif = self._TimeInForce.DAY
            if time_in_force.lower() == 'gtc':
                tif = self._TimeInForce.GTC
            elif time_in_force.lower() == 'ioc':
                tif = self._TimeInForce.IOC
            
            # Create appropriate order request
            if order_type == "market":
                order_request = self._MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif
                )
            elif order_type == "limit" and limit_price:
                order_request = self._LimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    limit_price=limit_price
                )
            elif order_type == "stop" and stop_price:
                order_request = self._StopOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    stop_price=stop_price
                )
            elif order_type == "stop_limit" and limit_price and stop_price:
                order_request = self._StopLimitOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=order_side,
                    time_in_force=tif,
                    limit_price=limit_price,
                    stop_price=stop_price
                )
            else:
                return {"error": f"Invalid order type or missing prices: {order_type}"}
            
            # Submit order
            order = self._api.submit_order(order_request)
            
            return {
                "id": order.id,
                "client_order_id": order.client_order_id,
                "symbol": order.symbol,
                "qty": str(order.qty),
                "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
                "type": order.type.value if hasattr(order.type, 'value') else str(order.type),
                "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
                "created_at": order.created_at.isoformat() if order.created_at else None,
                "mode": "PAPER" if self.paper else "LIVE"
            }
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return {"error": str(e)}
    
    def cancel_order(self, order_id: str) -> Dict:
        """Cancel a specific order by ID"""
        if not self.is_connected():
            return {"error": "Broker not connected"}
        
        try:
            self._api.cancel_order_by_id(order_id)
            return {
                "status": "cancelled",
                "order_id": order_id,
                "mode": "PAPER" if self.paper else "LIVE"
            }
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return {"error": str(e)}
    
    def cancel_all_orders(self) -> Dict:
        """Cancel all open orders"""
        if not self.is_connected():
            return {"error": "Broker not connected"}
        
        try:
            self._api.cancel_orders()
            return {
                "status": "all_cancelled",
                "mode": "PAPER" if self.paper else "LIVE"
            }
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return {"error": str(e)}
    
    def get_open_orders(self) -> List[Dict]:
        """Get all open orders"""
        if not self.is_connected():
            return []
        
        try:
            orders = self._api.get_orders()
            
            order_list = []
            for order in orders:
                order_list.append({
                    "id": order.id,
                    "symbol": order.symbol,
                    "qty": str(order.qty),
                    "side": order.side.value if hasattr(order.side, 'value') else str(order.side),
                    "type": order.type.value if hasattr(order.type, 'value') else str(order.type),
                    "status": order.status.value if hasattr(order.status, 'value') else str(order.status),
                    "created_at": order.created_at.isoformat() if order.created_at else None,
                })
            
            return order_list
        except Exception as e:
            logger.error(f"Error getting open orders: {e}")
            return []
    
    def close_position(self, symbol: str) -> Dict:
        """Close all positions for a symbol"""
        if not self.is_connected():
            return {"error": "Broker not connected"}
        
        try:
            self._api.close_position(symbol)
            return {
                "status": "closed",
                "symbol": symbol,
                "mode": "PAPER" if self.paper else "LIVE"
            }
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return {"error": str(e)}
    
    def close_all_positions(self) -> Dict:
        """Close all open positions"""
        if not self.is_connected():
            return {"error": "Broker not connected"}
        
        try:
            self._api.close_all_positions()
            return {
                "status": "all_closed",
                "mode": "PAPER" if self.paper else "LIVE"
            }
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return {"error": str(e)}
    
    def get_status(self) -> Dict:
        """Get broker connection status"""
        return {
            "connected": self.is_connected(),
            "mode": "PAPER" if self.paper else "LIVE",
            "base_url": self.base_url,
            "has_credentials": bool(self.api_key and self.secret_key)
        }
