"""
Algorithmic Trading System API
FastAPI backend for the trading dashboard.
"""
from fastapi import FastAPI, APIRouter, HTTPException, Query
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime, timezone
import asyncio

# Load environment
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Import trading modules
from src.config import Config, TradingMode
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.signals import SignalGenerator, Signal
from src.risk import RiskManager
from src.backtest import BacktestEngine
from src.execution import ExecutionEngine, OrderSide, OrderType

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app
app = FastAPI(
    title="Algo Trading System",
    description="Educational algorithmic trading system for backtesting and paper trading",
    version="1.0.0"
)

# Create router with /api prefix
api_router = APIRouter(prefix="/api")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize trading components
config = Config()
data_loader = DataLoader(symbol=config.symbol)
feature_engineer = FeatureEngineer(
    short_ma_window=config.strategy.short_ma_window,
    long_ma_window=config.strategy.long_ma_window,
    rsi_period=config.strategy.rsi_period,
    volatility_window=config.strategy.volatility_window
)
signal_generator = SignalGenerator(
    rsi_oversold=config.strategy.rsi_oversold,
    rsi_overbought=config.strategy.rsi_overbought
)
risk_manager = RiskManager(
    max_position_size_pct=config.risk.max_position_size,
    stop_loss_pct=config.risk.stop_loss_pct,
    take_profit_pct=config.risk.take_profit_pct,
    max_drawdown_pct=config.risk.max_drawdown_pct,
    max_open_positions=config.risk.max_open_positions
)
backtest_engine = BacktestEngine(
    initial_capital=config.backtest.initial_capital,
    commission_pct=config.backtest.commission_pct,
    slippage_pct=config.backtest.slippage_pct,
    position_size_pct=config.risk.max_position_size,
    stop_loss_pct=config.risk.stop_loss_pct
)
execution_engine = ExecutionEngine(config)

# Cache for processed data
_cached_data = None
_cached_results = None


# Pydantic models for API
class ConfigUpdate(BaseModel):
    symbol: Optional[str] = None
    short_ma_window: Optional[int] = None
    long_ma_window: Optional[int] = None
    stop_loss_pct: Optional[float] = None
    position_size_pct: Optional[float] = None


class OrderRequest(BaseModel):
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: int
    order_type: str = "market"
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None


# ================== API ENDPOINTS ==================

@api_router.get("/")
async def root():
    """API health check"""
    return {
        "message": "Algo Trading System API",
        "version": "1.0.0",
        "mode": config.mode.value,
        "kill_switch": config.kill_switch
    }


@api_router.get("/config")
async def get_config():
    """Get current system configuration"""
    return config.to_dict()


@api_router.get("/status")
async def get_system_status():
    """Get overall system status"""
    return {
        "mode": config.mode.value,
        "kill_switch": config.kill_switch,
        "symbol": config.symbol,
        "data_loaded": _cached_data is not None,
        "backtest_ready": _cached_results is not None,
        "execution_status": execution_engine.get_status(),
        "risk_status": risk_manager.get_status(),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@api_router.post("/data/load")
async def load_market_data(years: int = 10, symbol: str = "SPY"):
    """Load historical market data"""
    global _cached_data, data_loader
    
    try:
        data_loader = DataLoader(symbol=symbol)
        df = data_loader.fetch_data(years=years)
        
        # Calculate features
        df = feature_engineer.calculate_all_features(df)
        
        _cached_data = df
        
        return {
            "status": "success",
            "symbol": symbol,
            "data_info": data_loader.to_dict(),
            "features_calculated": len(feature_engineer.get_feature_columns()),
            "total_rows": len(df)
        }
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/data/summary")
async def get_data_summary():
    """Get summary of loaded data"""
    if _cached_data is None:
        return {"status": "no_data", "message": "Call /data/load first"}
    
    df = _cached_data
    latest = df.iloc[-1]
    
    return {
        "symbol": data_loader.symbol,
        "total_days": len(df),
        "date_range": {
            "start": df.iloc[0]['date'].isoformat(),
            "end": df.iloc[-1]['date'].isoformat()
        },
        "latest": {
            "date": latest['date'].isoformat(),
            "close": round(float(latest['close']), 2),
            "sma_short": round(float(latest.get('sma_short', 0)), 2),
            "sma_long": round(float(latest.get('sma_long', 0)), 2),
            "rsi": round(float(latest.get('rsi', 0)), 2),
            "volatility": round(float(latest.get('volatility', 0)), 4),
        }
    }


@api_router.get("/data/prices")
async def get_price_data(limit: int = Query(default=252, le=2520)):
    """Get recent price data for charting"""
    if _cached_data is None:
        raise HTTPException(status_code=400, detail="No data loaded. Call /data/load first")
    
    df = _cached_data.tail(limit)
    
    prices = []
    for _, row in df.iterrows():
        prices.append({
            "date": row['date'].isoformat(),
            "open": round(float(row['open']), 2),
            "high": round(float(row['high']), 2),
            "low": round(float(row['low']), 2),
            "close": round(float(row['close']), 2),
            "volume": int(row['volume']),
            "sma_short": round(float(row.get('sma_short', 0)), 2) if row.get('sma_short') else None,
            "sma_long": round(float(row.get('sma_long', 0)), 2) if row.get('sma_long') else None,
            "rsi": round(float(row.get('rsi', 0)), 2) if row.get('rsi') else None,
        })
    
    return {"prices": prices, "count": len(prices)}


@api_router.post("/signals/generate")
async def generate_signals(train_ml: bool = True, ml_model: str = "logistic"):
    """Generate trading signals using rule-based and ML strategies"""
    global _cached_data
    
    if _cached_data is None:
        raise HTTPException(status_code=400, detail="No data loaded. Call /data/load first")
    
    try:
        df = _cached_data.copy()
        
        # Generate rule-based signals
        df = signal_generator.generate_rule_based_signals(df)
        
        # Train and generate ML signals
        ml_metrics = None
        if train_ml:
            X, y, indices = feature_engineer.prepare_ml_features(df)
            ml_metrics = signal_generator.train_ml_model(X, y, model_type=ml_model)
            df = signal_generator.generate_ml_signals(df, feature_engineer.get_feature_columns())
            df = signal_generator.generate_combined_signals(df)
        
        _cached_data = df
        
        # Get latest signals
        latest_signals = signal_generator.get_latest_signals(df)
        
        return {
            "status": "success",
            "ml_trained": train_ml,
            "ml_metrics": ml_metrics,
            "latest_signals": latest_signals,
            "signal_counts": {
                "rule_buy": int((df['signal_rule'] == 1).sum()),
                "rule_sell": int((df['signal_rule'] == -1).sum()),
                "ml_buy": int((df.get('signal_ml', 0) == 1).sum()) if 'signal_ml' in df.columns else 0,
                "ml_sell": int((df.get('signal_ml', 0) == -1).sum()) if 'signal_ml' in df.columns else 0,
            }
        }
    except Exception as e:
        logger.error(f"Error generating signals: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/signals/latest")
async def get_latest_signals():
    """Get the most recent trading signals"""
    if _cached_data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    return signal_generator.get_latest_signals(_cached_data)


@api_router.get("/signals/history")
async def get_signal_history(limit: int = Query(default=50, le=500)):
    """Get historical signals for the trade log"""
    if _cached_data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    df = _cached_data.copy()
    
    # Filter to only rows with signals
    signals_df = df[df['signal_rule'] != 0].tail(limit)
    
    signals = []
    for _, row in signals_df.iterrows():
        signals.append({
            "date": row['date'].isoformat(),
            "close": round(float(row['close']), 2),
            "signal_rule": row.get('signal_rule_label', 'HOLD'),
            "signal_ml": row.get('signal_ml_label', 'N/A'),
            "rsi": round(float(row.get('rsi', 0)), 2),
        })
    
    return {"signals": signals, "count": len(signals)}


@api_router.post("/backtest/run")
async def run_backtest():
    """Run backtest with current data and signals"""
    global _cached_results
    
    if _cached_data is None:
        raise HTTPException(status_code=400, detail="No data loaded. Call /data/load first")
    
    if 'signal_rule' not in _cached_data.columns:
        raise HTTPException(status_code=400, detail="No signals generated. Call /signals/generate first")
    
    try:
        results = backtest_engine.run_comparison_backtest(_cached_data)
        _cached_results = results
        
        # Store results in MongoDB for persistence
        await db.backtest_results.insert_one({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": config.symbol,
            "metrics": {
                "rule_based": results.get('rule_based', {}).get('metrics', {}),
                "ml_strategy": results.get('ml_strategy', {}).get('metrics', {}),
                "buy_hold": results.get('buy_hold', {}).get('metrics', {}),
            }
        })
        
        return {
            "status": "success",
            "rule_based": results.get('rule_based', {}).get('metrics', {}),
            "ml_strategy": results.get('ml_strategy', {}).get('metrics', {}),
            "buy_hold": results.get('buy_hold', {}).get('metrics', {}),
        }
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/backtest/results")
async def get_backtest_results():
    """Get cached backtest results"""
    if _cached_results is None:
        raise HTTPException(status_code=400, detail="No backtest results. Run /backtest/run first")
    
    return {
        "rule_based": _cached_results.get('rule_based', {}).get('metrics', {}),
        "ml_strategy": _cached_results.get('ml_strategy', {}).get('metrics', {}),
        "buy_hold": _cached_results.get('buy_hold', {}).get('metrics', {}),
    }


@api_router.get("/backtest/equity-curve")
async def get_equity_curve(strategy: str = "rule_based"):
    """Get equity curve data for charting"""
    if _cached_results is None:
        raise HTTPException(status_code=400, detail="No backtest results")
    
    if strategy not in _cached_results:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")
    
    equity_curve = _cached_results[strategy].get('equity_curve', [])
    
    # Sample if too large (for chart performance)
    if len(equity_curve) > 500:
        step = len(equity_curve) // 500
        equity_curve = equity_curve[::step]
    
    return {"equity_curve": equity_curve, "strategy": strategy}


@api_router.get("/backtest/trades")
async def get_trades(strategy: str = "rule_based"):
    """Get trade history from backtest"""
    if _cached_results is None:
        raise HTTPException(status_code=400, detail="No backtest results")
    
    if strategy not in _cached_results:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {strategy}")
    
    trades = _cached_results[strategy].get('trades', [])
    
    return {"trades": trades, "count": len(trades), "strategy": strategy}


@api_router.get("/backtest/drawdown")
async def get_drawdown_data(strategy: str = "rule_based"):
    """Get drawdown data for charting"""
    if _cached_results is None:
        raise HTTPException(status_code=400, detail="No backtest results")
    
    equity_curve = _cached_results.get(strategy, {}).get('equity_curve', [])
    
    if not equity_curve:
        return {"drawdown": [], "strategy": strategy}
    
    # Calculate drawdown series
    import pandas as pd
    values = pd.Series([e['portfolio_value'] for e in equity_curve])
    dates = [e['date'] for e in equity_curve]
    
    rolling_max = values.expanding().max()
    drawdown = (values - rolling_max) / rolling_max * 100
    
    drawdown_data = [
        {"date": dates[i], "drawdown": round(float(drawdown.iloc[i]), 2)}
        for i in range(len(drawdown))
    ]
    
    # Sample if too large
    if len(drawdown_data) > 500:
        step = len(drawdown_data) // 500
        drawdown_data = drawdown_data[::step]
    
    return {"drawdown": drawdown_data, "strategy": strategy}


@api_router.get("/risk/status")
async def get_risk_status():
    """Get current risk management status"""
    portfolio_value = config.backtest.initial_capital
    if _cached_results:
        equity = _cached_results.get('rule_based', {}).get('equity_curve', [])
        if equity:
            portfolio_value = equity[-1].get('portfolio_value', portfolio_value)
    
    return risk_manager.get_status(portfolio_value)


@api_router.post("/risk/reset-kill-switch")
async def reset_kill_switch():
    """Reset the kill switch (use with caution)"""
    risk_manager.reset_kill_switch()
    return {"status": "Kill switch reset", "kill_switch": risk_manager.kill_switch_triggered}


@api_router.get("/execution/status")
async def get_execution_status():
    """Get execution engine status"""
    return execution_engine.get_status()


# Include router
app.include_router(api_router)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load initial data on startup"""
    logger.info("Starting Algo Trading System...")
    try:
        # Pre-load data
        global _cached_data, _cached_results
        df = data_loader.fetch_data(years=config.data_years)
        df = feature_engineer.calculate_all_features(df)
        df = signal_generator.generate_rule_based_signals(df)
        
        # Train ML model
        X, y, indices = feature_engineer.prepare_ml_features(df)
        signal_generator.train_ml_model(X, y, model_type="logistic")
        df = signal_generator.generate_ml_signals(df, feature_engineer.get_feature_columns())
        df = signal_generator.generate_combined_signals(df)
        
        _cached_data = df
        
        # Run initial backtest
        _cached_results = backtest_engine.run_comparison_backtest(df)
        
        logger.info(f"Startup complete: Loaded {len(df)} days of {config.symbol} data")
    except Exception as e:
        logger.error(f"Startup error (data will load on first request): {e}")


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
