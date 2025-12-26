"""
Feature Engineering Module
Calculates technical indicators for trading signals.
"""
import pandas as pd
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Generates features from OHLCV data for signal generation.
    All features are calculated using only past data (no look-ahead bias).
    """
    
    def __init__(
        self,
        short_ma_window: int = 20,
        long_ma_window: int = 50,
        rsi_period: int = 14,
        volatility_window: int = 20
    ):
        self.short_ma_window = short_ma_window
        self.long_ma_window = long_ma_window
        self.rsi_period = rsi_period
        self.volatility_window = volatility_window
    
    def calculate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added feature columns
        """
        df = df.copy()
        
        # Daily returns
        df = self.add_returns(df)
        
        # Moving averages
        df = self.add_moving_averages(df)
        
        # Volatility
        df = self.add_volatility(df)
        
        # RSI
        df = self.add_rsi(df)
        
        # MA crossover signals
        df = self.add_ma_crossover(df)
        
        # Price momentum
        df = self.add_momentum(df)
        
        logger.info(f"Calculated {len(df.columns) - 6} features")
        
        return df
    
    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate daily returns and log returns.
        """
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Cumulative returns
        df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
        
        return df
    
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate short and long moving averages.
        """
        df['sma_short'] = df['close'].rolling(window=self.short_ma_window).mean()
        df['sma_long'] = df['close'].rolling(window=self.long_ma_window).mean()
        
        # Exponential moving averages
        df['ema_short'] = df['close'].ewm(span=self.short_ma_window, adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=self.long_ma_window, adjust=False).mean()
        
        # Distance from MA (useful for mean reversion)
        df['dist_from_sma_short'] = (df['close'] - df['sma_short']) / df['sma_short']
        df['dist_from_sma_long'] = (df['close'] - df['sma_long']) / df['sma_long']
        
        return df
    
    def add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate rolling volatility measures.
        """
        # Historical volatility (annualized)
        df['volatility'] = df['returns'].rolling(window=self.volatility_window).std() * np.sqrt(252)
        
        # Average True Range (ATR)
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(window=self.volatility_window).mean()
        
        # ATR as percentage of price
        df['atr_pct'] = df['atr'] / df['close']
        
        return df
    
    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Relative Strength Index.
        RSI = 100 - (100 / (1 + RS))
        where RS = avg gain / avg loss
        """
        delta = df['close'].diff()
        
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        # Use exponential moving average for smoother RSI
        avg_gain = gain.ewm(span=self.rsi_period, adjust=False).mean()
        avg_loss = loss.ewm(span=self.rsi_period, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI zones
        df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
        df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
        
        return df
    
    def add_ma_crossover(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate moving average crossover signals.
        """
        # Current MA relationship
        df['ma_diff'] = df['sma_short'] - df['sma_long']
        df['ma_diff_pct'] = df['ma_diff'] / df['sma_long']
        
        # Crossover detection
        df['ma_cross_above'] = (
            (df['sma_short'] > df['sma_long']) & 
            (df['sma_short'].shift(1) <= df['sma_long'].shift(1))
        ).astype(int)
        
        df['ma_cross_below'] = (
            (df['sma_short'] < df['sma_long']) & 
            (df['sma_short'].shift(1) >= df['sma_long'].shift(1))
        ).astype(int)
        
        return df
    
    def add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate price momentum indicators.
        """
        # Rate of change (5, 10, 20 days)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = df['close'].pct_change(periods=period)
        
        # Price relative to recent high/low
        df['dist_from_high_20'] = df['close'] / df['high'].rolling(20).max() - 1
        df['dist_from_low_20'] = df['close'] / df['low'].rolling(20).min() - 1
        
        return df
    
    def get_feature_columns(self) -> list:
        """Return list of feature column names for ML model"""
        return [
            'returns', 'volatility', 'rsi', 
            'dist_from_sma_short', 'dist_from_sma_long',
            'ma_diff_pct', 'atr_pct',
            'roc_5', 'roc_10', 'roc_20',
            'dist_from_high_20', 'dist_from_low_20'
        ]
    
    def prepare_ml_features(self, df: pd.DataFrame) -> tuple:
        """
        Prepare features and target for ML model.
        Target: 1 if next day return is positive, 0 otherwise
        
        Returns:
            (X, y) tuple of features and target
        """
        df = df.copy()
        
        # Target: next day direction
        df['target'] = (df['returns'].shift(-1) > 0).astype(int)
        
        # Get feature columns
        feature_cols = self.get_feature_columns()
        
        # Drop rows with NaN
        df_clean = df.dropna(subset=feature_cols + ['target'])
        
        X = df_clean[feature_cols].values
        y = df_clean['target'].values
        
        return X, y, df_clean.index.tolist()
