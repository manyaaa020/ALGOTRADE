"""
Signal Generation Module
Generates trading signals using rule-based and ML strategies.
"""
import pandas as pd
import numpy as np
from enum import Enum
from typing import Optional, Tuple, Dict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class Signal(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0


class SignalGenerator:
    """
    Generates trading signals using multiple strategies.
    Supports both rule-based and ML-based approaches.
    """
    
    def __init__(
        self,
        rsi_oversold: int = 30,
        rsi_overbought: int = 70
    ):
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        
        # ML components
        self.scaler = StandardScaler()
        self.ml_model: Optional[object] = None
        self.ml_model_type: str = "logistic"
        self.ml_trained: bool = False
        self.ml_accuracy: float = 0.0
    
    def generate_rule_based_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals using MA crossover strategy.
        
        Strategy Rules:
        - BUY: Short MA crosses above Long MA
        - SELL: Short MA crosses below Long MA
        - HOLD: Otherwise
        
        Args:
            df: DataFrame with calculated features
            
        Returns:
            DataFrame with 'signal_rule' column
        """
        df = df.copy()
        
        # Initialize with HOLD
        df['signal_rule'] = Signal.HOLD.value
        
        # BUY signals: MA crossover above
        buy_condition = df['ma_cross_above'] == 1
        df.loc[buy_condition, 'signal_rule'] = Signal.BUY.value
        
        # SELL signals: MA crossover below
        sell_condition = df['ma_cross_below'] == 1
        df.loc[sell_condition, 'signal_rule'] = Signal.SELL.value
        
        # Add signal labels for readability
        df['signal_rule_label'] = df['signal_rule'].map({
            Signal.BUY.value: 'BUY',
            Signal.SELL.value: 'SELL',
            Signal.HOLD.value: 'HOLD'
        })
        
        buy_count = (df['signal_rule'] == Signal.BUY.value).sum()
        sell_count = (df['signal_rule'] == Signal.SELL.value).sum()
        logger.info(f"Rule-based signals: {buy_count} BUY, {sell_count} SELL")
        
        return df
    
    def train_ml_model(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        model_type: str = "logistic"
    ) -> Dict:
        """
        Train ML model for signal generation.
        
        Args:
            X: Feature matrix
            y: Target labels (1: up, 0: down)
            model_type: 'logistic' or 'random_forest'
            
        Returns:
            Dictionary with training metrics
        """
        self.ml_model_type = model_type
        
        # Split data (keeping time order - no shuffle)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train model
        if model_type == "logistic":
            self.ml_model = LogisticRegression(
                random_state=42, 
                max_iter=1000,
                C=0.1  # Regularization to prevent overfitting
            )
        else:  # random_forest
            self.ml_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,  # Limit depth to prevent overfitting
                random_state=42,
                n_jobs=-1
            )
        
        self.ml_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_accuracy = self.ml_model.score(X_train_scaled, y_train)
        test_accuracy = self.ml_model.score(X_test_scaled, y_test)
        
        self.ml_trained = True
        self.ml_accuracy = test_accuracy
        
        metrics = {
            "model_type": model_type,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "train_accuracy": round(train_accuracy, 4),
            "test_accuracy": round(test_accuracy, 4),
        }
        
        logger.info(f"ML model trained: {metrics}")
        
        # Feature importance (for Random Forest)
        if model_type == "random_forest":
            metrics["feature_importance"] = self.ml_model.feature_importances_.tolist()
        
        return metrics
    
    def generate_ml_signals(
        self, 
        df: pd.DataFrame,
        feature_cols: list
    ) -> pd.DataFrame:
        """
        Generate signals using trained ML model.
        
        Args:
            df: DataFrame with features
            feature_cols: List of feature column names
            
        Returns:
            DataFrame with 'signal_ml' column
        """
        df = df.copy()
        
        if not self.ml_trained:
            logger.warning("ML model not trained, returning HOLD signals")
            df['signal_ml'] = Signal.HOLD.value
            df['signal_ml_label'] = 'HOLD'
            df['ml_probability'] = 0.5
            return df
        
        # Get features
        X = df[feature_cols].values
        
        # Handle NaN values
        valid_mask = ~np.isnan(X).any(axis=1)
        
        # Initialize with HOLD
        df['signal_ml'] = Signal.HOLD.value
        df['ml_probability'] = 0.5
        
        if valid_mask.sum() > 0:
            X_valid = X[valid_mask]
            X_scaled = self.scaler.transform(X_valid)
            
            # Predict probabilities
            probas = self.ml_model.predict_proba(X_scaled)[:, 1]
            
            # Generate signals based on probability threshold
            signals = np.where(probas > 0.55, Signal.BUY.value,
                             np.where(probas < 0.45, Signal.SELL.value, Signal.HOLD.value))
            
            df.loc[valid_mask, 'signal_ml'] = signals
            df.loc[valid_mask, 'ml_probability'] = probas
        
        df['signal_ml_label'] = df['signal_ml'].map({
            Signal.BUY.value: 'BUY',
            Signal.SELL.value: 'SELL',
            Signal.HOLD.value: 'HOLD'
        })
        
        buy_count = (df['signal_ml'] == Signal.BUY.value).sum()
        sell_count = (df['signal_ml'] == Signal.SELL.value).sum()
        logger.info(f"ML signals: {buy_count} BUY, {sell_count} SELL")
        
        return df
    
    def generate_combined_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate combined signal from rule-based and ML strategies.
        Uses consensus: both must agree for a signal.
        """
        df = df.copy()
        
        # Consensus approach: both must agree
        df['signal_combined'] = Signal.HOLD.value
        
        # Both agree BUY
        buy_consensus = (df['signal_rule'] == Signal.BUY.value) & (df['signal_ml'] == Signal.BUY.value)
        df.loc[buy_consensus, 'signal_combined'] = Signal.BUY.value
        
        # Both agree SELL
        sell_consensus = (df['signal_rule'] == Signal.SELL.value) & (df['signal_ml'] == Signal.SELL.value)
        df.loc[sell_consensus, 'signal_combined'] = Signal.SELL.value
        
        df['signal_combined_label'] = df['signal_combined'].map({
            Signal.BUY.value: 'BUY',
            Signal.SELL.value: 'SELL',
            Signal.HOLD.value: 'HOLD'
        })
        
        return df
    
    def get_latest_signals(self, df: pd.DataFrame) -> Dict:
        """Get the most recent signals"""
        if df.empty:
            return {}
        
        latest = df.iloc[-1]
        
        return {
            "date": latest['date'].isoformat() if hasattr(latest['date'], 'isoformat') else str(latest['date']),
            "close": float(latest['close']),
            "signal_rule": latest.get('signal_rule_label', 'N/A'),
            "signal_ml": latest.get('signal_ml_label', 'N/A'),
            "ml_probability": float(latest.get('ml_probability', 0.5)),
            "signal_combined": latest.get('signal_combined_label', 'N/A'),
            "rsi": float(latest.get('rsi', 0)),
            "sma_short": float(latest.get('sma_short', 0)),
            "sma_long": float(latest.get('sma_long', 0)),
        }
