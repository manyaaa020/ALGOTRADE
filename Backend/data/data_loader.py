"""
Data Loader Module
Fetches historical OHLCV data using yfinance.
Handles data cleaning and validation.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads and prepares historical market data.
    IMPORTANT: No look-ahead bias - all data is properly sorted by date.
    """
    
    def __init__(self, symbol: str = "SPY"):
        self.symbol = symbol
        self._data: Optional[pd.DataFrame] = None
        self._last_fetch: Optional[datetime] = None
    
    def fetch_data(
        self, 
        years: int = 10, 
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Fetch historical daily OHLCV data.
        
        Args:
            years: Number of years of historical data
            end_date: End date for data (defaults to today)
            
        Returns:
            DataFrame with OHLCV data, sorted by date ascending
        """
        if end_date is None:
            end_date = datetime.now()
        
        start_date = end_date - timedelta(days=years * 365)
        
        logger.info(f"Fetching {self.symbol} data from {start_date.date()} to {end_date.date()}")
        
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1d")
            
            if df.empty:
                raise ValueError(f"No data returned for {self.symbol}")
            
            # Clean and prepare data
            df = self._clean_data(df)
            
            self._data = df
            self._last_fetch = datetime.now()
            
            logger.info(f"Loaded {len(df)} days of data for {self.symbol}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            raise
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the data.
        - Remove duplicates
        - Handle missing values
        - Ensure proper sorting (ascending by date)
        """
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Rename columns to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Remove timezone info for consistency
        if df['date'].dt.tz is not None:
            df['date'] = df['date'].dt.tz_localize(None)
        
        # Sort by date ascending (CRITICAL: prevents look-ahead bias)
        df = df.sort_values('date', ascending=True).reset_index(drop=True)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['date'], keep='last')
        
        # Handle missing values
        # Forward fill for minor gaps
        df = df.ffill()
        
        # Drop rows with any remaining NaN
        initial_len = len(df)
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        if len(df) < initial_len:
            logger.warning(f"Dropped {initial_len - len(df)} rows with missing values")
        
        # Validate data integrity
        self._validate_data(df)
        
        # Keep only essential columns
        essential_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
        df = df[essential_cols]
        
        return df
    
    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate data integrity.
        Raises ValueError if data is invalid.
        """
        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                raise ValueError(f"Found non-positive values in {col}")
        
        # Check high >= low
        if (df['high'] < df['low']).any():
            raise ValueError("Found rows where high < low")
        
        # Check high >= open and high >= close
        if (df['high'] < df['open']).any() or (df['high'] < df['close']).any():
            logger.warning("Found rows where high < open or high < close")
        
        # Check low <= open and low <= close
        if (df['low'] > df['open']).any() or (df['low'] > df['close']).any():
            logger.warning("Found rows where low > open or low > close")
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """Return cached data"""
        return self._data
    
    def get_latest_price(self) -> Optional[float]:
        """Get the most recent closing price"""
        if self._data is None or self._data.empty:
            return None
        return float(self._data['close'].iloc[-1])
    
    def get_date_range(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Get the date range of loaded data"""
        if self._data is None or self._data.empty:
            return None, None
        return (
            self._data['date'].iloc[0].to_pydatetime(),
            self._data['date'].iloc[-1].to_pydatetime()
        )
    
    def to_dict(self) -> dict:
        """Export data summary as dictionary"""
        if self._data is None:
            return {"status": "no_data"}
        
        start_date, end_date = self.get_date_range()
        
        return {
            "symbol": self.symbol,
            "total_days": len(self._data),
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "latest_price": self.get_latest_price(),
            "last_fetch": self._last_fetch.isoformat() if self._last_fetch else None
        }
