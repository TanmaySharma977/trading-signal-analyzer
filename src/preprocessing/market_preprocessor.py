"""
Market Data Preprocessor - Clean and prepare OHLCV data
"""

import pandas as pd
import numpy as np
from src.utils.logger import logger


class MarketPreprocessor:
    """Cleans and preprocesses market OHLCV data."""

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean OHLCV data.
        - Remove NaN rows
        - Remove zero-volume rows
        - Sort by datetime
        - Remove duplicates
        """
        if df.empty:
            return df

        original_len = len(df)

        # Sort by index (datetime)
        df = df.sort_index()

        # Remove duplicates
        df = df[~df.index.duplicated(keep="first")]

        # Drop NaN
        df = df.dropna(subset=["open", "high", "low", "close"])

        # Remove rows with zero or negative prices
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            df = df[df[col] > 0]

        # Remove zero volume (optional, some days have 0 vol)
        # df = df[df['volume'] > 0]

        # Ensure OHLC consistency: High >= Low
        df = df[df["high"] >= df["low"]]

        cleaned_len = len(df)
        removed = original_len - cleaned_len

        if removed > 0:
            logger.info(f"Cleaned {removed} invalid rows. {cleaned_len} rows remaining.")

        return df

    @staticmethod
    def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
        """Add basic derived features."""
        if df.empty:
            return df

        # Candle body and wick
        df["body"] = df["close"] - df["open"]
        df["body_abs"] = abs(df["body"])
        df["upper_wick"] = df["high"] - df[["open", "close"]].max(axis=1)
        df["lower_wick"] = df[["open", "close"]].min(axis=1) - df["low"]
        df["candle_range"] = df["high"] - df["low"]

        # Price change
        df["price_change"] = df["close"].pct_change()
        df["price_change_abs"] = df["price_change"].abs()

        # Body ratio (body / total range)
        df["body_ratio"] = np.where(
            df["candle_range"] > 0,
            df["body_abs"] / df["candle_range"],
            0,
        )

        # Is bullish / bearish
        df["is_bullish"] = (df["close"] > df["open"]).astype(int)
        df["is_bearish"] = (df["close"] < df["open"]).astype(int)

        return df