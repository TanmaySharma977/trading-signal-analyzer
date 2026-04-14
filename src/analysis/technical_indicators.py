"""
Technical Indicators - RSI, MACD, SMA, EMA, Bollinger Bands
"""

import pandas as pd
import numpy as np
from src.utils.logger import logger


class TechnicalIndicators:
    @staticmethod
    def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        """Volume Weighted Average Price — institutional favorite."""
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        df["vwap"] = (typical_price * df["volume"]).cumsum() / df["volume"].cumsum()
        return df

    @staticmethod
    def add_stochastic(df: pd.DataFrame, k_period=14, d_period=3) -> pd.DataFrame:
        """Stochastic Oscillator — momentum indicator."""
        low_min = df["low"].rolling(window=k_period).min()
        high_max = df["high"].rolling(window=k_period).max()
        df["stoch_k"] = ((df["close"] - low_min) / (high_max - low_min)) * 100
        df["stoch_d"] = df["stoch_k"].rolling(window=d_period).mean()
        return df

    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """On-Balance Volume — volume confirms price."""
        obv = [0]
        for i in range(1, len(df)):
            if df["close"].iloc[i] > df["close"].iloc[i - 1]:
                obv.append(obv[-1] + df["volume"].iloc[i])
            elif df["close"].iloc[i] < df["close"].iloc[i - 1]:
                obv.append(obv[-1] - df["volume"].iloc[i])
            else:
                obv.append(obv[-1])
        df["obv"] = obv
        return df

    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average Directional Index — trend strength (not direction)."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift()),
        ], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df["adx"] = dx.rolling(window=period).mean()
        df["plus_di"] = plus_di
        df["minus_di"] = minus_di
        return df

    @staticmethod
    def get_trend_signal(df: pd.DataFrame) -> float:
        """
        IMPROVED trend signal using ALL indicators.
        Returns: -1 (strong bearish) to +1 (strong bullish)
        """
        signals = []
        last = df.iloc[-1]

        # RSI
        if "rsi" in df.columns and not pd.isna(last.get("rsi")):
            rsi = last["rsi"]
            if rsi < 25:
                signals.append(("rsi", 0.9))
            elif rsi < 30:
                signals.append(("rsi", 0.6))
            elif rsi > 75:
                signals.append(("rsi", -0.9))
            elif rsi > 70:
                signals.append(("rsi", -0.6))
            else:
                signals.append(("rsi", 0.0))

        # MACD
        if "macd_histogram" in df.columns and not pd.isna(last.get("macd_histogram")):
            hist = last["macd_histogram"]
            prev_hist = df["macd_histogram"].iloc[-2] if len(df) > 1 else 0

            if hist > 0 and hist > prev_hist:
                signals.append(("macd", 0.7))     # Bullish + strengthening
            elif hist > 0:
                signals.append(("macd", 0.3))     # Bullish but weakening
            elif hist < 0 and hist < prev_hist:
                signals.append(("macd", -0.7))    # Bearish + strengthening
            elif hist < 0:
                signals.append(("macd", -0.3))

        # SMA crossover
        if "sma_10" in df.columns and "sma_20" in df.columns:
            sma10 = last.get("sma_10")
            sma20 = last.get("sma_20")
            if not pd.isna(sma10) and not pd.isna(sma20):
                if sma10 > sma20:
                    signals.append(("sma_cross", 0.5))
                else:
                    signals.append(("sma_cross", -0.5))

        # Stochastic
        if "stoch_k" in df.columns and not pd.isna(last.get("stoch_k")):
            k = last["stoch_k"]
            if k < 20:
                signals.append(("stoch", 0.6))
            elif k > 80:
                signals.append(("stoch", -0.6))
            else:
                signals.append(("stoch", 0.0))

        # ADX (trend strength)
        if "adx" in df.columns and not pd.isna(last.get("adx")):
            adx = last["adx"]
            if adx > 25:  # Strong trend
                # Amplify other signals
                for i, (name, score) in enumerate(signals):
                    signals[i] = (name, score * 1.3)

        # Price vs Bollinger Bands
        if "bb_lower" in df.columns and "bb_upper" in df.columns:
            if not pd.isna(last.get("bb_lower")):
                close = last["close"]
                if close <= last["bb_lower"]:
                    signals.append(("bb", 0.5))    # Near lower band = oversold
                elif close >= last["bb_upper"]:
                    signals.append(("bb", -0.5))   # Near upper band = overbought
                else:
                    signals.append(("bb", 0.0))

        if signals:
            avg = np.mean([s[1] for s in signals])
            return float(np.clip(avg, -1.0, 1.0))
        return 0.0
    """Calculates technical indicators on OHLCV data."""

    @staticmethod
    def add_all(df: pd.DataFrame) -> pd.DataFrame:
        """Add all technical indicators to dataframe."""
        df = TechnicalIndicators.add_sma(df)
        df = TechnicalIndicators.add_ema(df)
        df = TechnicalIndicators.add_rsi(df)
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_bollinger_bands(df)
        df = TechnicalIndicators.add_atr(df)
        return df

    @staticmethod
    def add_sma(df: pd.DataFrame, periods: list = [10, 20, 50]) -> pd.DataFrame:
        """Simple Moving Average."""
        for p in periods:
            df[f"sma_{p}"] = df["close"].rolling(window=p).mean()
        return df

    @staticmethod
    def add_ema(df: pd.DataFrame, periods: list = [12, 26]) -> pd.DataFrame:
        """Exponential Moving Average."""
        for p in periods:
            df[f"ema_{p}"] = df["close"].ewm(span=p, adjust=False).mean()
        return df

    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Relative Strength Index."""
        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
    ) -> pd.DataFrame:
        """MACD indicator."""
        ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
        df["macd"] = ema_fast - ema_slow
        df["macd_signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
        df["macd_histogram"] = df["macd"] - df["macd_signal"]
        return df

    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame, period: int = 20, std_dev: float = 2.0
    ) -> pd.DataFrame:
        """Bollinger Bands."""
        sma = df["close"].rolling(window=period).mean()
        std = df["close"].rolling(window=period).std()
        df["bb_upper"] = sma + (std * std_dev)
        df["bb_middle"] = sma
        df["bb_lower"] = sma - (std * std_dev)
        return df

    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Average True Range."""
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df["atr"] = true_range.rolling(window=period).mean()
        return df

    @staticmethod
    def get_trend_signal(df: pd.DataFrame) -> float:
        """
        Get overall trend signal from indicators.
        Returns: float between -1 (bearish) and +1 (bullish)
        """
        signals = []
        last = df.iloc[-1]

        # RSI signal
        if "rsi" in df.columns and not pd.isna(last.get("rsi")):
            rsi = last["rsi"]
            if rsi < 30:
                signals.append(0.7)  # Oversold = bullish
            elif rsi > 70:
                signals.append(-0.7)  # Overbought = bearish
            else:
                signals.append(0.0)

        # MACD signal
        if "macd_histogram" in df.columns and not pd.isna(
            last.get("macd_histogram")
        ):
            if last["macd_histogram"] > 0:
                signals.append(0.5)
            else:
                signals.append(-0.5)

        # SMA crossover
        if "sma_10" in df.columns and "sma_20" in df.columns:
            if not pd.isna(last.get("sma_10")) and not pd.isna(last.get("sma_20")):
                if last["sma_10"] > last["sma_20"]:
                    signals.append(0.4)
                else:
                    signals.append(-0.4)

        if signals:
            return np.clip(np.mean(signals), -1.0, 1.0)
        return 0.0