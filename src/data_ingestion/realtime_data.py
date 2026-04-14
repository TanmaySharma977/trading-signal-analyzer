"""
Real-Time Data Fetcher for Intraday Trading
- yfinance (free, ~15 min delayed)
- Support for future broker API integration
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import pytz
from typing import Optional, Dict
from src.utils.logger import logger


IST = pytz.timezone("Asia/Kolkata")

MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)
PRE_MARKET = time(9, 0)


class RealtimeDataFetcher:
    """Fetches real-time / intraday data for Indian stocks."""

    def __init__(self, exchange: str = "NSE"):
        self.exchange = exchange
        self.suffix = ".NS" if exchange == "NSE" else ".BO"
        self._cache = {}
        self._last_fetch = {}
        logger.info(f"RealtimeDataFetcher initialized for {exchange}")

    def _get_symbol(self, stock: str) -> str:
        if not stock.endswith((".NS", ".BO")):
            return f"{stock}{self.suffix}"
        return stock

    @staticmethod
    def is_market_open() -> Dict:
        """Check if Indian market is currently open."""
        now = datetime.now(IST)
        current_time = now.time()
        weekday = now.weekday()

        is_weekend = weekday >= 5  # Saturday=5, Sunday=6

        if is_weekend:
            return {
                "is_open": False,
                "reason": "Weekend - Market Closed",
                "current_time": now.strftime("%H:%M:%S IST"),
                "next_open": "Monday 9:15 AM IST",
            }

        if current_time < MARKET_OPEN:
            return {
                "is_open": False,
                "reason": "Pre-Market - Not yet open",
                "current_time": now.strftime("%H:%M:%S IST"),
                "next_open": "Today 9:15 AM IST",
            }

        if current_time > MARKET_CLOSE:
            return {
                "is_open": False,
                "reason": "Post-Market - Closed for today",
                "current_time": now.strftime("%H:%M:%S IST"),
                "next_open": "Tomorrow 9:15 AM IST",
            }

        return {
            "is_open": True,
            "reason": "Market is OPEN",
            "current_time": now.strftime("%H:%M:%S IST"),
            "closes_at": "3:30 PM IST",
            "minutes_remaining": int(
                (datetime.combine(now.date(), MARKET_CLOSE) -
                 datetime.combine(now.date(), current_time)).total_seconds() / 60
            ),
        }

    def fetch_intraday(
        self,
        symbol: str,
        interval: str = "5m",
        period: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch intraday OHLCV data.

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE')
            interval: '1m', '5m', '15m', '30m', '1h'
            period: '1d', '5d' (yfinance limits for intraday)
        """
        ticker_symbol = self._get_symbol(symbol)

        # yfinance limits for intraday
        valid_periods = {
            "1m": "5d",     # Max 7 days for 1m
            "5m": "5d",     # Max 60 days for 5m
            "15m": "5d",
            "30m": "5d",
            "1h": "5d",
        }

        max_period = valid_periods.get(interval, "5d")

        try:
            df = yf.download(
                ticker_symbol,
                period=period if period <= max_period else max_period,
                interval=interval,
                progress=False,
                auto_adjust=True,
            )

            if df.empty:
                logger.warning(f"No intraday data for {ticker_symbol}")
                return pd.DataFrame()

            # Flatten MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df.columns = [col.lower().strip() for col in df.columns]

            required = ["open", "high", "low", "close", "volume"]
            for col in required:
                if col not in df.columns:
                    return pd.DataFrame()

            df = df[required].dropna()

            # Filter to today only if period is 1d
            if period == "1d":
                today = datetime.now(IST).date()
                df = df[df.index.date == today] if hasattr(df.index, 'date') else df

            # Cache
            self._cache[symbol] = df
            self._last_fetch[symbol] = datetime.now()

            logger.info(f"Intraday: {len(df)} candles for {ticker_symbol} ({interval})")
            return df

        except Exception as e:
            logger.error(f"Error fetching intraday {ticker_symbol}: {e}")
            return self._cache.get(symbol, pd.DataFrame())

    def get_live_snapshot(self, symbol: str) -> Dict:
        """Get current price snapshot."""
        ticker_symbol = self._get_symbol(symbol)
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.fast_info

            # Get today's data
            df = yf.download(
                ticker_symbol, period="2d", interval="1d",
                progress=False, auto_adjust=True,
            )

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.columns = [col.lower().strip() for col in df.columns]
            df = df.dropna()

            if df.empty:
                return {}

            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else last
            change = last["close"] - prev["close"]
            change_pct = (change / prev["close"]) * 100

            return {
                "symbol": symbol,
                "price": round(float(last["close"]), 2),
                "change": round(float(change), 2),
                "change_pct": round(float(change_pct), 2),
                "open": round(float(last["open"]), 2),
                "high": round(float(last["high"]), 2),
                "low": round(float(last["low"]), 2),
                "prev_close": round(float(prev["close"]), 2),
                "volume": int(last["volume"]),
            }
        except Exception as e:
            logger.error(f"Snapshot error for {symbol}: {e}")
            return {}

    def fetch_multi_timeframe_intraday(self, symbol: str) -> Dict:
        """Fetch data for multiple intraday timeframes at once."""
        timeframes = {
            "1m": self.fetch_intraday(symbol, "1m", "1d"),
            "5m": self.fetch_intraday(symbol, "5m", "5d"),
            "15m": self.fetch_intraday(symbol, "15m", "5d"),
        }
        return {k: v for k, v in timeframes.items() if not v.empty}