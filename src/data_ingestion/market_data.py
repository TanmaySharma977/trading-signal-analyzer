"""
Market Data Fetcher - Supports Yahoo Finance for Indian NSE/BSE stocks
"""

import yfinance as yf
import pandas as pd
from typing import Optional
from src.utils.logger import logger


class MarketDataFetcher:
    """Fetches OHLCV data for Indian stocks."""

    def __init__(self, exchange: str = "NSE"):
        self.exchange = exchange
        self.suffix = ".NS" if exchange == "NSE" else ".BO"
        logger.info(f"MarketDataFetcher initialized for {exchange}")

    def _get_symbol(self, stock: str) -> str:
        if not stock.endswith((".NS", ".BO")):
            return f"{stock}{self.suffix}"
        return stock

    def fetch_historical(
        self,
        symbol: str,
        period: str = "3mo",
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:

        ticker_symbol = self._get_symbol(symbol)
        logger.info(
            f"Fetching {ticker_symbol} | period={period} | interval={interval}"
        )

        try:
            if start_date and end_date:
                df = yf.download(
                    ticker_symbol,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                )
            else:
                df = yf.download(
                    ticker_symbol,
                    period=period,
                    interval=interval,
                    progress=False,
                    auto_adjust=True,
                )

            if df.empty:
                logger.warning(f"No data returned for {ticker_symbol}")
                return pd.DataFrame()

            # FIX: Flatten MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # Lowercase all column names
            df.columns = [col.lower().strip() for col in df.columns]

            # Keep only needed columns
            required = ["open", "high", "low", "close", "volume"]
            for col in required:
                if col not in df.columns:
                    logger.error(f"Missing column: {col}")
                    return pd.DataFrame()

            df = df[required]
            df.index.name = "datetime"
            df = df.dropna()

            logger.info(f"Fetched {len(df)} candles for {ticker_symbol}")
            return df

        except Exception as e:
            logger.error(f"Error fetching {ticker_symbol}: {e}")
            return pd.DataFrame()

    def fetch_live_price(self, symbol: str) -> dict:
        ticker_symbol = self._get_symbol(symbol)
        try:
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

            return {
                "symbol": symbol,
                "price": float(last["close"]),
                "prev_close": float(prev["close"]),
                "open": float(last["open"]),
                "day_high": float(last["high"]),
                "day_low": float(last["low"]),
                "volume": int(last["volume"]),
            }
        except Exception as e:
            logger.error(f"Error fetching live price for {symbol}: {e}")
            return {}

    def get_stock_info(self, symbol: str) -> dict:
        ticker_symbol = self._get_symbol(symbol)
        try:
            ticker = yf.Ticker(ticker_symbol)
            info = ticker.info
            return {
                "name": info.get("longName", symbol),
                "sector": info.get("sector", "N/A"),
                "industry": info.get("industry", "N/A"),
            }
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {"name": symbol}