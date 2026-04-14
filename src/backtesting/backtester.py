"""
Backtesting Module - Simulate trading signals on historical data
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from src.utils.logger import logger


class Backtester:
    """Simple backtester for evaluating signal performance."""

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_pct: float = 0.001,
    ):
        self.initial_capital = initial_capital
        self.commission_pct = commission_pct
        logger.info(
            f"Backtester initialized: ₹{initial_capital:,.0f} capital, "
            f"{commission_pct*100}% commission"
        )

    def calculate_sharpe_ratio(self, portfolio_df: pd.DataFrame,
                           risk_free_rate: float = 0.06) -> float:
        """
        Sharpe Ratio = (Return - Risk Free) / Volatility
        risk_free_rate = 6% for India (FD rate approx)
        """
        if portfolio_df.empty or len(portfolio_df) < 2:
            return 0.0

        daily_returns = portfolio_df["value"].pct_change().dropna()

        if daily_returns.std() == 0:
            return 0.0

        annual_return = daily_returns.mean() * 252
        annual_vol = daily_returns.std() * np.sqrt(252)

        sharpe = (annual_return - risk_free_rate) / annual_vol
        return round(sharpe, 2)

    def run(self, df: pd.DataFrame, signals: List[Dict]) -> Dict:
        """
        Run backtest on historical data with generated signals.

        Args:
            df: OHLCV DataFrame
            signals: List of {'date': ..., 'signal': 'BUY'|'SELL'|'HOLD'}

        Returns:
            Backtest results dict
        """
        capital = self.initial_capital
        position = 0  # Number of shares held
        buy_price = 0
        trades = []
        portfolio_values = []

        signal_map = {s["date"]: s["signal"] for s in signals}

        for i, (date, row) in enumerate(df.iterrows()):
            current_signal = signal_map.get(date, "HOLD")
            price = row["close"]

            if current_signal == "BUY" and position == 0:
                # Buy with all capital
                shares = int(capital / (price * (1 + self.commission_pct)))
                if shares > 0:
                    cost = shares * price * (1 + self.commission_pct)
                    capital -= cost
                    position = shares
                    buy_price = price
                    trades.append({
                        "date": date,
                        "type": "BUY",
                        "price": price,
                        "shares": shares,
                        "cost": cost,
                    })

            elif current_signal == "SELL" and position > 0:
                # Sell all
                revenue = position * price * (1 - self.commission_pct)
                pnl = revenue - (position * buy_price)
                capital += revenue
                trades.append({
                    "date": date,
                    "type": "SELL",
                    "price": price,
                    "shares": position,
                    "revenue": revenue,
                    "pnl": pnl,
                })
                position = 0
                buy_price = 0

            # Track portfolio value
            portfolio_value = capital + (position * price)
            portfolio_values.append({
                "date": date,
                "value": portfolio_value,
                "capital": capital,
                "position_value": position * price,
            })

        # Final value
        final_value = capital + (position * df.iloc[-1]["close"] if position > 0 else 0)
        total_return = ((final_value - self.initial_capital) / self.initial_capital) * 100

        # Calculate metrics
        sell_trades = [t for t in trades if t["type"] == "SELL"]
        wins = [t for t in sell_trades if t.get("pnl", 0) > 0]
        losses = [t for t in sell_trades if t.get("pnl", 0) <= 0]

        win_rate = (len(wins) / len(sell_trades) * 100) if sell_trades else 0

        portfolio_df = pd.DataFrame(portfolio_values)
        if not portfolio_df.empty:
            peak = portfolio_df["value"].expanding().max()
            drawdown = ((portfolio_df["value"] - peak) / peak) * 100
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0

        results = {
            "initial_capital": self.initial_capital,
            "final_value": round(final_value, 2),
            "total_return_pct": round(total_return, 2),
            "total_trades": len(trades),
            "buy_trades": len([t for t in trades if t["type"] == "BUY"]),
            "sell_trades": len(sell_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": round(win_rate, 1),
            "max_drawdown_pct": round(max_drawdown, 2),
            "trades": trades,
            "portfolio_values": portfolio_df,
        }

        logger.info(
            f"Backtest complete: Return={total_return:.1f}%, "
            f"Win Rate={win_rate:.0f}%, Trades={len(trades)}"
        )
        return results