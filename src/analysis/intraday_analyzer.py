"""
Intraday-Specific Trading Strategies
- VWAP Strategy
- Opening Range Breakout (ORB)
- Pivot Point Strategy
- EMA Crossover (9/21)
- RSI Divergence
- Intraday Momentum
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from src.utils.logger import logger


class IntradayAnalyzer:
    """Intraday-specific analysis strategies."""

    def __init__(self, df: pd.DataFrame, prev_day_data: pd.DataFrame = None):
        """
        Args:
            df: Intraday OHLCV DataFrame (e.g., 5m candles)
            prev_day_data: Previous day's daily data for pivot calculation
        """
        self.df = df.copy()
        self.prev_day = prev_day_data
        self.signals: List[Dict] = []
        logger.info(f"IntradayAnalyzer initialized with {len(df)} candles")

    def run_all_strategies(self) -> Dict:
        """Run all intraday strategies and combine signals."""
        self.signals = []

        strategies = {
            "vwap": self._vwap_strategy(),
            "orb": self._opening_range_breakout(),
            "ema_cross": self._ema_crossover_strategy(),
            "rsi_intraday": self._rsi_intraday_strategy(),
            "momentum": self._intraday_momentum(),
            "pivot": self._pivot_point_strategy(),
            "volume_spike": self._volume_spike_strategy(),
            "candle_strength": self._candle_strength(),
        }

        # Calculate combined score
        scores = []
        weights = []
        details = {}

        weight_map = {
            "vwap": 0.20,
            "orb": 0.15,
            "ema_cross": 0.15,
            "rsi_intraday": 0.15,
            "momentum": 0.10,
            "pivot": 0.10,
            "volume_spike": 0.08,
            "candle_strength": 0.07,
        }

        for name, result in strategies.items():
            if result is not None:
                w = weight_map.get(name, 0.1)
                scores.append(result["score"] * w)
                weights.append(w)
                details[name] = result

        total_weight = sum(weights) if weights else 1
        combined_score = sum(scores) / total_weight if total_weight > 0 else 0

        # Determine signal
        if combined_score > 0.15:
            signal = "BUY"
        elif combined_score < -0.15:
            signal = "SELL"
        else:
            signal = "HOLD"

        confidence = min(abs(combined_score) * 150, 95)

        # Build reason
        bullish_reasons = [
            name for name, r in details.items() if r["score"] > 0.1
        ]
        bearish_reasons = [
            name for name, r in details.items() if r["score"] < -0.1
        ]

        reason_parts = []
        if bullish_reasons:
            reason_parts.append(f"Bullish: {', '.join(bullish_reasons)}")
        if bearish_reasons:
            reason_parts.append(f"Bearish: {', '.join(bearish_reasons)}")

        return {
            "signal": signal,
            "confidence": round(confidence, 1),
            "score": round(combined_score, 4),
            "reason": " | ".join(reason_parts) if reason_parts else "Mixed signals",
            "strategies": details,
            "active_strategies": len(details),
        }

    # ==========================
    # STRATEGY 1: VWAP
    # ==========================

    def _vwap_strategy(self) -> Dict:
        """
        VWAP (Volume Weighted Average Price) Strategy
        - Price above VWAP = Bullish bias
        - Price below VWAP = Bearish bias
        - Bouncing off VWAP = Entry signal
        """
        if len(self.df) < 5 or "volume" not in self.df.columns:
            return None

        df = self.df.copy()

        # Calculate VWAP
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cum_tp_vol = (typical_price * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()
        df["vwap"] = cum_tp_vol / cum_vol

        last = df.iloc[-1]
        vwap = last["vwap"]
        price = last["close"]

        if pd.isna(vwap) or vwap == 0:
            return None

        # Distance from VWAP as percentage
        dist_pct = (price - vwap) / vwap * 100

        # Score
        if dist_pct > 0.5:
            score = min(dist_pct / 2, 1.0)
            desc = f"Price above VWAP by {dist_pct:.2f}% (Bullish)"
        elif dist_pct < -0.5:
            score = max(dist_pct / 2, -1.0)
            desc = f"Price below VWAP by {abs(dist_pct):.2f}% (Bearish)"
        else:
            score = 0.0
            desc = f"Price near VWAP (Neutral)"

        # Check for VWAP bounce
        if len(df) >= 3:
            prev_below = df["close"].iloc[-3] < df["vwap"].iloc[-3]
            now_above = price > vwap
            if prev_below and now_above:
                score = max(score, 0.6)
                desc = "VWAP Bounce (Bullish reversal)"

            prev_above = df["close"].iloc[-3] > df["vwap"].iloc[-3]
            now_below = price < vwap
            if prev_above and now_below:
                score = min(score, -0.6)
                desc = "VWAP Rejection (Bearish reversal)"

        self.df["vwap"] = df["vwap"]

        return {
            "name": "VWAP",
            "score": round(score, 4),
            "description": desc,
            "vwap_value": round(vwap, 2),
            "distance_pct": round(dist_pct, 2),
        }

    # ==========================
    # STRATEGY 2: Opening Range Breakout
    # ==========================

    def _opening_range_breakout(self) -> Dict:
        """
        ORB Strategy:
        - Define range of first 15-30 mins
        - Breakout above = BUY
        - Breakdown below = SELL
        """
        if len(self.df) < 6:  # Need at least 30 mins of 5m data
            return None

        # First 6 candles = first 30 mins (if 5m interval)
        # Adjust based on actual data
        n_opening = min(6, len(self.df) // 3)
        opening_range = self.df.iloc[:n_opening]

        orb_high = opening_range["high"].max()
        orb_low = opening_range["low"].min()
        orb_mid = (orb_high + orb_low) / 2

        current_price = self.df["close"].iloc[-1]

        if orb_high == orb_low:
            return None

        # Score
        if current_price > orb_high:
            dist = (current_price - orb_high) / (orb_high - orb_low)
            score = min(dist * 0.5 + 0.3, 1.0)
            desc = f"ORB Breakout! Price above {orb_high:.2f}"
        elif current_price < orb_low:
            dist = (orb_low - current_price) / (orb_high - orb_low)
            score = max(-dist * 0.5 - 0.3, -1.0)
            desc = f"ORB Breakdown! Price below {orb_low:.2f}"
        elif current_price > orb_mid:
            score = 0.15
            desc = f"Inside ORB, leaning bullish"
        else:
            score = -0.15
            desc = f"Inside ORB, leaning bearish"

        return {
            "name": "Opening Range Breakout",
            "score": round(score, 4),
            "description": desc,
            "orb_high": round(orb_high, 2),
            "orb_low": round(orb_low, 2),
            "current": round(current_price, 2),
        }

    # ==========================
    # STRATEGY 3: EMA 9/21 Crossover
    # ==========================

    def _ema_crossover_strategy(self) -> Dict:
        """
        EMA 9/21 Crossover:
        - 9 EMA crosses above 21 EMA = BUY
        - 9 EMA crosses below 21 EMA = SELL
        """
        if len(self.df) < 25:
            return None

        df = self.df.copy()
        df["ema_9"] = df["close"].ewm(span=9, adjust=False).mean()
        df["ema_21"] = df["close"].ewm(span=21, adjust=False).mean()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        ema9 = last["ema_9"]
        ema21 = last["ema_21"]
        prev_ema9 = prev["ema_9"]
        prev_ema21 = prev["ema_21"]

        # Fresh crossover
        bullish_cross = prev_ema9 <= prev_ema21 and ema9 > ema21
        bearish_cross = prev_ema9 >= prev_ema21 and ema9 < ema21

        if bullish_cross:
            score = 0.8
            desc = "EMA 9/21 Bullish Crossover (Fresh!)"
        elif bearish_cross:
            score = -0.8
            desc = "EMA 9/21 Bearish Crossover (Fresh!)"
        elif ema9 > ema21:
            gap_pct = (ema9 - ema21) / ema21 * 100
            score = min(gap_pct * 2, 0.6)
            desc = f"EMA 9 above EMA 21 (Bullish, gap: {gap_pct:.2f}%)"
        else:
            gap_pct = (ema21 - ema9) / ema21 * 100
            score = max(-gap_pct * 2, -0.6)
            desc = f"EMA 9 below EMA 21 (Bearish, gap: {gap_pct:.2f}%)"

        self.df["ema_9"] = df["ema_9"]
        self.df["ema_21"] = df["ema_21"]

        return {
            "name": "EMA 9/21 Crossover",
            "score": round(score, 4),
            "description": desc,
            "ema_9": round(ema9, 2),
            "ema_21": round(ema21, 2),
        }

    # ==========================
    # STRATEGY 4: Intraday RSI
    # ==========================

    def _rsi_intraday_strategy(self) -> Dict:
        """
        RSI for intraday (period=7, faster than daily RSI-14):
        - RSI < 25 = Oversold = BUY
        - RSI > 75 = Overbought = SELL
        """
        if len(self.df) < 10:
            return None

        period = 7  # Faster RSI for intraday
        delta = self.df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2] if len(rsi) > 1 else current_rsi

        if pd.isna(current_rsi):
            return None

        self.df["rsi_7"] = rsi

        # Score
        if current_rsi < 20:
            score = 0.9
            desc = f"RSI {current_rsi:.0f} - Extremely Oversold (Strong BUY)"
        elif current_rsi < 30:
            score = 0.6
            desc = f"RSI {current_rsi:.0f} - Oversold (BUY)"
        elif current_rsi > 80:
            score = -0.9
            desc = f"RSI {current_rsi:.0f} - Extremely Overbought (Strong SELL)"
        elif current_rsi > 70:
            score = -0.6
            desc = f"RSI {current_rsi:.0f} - Overbought (SELL)"
        elif current_rsi > 50 and current_rsi > prev_rsi:
            score = 0.2
            desc = f"RSI {current_rsi:.0f} - Rising momentum"
        elif current_rsi < 50 and current_rsi < prev_rsi:
            score = -0.2
            desc = f"RSI {current_rsi:.0f} - Falling momentum"
        else:
            score = 0.0
            desc = f"RSI {current_rsi:.0f} - Neutral"

        # RSI divergence check
        if len(self.df) >= 10:
            price_higher = self.df["close"].iloc[-1] > self.df["close"].iloc[-5]
            rsi_lower = current_rsi < rsi.iloc[-5]

            if price_higher and rsi_lower:
                score = min(score - 0.3, -0.3)
                desc += " | Bearish Divergence!"

            price_lower = self.df["close"].iloc[-1] < self.df["close"].iloc[-5]
            rsi_higher = current_rsi > rsi.iloc[-5]

            if price_lower and rsi_higher:
                score = max(score + 0.3, 0.3)
                desc += " | Bullish Divergence!"

        return {
            "name": "RSI Intraday",
            "score": round(float(score), 4),
            "description": desc,
            "rsi": round(float(current_rsi), 1),
        }

    # ==========================
    # STRATEGY 5: Intraday Momentum
    # ==========================

    def _intraday_momentum(self) -> Dict:
        """
        Momentum: Rate of change in last N candles.
        Fast-moving stocks get stronger signals.
        """
        if len(self.df) < 10:
            return None

        close = self.df["close"]

        # Short-term momentum (last 5 candles)
        roc_5 = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5] * 100

        # Medium-term momentum (last 10 candles)
        roc_10 = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] * 100

        # Acceleration: is momentum increasing?
        if len(self.df) >= 15:
            prev_roc_5 = (close.iloc[-6] - close.iloc[-11]) / close.iloc[-11] * 100
            acceleration = roc_5 - prev_roc_5
        else:
            acceleration = 0

        # Score
        momentum = roc_5 * 0.6 + roc_10 * 0.4
        score = np.clip(momentum / 2, -1.0, 1.0)

        # Acceleration bonus
        if acceleration > 0.5 and score > 0:
            score = min(score * 1.3, 1.0)
        elif acceleration < -0.5 and score < 0:
            score = max(score * 1.3, -1.0)

        if score > 0.1:
            desc = f"Bullish momentum: +{roc_5:.2f}% (5-candle)"
        elif score < -0.1:
            desc = f"Bearish momentum: {roc_5:.2f}% (5-candle)"
        else:
            desc = f"Flat momentum: {roc_5:.2f}%"

        return {
            "name": "Momentum",
            "score": round(float(score), 4),
            "description": desc,
            "roc_5": round(float(roc_5), 2),
            "roc_10": round(float(roc_10), 2),
        }

    # ==========================
    # STRATEGY 6: Pivot Points
    # ==========================

    def _pivot_point_strategy(self) -> Dict:
        """
        Classic Pivot Points from previous day:
        - Above R1 = Strong bullish
        - Below S1 = Strong bearish
        - Between = Use as support/resistance
        """
        if self.prev_day is None or self.prev_day.empty:
            # Use first candle of current data as reference
            if len(self.df) < 5:
                return None
            ref = self.df.iloc[0]
        else:
            ref = self.prev_day.iloc[-1]

        prev_high = ref["high"]
        prev_low = ref["low"]
        prev_close = ref["close"]

        # Calculate pivots
        pivot = (prev_high + prev_low + prev_close) / 3
        r1 = 2 * pivot - prev_low
        s1 = 2 * pivot - prev_high
        r2 = pivot + (prev_high - prev_low)
        s2 = pivot - (prev_high - prev_low)

        current_price = self.df["close"].iloc[-1]

        # Score
        if current_price > r2:
            score = 0.9
            desc = f"Above R2 ({r2:.2f}) - Very bullish"
        elif current_price > r1:
            score = 0.6
            desc = f"Above R1 ({r1:.2f}) - Bullish"
        elif current_price > pivot:
            score = 0.2
            desc = f"Above Pivot ({pivot:.2f}) - Mildly bullish"
        elif current_price > s1:
            score = -0.2
            desc = f"Below Pivot ({pivot:.2f}) - Mildly bearish"
        elif current_price > s2:
            score = -0.6
            desc = f"Below S1 ({s1:.2f}) - Bearish"
        else:
            score = -0.9
            desc = f"Below S2 ({s2:.2f}) - Very bearish"

        return {
            "name": "Pivot Points",
            "score": round(score, 4),
            "description": desc,
            "pivot": round(pivot, 2),
            "r1": round(r1, 2),
            "r2": round(r2, 2),
            "s1": round(s1, 2),
            "s2": round(s2, 2),
            "current": round(current_price, 2),
        }

    # ==========================
    # STRATEGY 7: Volume Spike
    # ==========================

    def _volume_spike_strategy(self) -> Dict:
        """
        Volume spike detection:
        - Sudden volume increase + price up = Bullish
        - Sudden volume increase + price down = Bearish
        """
        if len(self.df) < 10 or "volume" not in self.df.columns:
            return None

        avg_vol = self.df["volume"].iloc[-20:].mean() if len(self.df) >= 20 else self.df["volume"].mean()
        current_vol = self.df["volume"].iloc[-1]
        prev_vol = self.df["volume"].iloc[-2]

        if avg_vol == 0:
            return None

        vol_ratio = current_vol / avg_vol
        price_change = (self.df["close"].iloc[-1] - self.df["close"].iloc[-2]) / self.df["close"].iloc[-2]

        if vol_ratio > 2.0:
            if price_change > 0:
                score = min(vol_ratio * 0.2, 0.8)
                desc = f"Volume spike ({vol_ratio:.1f}x avg) + price UP = Bullish"
            elif price_change < 0:
                score = max(-vol_ratio * 0.2, -0.8)
                desc = f"Volume spike ({vol_ratio:.1f}x avg) + price DOWN = Bearish"
            else:
                score = 0.0
                desc = f"Volume spike ({vol_ratio:.1f}x avg) but flat price"
        elif vol_ratio > 1.5:
            score = 0.2 if price_change > 0 else -0.2
            desc = f"Above avg volume ({vol_ratio:.1f}x)"
        else:
            score = 0.0
            desc = f"Normal volume ({vol_ratio:.1f}x avg)"

        return {
            "name": "Volume Spike",
            "score": round(float(score), 4),
            "description": desc,
            "volume_ratio": round(float(vol_ratio), 2),
        }

    # ==========================
    # STRATEGY 8: Candle Strength
    # ==========================

    def _candle_strength(self) -> Dict:
        """
        Analyze last few candles for buying/selling pressure.
        - Consecutive green candles = Bullish
        - Consecutive red candles = Bearish
        - Big body candles = Strong conviction
        """
        if len(self.df) < 5:
            return None

        last_5 = self.df.tail(5)

        green_count = sum(last_5["close"] > last_5["open"])
        red_count = 5 - green_count

        # Average body size relative to range
        body_sizes = abs(last_5["close"] - last_5["open"])
        ranges = last_5["high"] - last_5["low"]
        avg_body_ratio = (body_sizes / ranges.replace(0, np.nan)).mean()

        if pd.isna(avg_body_ratio):
            avg_body_ratio = 0.5

        # Score
        if green_count >= 4:
            score = 0.5 + avg_body_ratio * 0.3
            desc = f"{green_count}/5 green candles (Strong buying)"
        elif red_count >= 4:
            score = -(0.5 + avg_body_ratio * 0.3)
            desc = f"{red_count}/5 red candles (Strong selling)"
        elif green_count == 3:
            score = 0.2
            desc = f"Slightly more buyers ({green_count}/5 green)"
        elif red_count == 3:
            score = -0.2
            desc = f"Slightly more sellers ({red_count}/5 red)"
        else:
            score = 0.0
            desc = "Mixed candles - indecision"

        return {
            "name": "Candle Strength",
            "score": round(float(score), 4),
            "description": desc,
            "green_candles": int(green_count),
            "body_ratio": round(float(avg_body_ratio), 2),
        }