"""
Advanced Pattern Detector - More patterns + Trend analysis + Better scoring
"""

import pandas as pd
import numpy as np
from typing import List, Dict
from src.utils.constants import PatternType
from src.utils.logger import logger


class PatternDetector:
    """Detects candlestick patterns + price action + trend signals."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.patterns_detected: List[Dict] = []
        logger.info(f"PatternDetector initialized with {len(df)} candles")

    def detect_all(self) -> List[Dict]:
        """Run ALL pattern detectors."""
        self.patterns_detected = []

        detectors = [
            # Single candle patterns
            self._detect_doji,
            self._detect_hammer,
            self._detect_inverted_hammer,
            self._detect_shooting_star,
            self._detect_spinning_top,
            self._detect_marubozu,

            # Two candle patterns
            self._detect_bullish_engulfing,
            self._detect_bearish_engulfing,
            self._detect_piercing_line,
            self._detect_dark_cloud_cover,
            self._detect_tweezer_top,
            self._detect_tweezer_bottom,

            # Three candle patterns
            self._detect_morning_star,
            self._detect_evening_star,
            self._detect_three_white_soldiers,
            self._detect_three_black_crows,

            # Trend / Price action signals
            self._detect_higher_highs,
            self._detect_lower_lows,
            self._detect_breakout,
            self._detect_trend_reversal,
            self._detect_momentum_shift,
        ]

        for detector in detectors:
            try:
                detector()
            except Exception as e:
                logger.error(f"Error in {detector.__name__}: {e}")

        logger.info(f"Detected {len(self.patterns_detected)} pattern occurrences")
        return self.patterns_detected

    def get_latest_patterns(self, n: int = 5) -> List[Dict]:
        """Get patterns from the last n candles."""
        if not self.patterns_detected:
            self.detect_all()

        last_n_dates = self.df.index[-n:]
        recent = [p for p in self.patterns_detected if p["date"] in last_n_dates]
        return recent

    def get_signal_score(self) -> float:
        """
        IMPROVED: Calculate signal from multiple sources.
        Combines: recent patterns + overall trend + momentum
        """
        recent = self.get_latest_patterns(n=10)

        # 1. Pattern score from recent detections
        pattern_score = 0.0
        if recent:
            scores = []
            for p in recent:
                if p["type"] == PatternType.BULLISH.value:
                    scores.append(p.get("strength", 0.5))
                elif p["type"] == PatternType.BEARISH.value:
                    scores.append(-p.get("strength", 0.5))
                else:
                    scores.append(0.0)
            pattern_score = np.mean(scores)

        # 2. Price trend score (last 10 candles)
        trend_score = self._calculate_trend_score()

        # 3. Momentum score
        momentum_score = self._calculate_momentum_score()

        # 4. Volume trend
        volume_score = self._calculate_volume_score()

        # Weighted combination
        combined = (
            pattern_score * 0.3
            + trend_score * 0.3
            + momentum_score * 0.25
            + volume_score * 0.15
        )

        return float(np.clip(combined, -1.0, 1.0))

    def _calculate_trend_score(self) -> float:
        """Score based on recent price trend direction."""
        if len(self.df) < 10:
            return 0.0

        recent = self.df.tail(10)
        closes = recent["close"].values

        # Linear regression slope
        x = np.arange(len(closes))
        slope = np.polyfit(x, closes, 1)[0]

        # Normalize slope relative to price
        avg_price = np.mean(closes)
        normalized_slope = slope / avg_price * 100  # As percentage

        # Convert to -1 to +1 score
        if normalized_slope > 0.5:
            return min(normalized_slope / 2, 1.0)
        elif normalized_slope < -0.5:
            return max(normalized_slope / 2, -1.0)
        return normalized_slope / 2

    def _calculate_momentum_score(self) -> float:
        """Score based on price momentum (rate of change)."""
        if len(self.df) < 15:
            return 0.0

        close = self.df["close"]

        # 5-day rate of change
        roc_5 = (close.iloc[-1] - close.iloc[-5]) / close.iloc[-5]

        # 10-day rate of change
        roc_10 = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]

        # Average momentum
        momentum = (roc_5 * 0.6 + roc_10 * 0.4) * 10  # Scale up

        return float(np.clip(momentum, -1.0, 1.0))

    def _calculate_volume_score(self) -> float:
        """Score based on volume trend confirming price."""
        if len(self.df) < 10 or "volume" not in self.df.columns:
            return 0.0

        recent = self.df.tail(10)
        avg_vol = self.df["volume"].tail(20).mean()

        # Recent volume vs average
        recent_vol = recent["volume"].mean()
        vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0

        # Price direction
        price_up = recent["close"].iloc[-1] > recent["close"].iloc[0]

        if price_up and vol_ratio > 1.2:
            return 0.5   # Rising price + high volume = bullish
        elif not price_up and vol_ratio > 1.2:
            return -0.5  # Falling price + high volume = bearish
        elif price_up and vol_ratio < 0.8:
            return -0.2  # Rising price + low volume = weak bullish
        elif not price_up and vol_ratio < 0.8:
            return 0.2   # Falling price + low volume = weak bearish

        return 0.0

    def _has_volume_confirmation(self, idx: int, threshold: float = 1.3) -> bool:
        """Check if current candle has above-average volume."""
        if "volume" not in self.df.columns or idx < 20:
            return True

        avg_volume = self.df["volume"].iloc[max(0, idx - 20):idx].mean()
        current_volume = self.df.iloc[idx]["volume"]
        return current_volume > (avg_volume * threshold)

    def _is_downtrend(self, idx: int, lookback: int = 5) -> bool:
        """Check if we're in a downtrend."""
        if idx < lookback:
            return False
        closes = self.df["close"].iloc[idx - lookback:idx]
        return closes.iloc[-1] < closes.iloc[0]

    def _is_uptrend(self, idx: int, lookback: int = 5) -> bool:
        """Check if we're in an uptrend."""
        if idx < lookback:
            return False
        closes = self.df["close"].iloc[idx - lookback:idx]
        return closes.iloc[-1] > closes.iloc[0]

    # =====================
    # SINGLE CANDLE PATTERNS
    # =====================

    def _detect_doji(self):
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            candle_range = row["high"] - row["low"]
            if candle_range == 0:
                continue

            body = abs(row["close"] - row["open"])
            if body / candle_range < 0.1:
                signal_type = PatternType.NEUTRAL.value
                strength = 0.3

                # Doji after trend = reversal signal
                if self._is_downtrend(i):
                    signal_type = PatternType.BULLISH.value
                    strength = 0.5
                elif self._is_uptrend(i):
                    signal_type = PatternType.BEARISH.value
                    strength = 0.5

                self.patterns_detected.append({
                    "pattern": "Doji",
                    "type": signal_type,
                    "date": self.df.index[i],
                    "strength": strength,
                    "description": "Indecision — possible reversal",
                })

    def _detect_hammer(self):
        for i in range(2, len(self.df)):
            row = self.df.iloc[i]
            candle_range = row["high"] - row["low"]
            if candle_range == 0:
                continue

            body = abs(row["close"] - row["open"])
            lower_wick = min(row["open"], row["close"]) - row["low"]
            upper_wick = row["high"] - max(row["open"], row["close"])

            if lower_wick >= 2 * body and upper_wick <= body * 0.5:
                if self._is_downtrend(i):
                    vol = self._has_volume_confirmation(i)
                    self.patterns_detected.append({
                        "pattern": "Hammer",
                        "type": PatternType.BULLISH.value,
                        "date": self.df.index[i],
                        "strength": 0.8 if vol else 0.55,
                        "description": f"Bullish reversal {'(Vol ✅)' if vol else ''}",
                    })

    def _detect_inverted_hammer(self):
        for i in range(2, len(self.df)):
            row = self.df.iloc[i]
            candle_range = row["high"] - row["low"]
            if candle_range == 0:
                continue

            body = abs(row["close"] - row["open"])
            upper_wick = row["high"] - max(row["open"], row["close"])
            lower_wick = min(row["open"], row["close"]) - row["low"]

            if upper_wick >= 2 * body and lower_wick <= body * 0.5:
                if self._is_downtrend(i):
                    self.patterns_detected.append({
                        "pattern": "Inverted Hammer",
                        "type": PatternType.BULLISH.value,
                        "date": self.df.index[i],
                        "strength": 0.6,
                        "description": "Possible bullish reversal after downtrend",
                    })

    def _detect_shooting_star(self):
        """Shooting Star: Long upper wick in uptrend = bearish"""
        for i in range(2, len(self.df)):
            row = self.df.iloc[i]
            candle_range = row["high"] - row["low"]
            if candle_range == 0:
                continue

            body = abs(row["close"] - row["open"])
            upper_wick = row["high"] - max(row["open"], row["close"])
            lower_wick = min(row["open"], row["close"]) - row["low"]

            if upper_wick >= 2 * body and lower_wick <= body * 0.5:
                if self._is_uptrend(i):
                    self.patterns_detected.append({
                        "pattern": "Shooting Star",
                        "type": PatternType.BEARISH.value,
                        "date": self.df.index[i],
                        "strength": 0.7,
                        "description": "Bearish reversal signal in uptrend",
                    })

    def _detect_spinning_top(self):
        """Spinning Top: Small body, both wicks present = indecision"""
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            candle_range = row["high"] - row["low"]
            if candle_range == 0:
                continue

            body = abs(row["close"] - row["open"])
            upper_wick = row["high"] - max(row["open"], row["close"])
            lower_wick = min(row["open"], row["close"]) - row["low"]
            body_ratio = body / candle_range

            if 0.1 < body_ratio < 0.3 and upper_wick > body * 0.5 and lower_wick > body * 0.5:
                self.patterns_detected.append({
                    "pattern": "Spinning Top",
                    "type": PatternType.NEUTRAL.value,
                    "date": self.df.index[i],
                    "strength": 0.25,
                    "description": "Indecision — small body with both wicks",
                })

    def _detect_marubozu(self):
        """Marubozu: Full body, no/tiny wicks = strong conviction"""
        for i in range(len(self.df)):
            row = self.df.iloc[i]
            candle_range = row["high"] - row["low"]
            if candle_range == 0:
                continue

            body = abs(row["close"] - row["open"])
            body_ratio = body / candle_range

            if body_ratio > 0.9:
                is_bull = row["close"] > row["open"]
                self.patterns_detected.append({
                    "pattern": f"{'Bullish' if is_bull else 'Bearish'} Marubozu",
                    "type": PatternType.BULLISH.value if is_bull else PatternType.BEARISH.value,
                    "date": self.df.index[i],
                    "strength": 0.75,
                    "description": f"Strong {'buying' if is_bull else 'selling'} pressure — full body candle",
                })

    # =====================
    # TWO CANDLE PATTERNS
    # =====================

    def _detect_bullish_engulfing(self):
        for i in range(1, len(self.df)):
            prev = self.df.iloc[i - 1]
            curr = self.df.iloc[i]

            if (
                prev["close"] < prev["open"]
                and curr["close"] > curr["open"]
                and curr["open"] <= prev["close"]
                and curr["close"] >= prev["open"]
            ):
                vol = self._has_volume_confirmation(i)
                self.patterns_detected.append({
                    "pattern": "Bullish Engulfing",
                    "type": PatternType.BULLISH.value,
                    "date": self.df.index[i],
                    "strength": 0.85 if vol else 0.6,
                    "description": f"Strong bullish reversal {'(Vol ✅)' if vol else ''}",
                })

    def _detect_bearish_engulfing(self):
        for i in range(1, len(self.df)):
            prev = self.df.iloc[i - 1]
            curr = self.df.iloc[i]

            if (
                prev["close"] > prev["open"]
                and curr["close"] < curr["open"]
                and curr["open"] >= prev["close"]
                and curr["close"] <= prev["open"]
            ):
                vol = self._has_volume_confirmation(i)
                self.patterns_detected.append({
                    "pattern": "Bearish Engulfing",
                    "type": PatternType.BEARISH.value,
                    "date": self.df.index[i],
                    "strength": 0.85 if vol else 0.6,
                    "description": f"Strong bearish reversal {'(Vol ✅)' if vol else ''}",
                })

    def _detect_piercing_line(self):
        """Piercing Line: Bullish - opens below prev low, closes above midpoint"""
        for i in range(1, len(self.df)):
            prev = self.df.iloc[i - 1]
            curr = self.df.iloc[i]

            prev_mid = (prev["open"] + prev["close"]) / 2

            if (
                prev["close"] < prev["open"]          # Previous bearish
                and curr["close"] > curr["open"]       # Current bullish
                and curr["open"] < prev["close"]       # Opens below prev close
                and curr["close"] > prev_mid           # Closes above prev midpoint
                and curr["close"] < prev["open"]       # But not above prev open
            ):
                self.patterns_detected.append({
                    "pattern": "Piercing Line",
                    "type": PatternType.BULLISH.value,
                    "date": self.df.index[i],
                    "strength": 0.7,
                    "description": "Bullish reversal — pierces into previous bearish candle",
                })

    def _detect_dark_cloud_cover(self):
        """Dark Cloud Cover: Bearish - opens above prev high, closes below midpoint"""
        for i in range(1, len(self.df)):
            prev = self.df.iloc[i - 1]
            curr = self.df.iloc[i]

            prev_mid = (prev["open"] + prev["close"]) / 2

            if (
                prev["close"] > prev["open"]
                and curr["close"] < curr["open"]
                and curr["open"] > prev["close"]
                and curr["close"] < prev_mid
                and curr["close"] > prev["open"]
            ):
                self.patterns_detected.append({
                    "pattern": "Dark Cloud Cover",
                    "type": PatternType.BEARISH.value,
                    "date": self.df.index[i],
                    "strength": 0.7,
                    "description": "Bearish reversal — dark cloud over previous bullish candle",
                })

    def _detect_tweezer_top(self):
        """Tweezer Top: Both candles have same high = resistance"""
        for i in range(1, len(self.df)):
            prev = self.df.iloc[i - 1]
            curr = self.df.iloc[i]

            tolerance = (prev["high"] - prev["low"]) * 0.05
            if (
                abs(prev["high"] - curr["high"]) <= tolerance
                and prev["close"] > prev["open"]
                and curr["close"] < curr["open"]
            ):
                self.patterns_detected.append({
                    "pattern": "Tweezer Top",
                    "type": PatternType.BEARISH.value,
                    "date": self.df.index[i],
                    "strength": 0.65,
                    "description": "Bearish reversal — double top rejection",
                })

    def _detect_tweezer_bottom(self):
        """Tweezer Bottom: Both candles have same low = support"""
        for i in range(1, len(self.df)):
            prev = self.df.iloc[i - 1]
            curr = self.df.iloc[i]

            tolerance = (prev["high"] - prev["low"]) * 0.05
            if (
                abs(prev["low"] - curr["low"]) <= tolerance
                and prev["close"] < prev["open"]
                and curr["close"] > curr["open"]
            ):
                self.patterns_detected.append({
                    "pattern": "Tweezer Bottom",
                    "type": PatternType.BULLISH.value,
                    "date": self.df.index[i],
                    "strength": 0.65,
                    "description": "Bullish reversal — double bottom support",
                })

    # =====================
    # THREE CANDLE PATTERNS
    # =====================

    def _detect_morning_star(self):
        for i in range(2, len(self.df)):
            first = self.df.iloc[i - 2]
            second = self.df.iloc[i - 1]
            third = self.df.iloc[i]

            first_bearish = first["close"] < first["open"]
            second_small = abs(second["close"] - second["open"]) < abs(first["close"] - first["open"]) * 0.3
            third_bullish = third["close"] > third["open"]
            third_recovers = third["close"] > (first["open"] + first["close"]) / 2

            if first_bearish and second_small and third_bullish and third_recovers:
                self.patterns_detected.append({
                    "pattern": "Morning Star",
                    "type": PatternType.BULLISH.value,
                    "date": self.df.index[i],
                    "strength": 0.85,
                    "description": "Strong 3-candle bullish reversal",
                })

    def _detect_evening_star(self):
        for i in range(2, len(self.df)):
            first = self.df.iloc[i - 2]
            second = self.df.iloc[i - 1]
            third = self.df.iloc[i]

            first_bullish = first["close"] > first["open"]
            second_small = abs(second["close"] - second["open"]) < abs(first["close"] - first["open"]) * 0.3
            third_bearish = third["close"] < third["open"]
            third_drops = third["close"] < (first["open"] + first["close"]) / 2

            if first_bullish and second_small and third_bearish and third_drops:
                self.patterns_detected.append({
                    "pattern": "Evening Star",
                    "type": PatternType.BEARISH.value,
                    "date": self.df.index[i],
                    "strength": 0.85,
                    "description": "Strong 3-candle bearish reversal",
                })

    def _detect_three_white_soldiers(self):
        for i in range(2, len(self.df)):
            candles = [self.df.iloc[i - 2], self.df.iloc[i - 1], self.df.iloc[i]]

            all_bullish = all(c["close"] > c["open"] for c in candles)
            ascending = candles[2]["close"] > candles[1]["close"] > candles[0]["close"]

            if all_bullish and ascending:
                self.patterns_detected.append({
                    "pattern": "Three White Soldiers",
                    "type": PatternType.BULLISH.value,
                    "date": self.df.index[i],
                    "strength": 0.9,
                    "description": "Strong bullish — 3 ascending green candles",
                })

    def _detect_three_black_crows(self):
        for i in range(2, len(self.df)):
            candles = [self.df.iloc[i - 2], self.df.iloc[i - 1], self.df.iloc[i]]

            all_bearish = all(c["close"] < c["open"] for c in candles)
            descending = candles[2]["close"] < candles[1]["close"] < candles[0]["close"]

            if all_bearish and descending:
                self.patterns_detected.append({
                    "pattern": "Three Black Crows",
                    "type": PatternType.BEARISH.value,
                    "date": self.df.index[i],
                    "strength": 0.9,
                    "description": "Strong bearish — 3 descending red candles",
                })

    # =====================
    # TREND / PRICE ACTION
    # =====================

    def _detect_higher_highs(self):
        """Higher Highs + Higher Lows = Uptrend"""
        if len(self.df) < 6:
            return

        for i in range(5, len(self.df)):
            recent = self.df.iloc[i - 5 : i + 1]
            highs = recent["high"].values
            lows = recent["low"].values

            hh = all(highs[j] > highs[j - 1] for j in range(1, len(highs)))
            hl = all(lows[j] > lows[j - 1] for j in range(1, len(lows)))

            if hh and hl:
                self.patterns_detected.append({
                    "pattern": "Higher Highs & Higher Lows",
                    "type": PatternType.BULLISH.value,
                    "date": self.df.index[i],
                    "strength": 0.8,
                    "description": "Strong uptrend — consecutive higher highs and lows",
                })

    def _detect_lower_lows(self):
        """Lower Lows + Lower Highs = Downtrend"""
        if len(self.df) < 6:
            return

        for i in range(5, len(self.df)):
            recent = self.df.iloc[i - 5 : i + 1]
            highs = recent["high"].values
            lows = recent["low"].values

            lh = all(highs[j] < highs[j - 1] for j in range(1, len(highs)))
            ll = all(lows[j] < lows[j - 1] for j in range(1, len(lows)))

            if lh and ll:
                self.patterns_detected.append({
                    "pattern": "Lower Lows & Lower Highs",
                    "type": PatternType.BEARISH.value,
                    "date": self.df.index[i],
                    "strength": 0.8,
                    "description": "Strong downtrend — consecutive lower lows and highs",
                })

    def _detect_breakout(self):
        """Price breaks above recent high or below recent low"""
        if len(self.df) < 21:
            return

        for i in range(20, len(self.df)):
            lookback = self.df.iloc[i - 20 : i]
            curr = self.df.iloc[i]

            high_20 = lookback["high"].max()
            low_20 = lookback["low"].min()

            # Breakout above
            if curr["close"] > high_20:
                vol = self._has_volume_confirmation(i)
                self.patterns_detected.append({
                    "pattern": "Bullish Breakout",
                    "type": PatternType.BULLISH.value,
                    "date": self.df.index[i],
                    "strength": 0.85 if vol else 0.5,
                    "description": f"Broke above 20-period high (₹{high_20:.2f}) {'(Vol ✅)' if vol else ''}",
                })

            # Breakdown below
            elif curr["close"] < low_20:
                vol = self._has_volume_confirmation(i)
                self.patterns_detected.append({
                    "pattern": "Bearish Breakdown",
                    "type": PatternType.BEARISH.value,
                    "date": self.df.index[i],
                    "strength": 0.85 if vol else 0.5,
                    "description": f"Broke below 20-period low (₹{low_20:.2f}) {'(Vol ✅)' if vol else ''}",
                })

    def _detect_trend_reversal(self):
        """Detect when short-term trend changes direction"""
        if len(self.df) < 15:
            return

        for i in range(10, len(self.df)):
            prev_5 = self.df["close"].iloc[i - 10 : i - 5].mean()
            curr_5 = self.df["close"].iloc[i - 5 : i].mean()
            last_price = self.df["close"].iloc[i]

            change = (curr_5 - prev_5) / prev_5

            # Trend was down, now turning up
            if change < -0.02 and last_price > curr_5:
                self.patterns_detected.append({
                    "pattern": "Bullish Trend Reversal",
                    "type": PatternType.BULLISH.value,
                    "date": self.df.index[i],
                    "strength": 0.65,
                    "description": "Price recovering above recent average after decline",
                })

            # Trend was up, now turning down
            elif change > 0.02 and last_price < curr_5:
                self.patterns_detected.append({
                    "pattern": "Bearish Trend Reversal",
                    "type": PatternType.BEARISH.value,
                    "date": self.df.index[i],
                    "strength": 0.65,
                    "description": "Price falling below recent average after rally",
                })

    def _detect_momentum_shift(self):
        """Detect when price momentum is shifting"""
        if len(self.df) < 12:
            return

        for i in range(10, len(self.df)):
            close = self.df["close"]

            roc_3 = (close.iloc[i] - close.iloc[i - 3]) / close.iloc[i - 3]
            roc_7 = (close.iloc[i] - close.iloc[i - 7]) / close.iloc[i - 7]

            # Short-term bullish momentum stronger than medium-term
            if roc_3 > 0.02 and roc_3 > roc_7:
                self.patterns_detected.append({
                    "pattern": "Bullish Momentum",
                    "type": PatternType.BULLISH.value,
                    "date": self.df.index[i],
                    "strength": min(abs(roc_3) * 10, 0.8),
                    "description": f"Accelerating upward momentum ({roc_3:.1%} in 3 days)",
                })

            elif roc_3 < -0.02 and roc_3 < roc_7:
                self.patterns_detected.append({
                    "pattern": "Bearish Momentum",
                    "type": PatternType.BEARISH.value,
                    "date": self.df.index[i],
                    "strength": min(abs(roc_3) * 10, 0.8),
                    "description": f"Accelerating downward momentum ({roc_3:.1%} in 3 days)",
                })