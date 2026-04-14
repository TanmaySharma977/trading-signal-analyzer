"""
Decision Engine - Improved thresholds + multi-factor scoring
"""

from typing import Dict
from src.utils.constants import Signal
from src.utils.logger import logger


class RuleBasedEngine:
    """Improved rule-based decision engine."""

    def __init__(self, config: Dict = None):
        self.config = config or {
            "buy_threshold": 0.15,      # Was 0.3 — now more sensitive
            "sell_threshold": -0.15,     # Was -0.3 — now more sensitive
            "strong_buy": 0.4,
            "strong_sell": -0.4,
            "pattern_weight": 0.35,
            "sentiment_weight": 0.30,
            "technical_weight": 0.35,
        }
        logger.info("RuleBasedEngine initialized")

    def generate_signal(
        self,
        pattern_score: float,
        sentiment_score: float,
        technical_score: float = 0.0,
        include_news: bool = True,
        include_technical: bool = True,
    ) -> Dict:

        weights = {}
        scores = {}

        # Always include patterns
        weights["pattern"] = self.config["pattern_weight"]
        scores["pattern"] = pattern_score

        # Sentiment
        if include_news and sentiment_score != 0:
            weights["sentiment"] = self.config["sentiment_weight"]
            scores["sentiment"] = sentiment_score
        else:
            # Redistribute weight equally
            extra = self.config["sentiment_weight"]
            weights["pattern"] = self.config["pattern_weight"] + extra * 0.5

        # Technical
        if include_technical:
            weights["technical"] = self.config["technical_weight"]
            scores["technical"] = technical_score
        else:
            extra = self.config["technical_weight"]
            weights["pattern"] = weights.get("pattern", self.config["pattern_weight"]) + extra * 0.5

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        # Weighted composite score
        composite = sum(weights.get(k, 0) * scores.get(k, 0) for k in weights)

        # Agreement bonus: if all factors point same direction
        all_scores = [s for s in scores.values() if s != 0]
        if all_scores:
            all_positive = all(s > 0 for s in all_scores)
            all_negative = all(s < 0 for s in all_scores)

            if all_positive:
                composite *= 1.3  # 30% boost for agreement
            elif all_negative:
                composite *= 1.3

        # Determine signal
        if composite >= self.config["strong_buy"]:
            signal = Signal.BUY.value
            confidence = min(abs(composite) * 120, 95)
        elif composite >= self.config["buy_threshold"]:
            signal = Signal.BUY.value
            confidence = min(abs(composite) * 100, 80)
        elif composite <= self.config["strong_sell"]:
            signal = Signal.SELL.value
            confidence = min(abs(composite) * 120, 95)
        elif composite <= self.config["sell_threshold"]:
            signal = Signal.SELL.value
            confidence = min(abs(composite) * 100, 80)
        else:
            signal = Signal.HOLD.value
            confidence = max(20, (1 - abs(composite)) * 50)

        # Build reason
        reasons = []
        if pattern_score > 0.1:
            reasons.append(f"Bullish Patterns (+{pattern_score:.2f})")
        elif pattern_score < -0.1:
            reasons.append(f"Bearish Patterns ({pattern_score:.2f})")

        if include_news and sentiment_score != 0:
            if sentiment_score > 0.1:
                reasons.append(f"Positive News (+{sentiment_score:.2f})")
            elif sentiment_score < -0.1:
                reasons.append(f"Negative News ({sentiment_score:.2f})")
            else:
                reasons.append("Neutral News")

        if include_technical:
            if technical_score > 0.1:
                reasons.append(f"Bullish Indicators (+{technical_score:.2f})")
            elif technical_score < -0.1:
                reasons.append(f"Bearish Indicators ({technical_score:.2f})")

        reason = " + ".join(reasons) if reasons else "Weak/mixed signals"

        result = {
            "signal": signal,
            "confidence": round(confidence, 1),
            "composite_score": round(composite, 4),
            "reason": reason,
            "breakdown": {
                "pattern_score": round(pattern_score, 4),
                "sentiment_score": round(sentiment_score, 4),
                "technical_score": round(technical_score, 4),
                "weights": {k: round(v, 3) for k, v in weights.items()},
            },
        }

        logger.info(f"Signal: {signal} | Confidence: {confidence:.1f}% | {reason}")
        return result