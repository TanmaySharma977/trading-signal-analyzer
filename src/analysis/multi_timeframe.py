"""
Multi-Timeframe Analysis - Analyze same stock across different timeframes
"""

import pandas as pd
from src.data_ingestion.market_data import MarketDataFetcher
from src.analysis.pattern_detector import PatternDetector
from src.analysis.technical_indicators import TechnicalIndicators
from src.utils.logger import logger


class MultiTimeframeAnalyzer:
    """
    Analyze stock on multiple timeframes.
    If daily + weekly + monthly all agree → MUCH stronger signal.
    """

    def __init__(self, exchange: str = "NSE"):
        self.fetcher = MarketDataFetcher(exchange)
        self.timeframes = {
            "short": {"period": "1mo", "interval": "1d", "weight": 0.3},
            "medium": {"period": "6mo", "interval": "1wk", "weight": 0.4},
            "long": {"period": "2y", "interval": "1mo", "weight": 0.3},
        }

    def analyze(self, symbol: str) -> dict:
        """
        Analyze across all timeframes and return combined score.
        """
        results = {}
        weighted_score = 0.0

        for tf_name, tf_config in self.timeframes.items():
            logger.info(f"Analyzing {symbol} on {tf_name} timeframe")

            df = self.fetcher.fetch_historical(
                symbol,
                period=tf_config["period"],
                interval=tf_config["interval"],
            )

            if df.empty:
                continue

            # Pattern score
            detector = PatternDetector(df)
            detector.detect_all()
            pattern_score = detector.get_signal_score()

            # Technical score
            df = TechnicalIndicators.add_all(df)
            tech_score = TechnicalIndicators.get_trend_signal(df)

            # Combined score for this timeframe
            tf_score = (pattern_score * 0.5) + (tech_score * 0.5)

            results[tf_name] = {
                "pattern_score": round(pattern_score, 4),
                "technical_score": round(tech_score, 4),
                "combined": round(tf_score, 4),
            }

            weighted_score += tf_score * tf_config["weight"]

        # Agreement check — are all timeframes pointing same direction?
        directions = [
            1 if r["combined"] > 0.1 else -1 if r["combined"] < -0.1 else 0
            for r in results.values()
        ]

        all_agree = len(set(d for d in directions if d != 0)) <= 1
        agreement_bonus = 0.2 if all_agree and len(directions) > 1 else 0.0

        if weighted_score > 0:
            weighted_score += agreement_bonus
        elif weighted_score < 0:
            weighted_score -= agreement_bonus

        return {
            "multi_tf_score": round(weighted_score, 4),
            "timeframes": results,
            "all_agree": all_agree,
            "confidence_boost": agreement_bonus > 0,
        }