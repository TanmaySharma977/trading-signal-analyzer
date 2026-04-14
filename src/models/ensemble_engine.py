"""
Ensemble Decision Engine - Combines Rule-Based + ML for best accuracy
"""

from typing import Dict
from src.models.rule_based_engine import RuleBasedEngine
from src.models.ml_engine import MLEngine
from src.utils.constants import Signal
from src.utils.logger import logger
import pandas as pd


class EnsembleEngine:
    """
    Combines rule-based + ML predictions.
    
    If both agree → HIGH confidence signal
    If they disagree → HOLD (safer)
    """

    def __init__(self):
        self.rule_engine = RuleBasedEngine()
        self.ml_engine = MLEngine(model_type="random_forest")
        logger.info("EnsembleEngine initialized")

    def generate_signal(
        self,
        df: pd.DataFrame,
        pattern_score: float,
        sentiment_score: float,
        technical_score: float,
        include_news: bool = True,
    ) -> Dict:

        # 1. Rule-based signal
        rule_result = self.rule_engine.generate_signal(
            pattern_score=pattern_score,
            sentiment_score=sentiment_score,
            technical_score=technical_score,
            include_news=include_news,
        )

        # 2. ML signal (train + predict)
        train_metrics = self.ml_engine.train(df, pattern_score, sentiment_score)
        ml_result = self.ml_engine.predict(df, pattern_score, sentiment_score)

        # 3. Combine
        rule_signal = rule_result["signal"]
        ml_signal = ml_result["signal"]

        # Agreement = strong signal
        if rule_signal == ml_signal:
            final_signal = rule_signal
            confidence = (rule_result["confidence"] + ml_result["confidence"]) / 2
            confidence = min(confidence * 1.2, 100)  # Boost for agreement
            agreement = "✅ Both engines agree"
        else:
            # Disagreement = lower confidence, lean towards HOLD
            if rule_result["confidence"] > ml_result["confidence"]:
                final_signal = rule_signal
            else:
                final_signal = ml_signal

            confidence = abs(rule_result["confidence"] - ml_result["confidence"]) / 2
            agreement = "⚠️ Engines disagree — lower confidence"

            # If confidence is too low, default to HOLD
            if confidence < 30:
                final_signal = "HOLD"

        return {
            "signal": final_signal,
            "confidence": round(confidence, 1),
            "agreement": agreement,
            "rule_based": rule_result,
            "ml_based": ml_result,
            "ml_accuracy": train_metrics.get("cv_accuracy", 0),
            "reason": f"Rule: {rule_signal} | ML: {ml_signal} | {agreement}",
        }