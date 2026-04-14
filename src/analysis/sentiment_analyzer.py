"""
Improved Sentiment Analyzer - Financial keyword boosting + VADER + FinBERT
"""

from typing import List, Dict
import numpy as np
import re
from src.utils.constants import Sentiment
from src.utils.logger import logger


# Financial keywords that VADER misses
BULLISH_KEYWORDS = [
    "surge", "surges", "surging", "soar", "soars", "soaring",
    "rally", "rallies", "rallying", "jump", "jumps", "jumping",
    "gain", "gains", "gaining", "rise", "rises", "rising",
    "bull", "bullish", "breakout", "upgrade", "upgraded",
    "outperform", "buy", "accumulate", "strong buy",
    "profit", "profits", "profitable", "beat", "beats", "beating",
    "record high", "all-time high", "52-week high",
    "growth", "growing", "revenue growth", "earnings beat",
    "positive", "optimistic", "upside", "recovery", "recovering",
    "dividend", "bonus", "buyback", "expansion",
    "strong results", "better than expected", "exceeds",
    "outpace", "boom", "booming", "upbeat", "robust",
]

BEARISH_KEYWORDS = [
    "crash", "crashes", "crashing", "plunge", "plunges", "plunging",
    "fall", "falls", "falling", "drop", "drops", "dropping",
    "decline", "declines", "declining", "slump", "slumps", "slumping",
    "bear", "bearish", "breakdown", "downgrade", "downgraded",
    "underperform", "sell", "reduce", "avoid",
    "loss", "losses", "losing", "miss", "misses", "missing",
    "record low", "52-week low", "all-time low",
    "shrink", "shrinking", "revenue decline", "earnings miss",
    "negative", "pessimistic", "downside", "recession",
    "debt", "default", "fraud", "scam", "investigation",
    "weak results", "worse than expected", "disappoints",
    "layoff", "layoffs", "restructuring", "warning",
    "correction", "selloff", "sell-off", "tank", "tanks",
]


class SentimentAnalyzer:
    """Enhanced sentiment analyzer with financial keyword boosting."""

    def __init__(self, model_type: str = "vader"):
        self.model_type = model_type
        self._model = None
        self._tokenizer = None
        self._vader = None
        self._pipeline = None

        if model_type == "finbert":
            self._load_finbert()
        else:
            self._load_vader()

        logger.info(f"SentimentAnalyzer initialized with '{model_type}'")

    def _load_finbert(self):
        try:
            from transformers import (
                AutoTokenizer,
                AutoModelForSequenceClassification,
                pipeline,
            )
            model_name = "ProsusAI/finbert"
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self._model,
                tokenizer=self._tokenizer,
                top_k=None,
            )
            logger.info("FinBERT loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load FinBERT: {e}. Falling back to VADER.")
            self.model_type = "vader"
            self._load_vader()

    def _load_vader(self):
        try:
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            import nltk
            nltk.download("vader_lexicon", quiet=True)
            self._vader = SentimentIntensityAnalyzer()

            # Add financial terms to VADER lexicon
            financial_lexicon = {
                "surge": 3.0, "surges": 3.0, "surging": 3.0,
                "soar": 3.0, "soars": 3.0, "rally": 2.5,
                "bullish": 2.5, "breakout": 2.0, "upgrade": 2.0,
                "outperform": 2.0, "buy": 1.5, "accumulate": 1.5,
                "profit": 2.0, "growth": 1.5, "record high": 3.0,
                "beat": 2.0, "beats": 2.0, "dividend": 1.5,
                "strong results": 2.5, "better than expected": 2.5,
                "crash": -3.5, "crashes": -3.5, "plunge": -3.0,
                "plunges": -3.0, "bearish": -2.5, "downgrade": -2.5,
                "underperform": -2.0, "sell": -1.5, "avoid": -1.5,
                "loss": -2.0, "losses": -2.0, "decline": -2.0,
                "slump": -2.5, "selloff": -3.0, "sell-off": -3.0,
                "weak results": -2.5, "worse than expected": -2.5,
                "fraud": -3.5, "scam": -3.5, "default": -3.0,
                "warning": -1.5, "layoff": -2.0, "layoffs": -2.0,
            }
            self._vader.lexicon.update(financial_lexicon)
            logger.info("VADER loaded with financial lexicon")
        except Exception as e:
            logger.error(f"Failed to load VADER: {e}")

    def _keyword_score(self, text: str) -> float:
        """
        Score text based on financial keywords.
        Returns -1 to +1.
        """
        text_lower = text.lower()

        bull_count = sum(1 for kw in BULLISH_KEYWORDS if kw in text_lower)
        bear_count = sum(1 for kw in BEARISH_KEYWORDS if kw in text_lower)

        total = bull_count + bear_count
        if total == 0:
            return 0.0

        score = (bull_count - bear_count) / total
        return float(np.clip(score, -1.0, 1.0))

    def _price_movement_score(self, text: str) -> float:
        """
        Extract percentage movements from text.
        "shares jump 4%" → positive
        "stock drops 3%" → negative
        """
        # Find patterns like "up 4%", "down 3.5%", "rises 2%", "falls 5%"
        patterns = [
            (r"(?:up|rise[sd]?|gain[sd]?|jump[sd]?|surge[sd]?)\s*(?:by\s*)?(\d+\.?\d*)\s*%", 1),
            (r"(\d+\.?\d*)\s*%\s*(?:up|rise|gain|jump|surge|higher)", 1),
            (r"(?:down|fall[sd]?|drop[sd]?|decline[sd]?|slump[sd]?|slip[sd]?)\s*(?:by\s*)?(\d+\.?\d*)\s*%", -1),
            (r"(\d+\.?\d*)\s*%\s*(?:down|fall|drop|decline|lower)", -1),
        ]

        scores = []
        text_lower = text.lower()

        for pattern, direction in patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                try:
                    pct = float(match)
                    # Bigger moves = stronger signal, cap at 1.0
                    strength = min(pct / 5.0, 1.0) * direction
                    scores.append(strength)
                except ValueError:
                    continue

        if scores:
            return float(np.clip(np.mean(scores), -1.0, 1.0))
        return 0.0

    def analyze_text(self, text: str) -> Dict:
        """
        Analyze sentiment combining multiple methods.
        """
        if not text:
            return {
                "sentiment": Sentiment.NEUTRAL.value,
                "score": 0.0,
                "confidence": 0.0,
            }

        # Method 1: VADER or FinBERT base score
        if self.model_type == "finbert":
            base_result = self._analyze_finbert(text)
        else:
            base_result = self._analyze_vader(text)

        base_score = base_result["score"]

        # Method 2: Keyword score
        keyword_score = self._keyword_score(text)

        # Method 3: Price movement score
        price_score = self._price_movement_score(text)

        # Combine: base 50% + keywords 30% + price 20%
        combined_score = (
            base_score * 0.5
            + keyword_score * 0.3
            + price_score * 0.2
        )

        # Determine sentiment with tighter thresholds
        if combined_score > 0.05:
            sentiment = Sentiment.POSITIVE.value
        elif combined_score < -0.05:
            sentiment = Sentiment.NEGATIVE.value
        else:
            sentiment = Sentiment.NEUTRAL.value

        confidence = min(abs(combined_score) * 1.5, 1.0)

        return {
            "sentiment": sentiment,
            "score": round(combined_score, 4),
            "confidence": round(confidence, 4),
        }

    def _analyze_finbert(self, text: str) -> Dict:
        try:
            text = text[:512]
            results = self._pipeline(text)

            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    results = results[0]

                scores = {r["label"]: r["score"] for r in results}
                pos = scores.get("positive", 0)
                neg = scores.get("negative", 0)
                neu = scores.get("neutral", 0)

                compound = pos - neg

                if pos > neg and pos > neu:
                    sentiment = Sentiment.POSITIVE.value
                    confidence = pos
                elif neg > pos and neg > neu:
                    sentiment = Sentiment.NEGATIVE.value
                    confidence = neg
                else:
                    sentiment = Sentiment.NEUTRAL.value
                    confidence = neu

                return {
                    "sentiment": sentiment,
                    "score": round(compound, 4),
                    "confidence": round(confidence, 4),
                }
        except Exception as e:
            logger.error(f"FinBERT error: {e}")

        return {"sentiment": Sentiment.NEUTRAL.value, "score": 0.0, "confidence": 0.0}

    def _analyze_vader(self, text: str) -> Dict:
        try:
            scores = self._vader.polarity_scores(text)
            compound = scores["compound"]

            if compound >= 0.05:
                sentiment = Sentiment.POSITIVE.value
                confidence = scores["pos"]
            elif compound <= -0.05:
                sentiment = Sentiment.NEGATIVE.value
                confidence = scores["neg"]
            else:
                sentiment = Sentiment.NEUTRAL.value
                confidence = scores["neu"]

            return {
                "sentiment": sentiment,
                "score": round(compound, 4),
                "confidence": round(confidence, 4),
            }
        except Exception as e:
            logger.error(f"VADER error: {e}")
            return {"sentiment": Sentiment.NEUTRAL.value, "score": 0.0, "confidence": 0.0}

    def analyze_articles(self, articles: List[Dict]) -> Dict:
        """Analyze list of articles with improved aggregation."""
        if not articles:
            return {
                "overall_sentiment": Sentiment.NEUTRAL.value,
                "overall_score": 0.0,
                "overall_confidence": 0.0,
                "article_sentiments": [],
            }

        article_results = []
        scores = []

        for article in articles:
            text = article.get("combined_text", article.get("title", ""))
            result = self.analyze_text(text)
            result["title"] = article.get("title", "")[:100]
            article_results.append(result)
            scores.append(result["score"])

        # Weighted average: stronger sentiments get more weight
        weights = [abs(s) + 0.1 for s in scores]  # +0.1 to avoid zero weight
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        total_weight = sum(weights)
        avg_score = weighted_sum / total_weight if total_weight > 0 else 0

        avg_confidence = np.mean([r["confidence"] for r in article_results])

        # Tighter thresholds
        if avg_score > 0.05:
            overall = Sentiment.POSITIVE.value
        elif avg_score < -0.05:
            overall = Sentiment.NEGATIVE.value
        else:
            overall = Sentiment.NEUTRAL.value

        return {
            "overall_sentiment": overall,
            "overall_score": round(avg_score, 4),
            "overall_confidence": round(avg_confidence, 4),
            "article_sentiments": article_results,
            "positive_count": sum(1 for r in article_results if r["sentiment"] == "Positive"),
            "negative_count": sum(1 for r in article_results if r["sentiment"] == "Negative"),
            "neutral_count": sum(1 for r in article_results if r["sentiment"] == "Neutral"),
        }