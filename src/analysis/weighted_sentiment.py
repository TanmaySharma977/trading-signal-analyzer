"""
Weighted Sentiment - Recent articles get more weight
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.utils.logger import logger


class WeightedSentimentAnalyzer:
    """
    Weighs recent news more than old news.
    Today's news = 1.0x weight
    Yesterday = 0.7x
    2 days ago = 0.5x
    3+ days ago = 0.3x
    """

    def __init__(self, model_type: str = "vader"):
        self.analyzer = SentimentAnalyzer(model_type=model_type)
        self.decay_weights = {
            0: 1.0,    # Today
            1: 0.7,    # Yesterday
            2: 0.5,    # 2 days ago
            3: 0.3,    # 3+ days ago
        }

    def analyze_weighted(self, articles: List[Dict]) -> Dict:
        """Analyze with time-decay weighting."""
        if not articles:
            return {
                "weighted_score": 0.0,
                "sentiment": "Neutral",
                "confidence": 0.0,
                "num_articles": 0,
            }

        now = datetime.now()
        weighted_scores = []
        weights = []

        for article in articles:
            # Get sentiment
            text = article.get("combined_text", article.get("title", ""))
            result = self.analyzer.analyze_text(text)
            score = result["score"]

            # Calculate age in days
            pub_date = article.get("published", "")
            try:
                if pub_date:
                    # Try common date formats
                    for fmt in ["%Y-%m-%dT%H:%M:%SZ", "%a, %d %b %Y %H:%M:%S %Z",
                                "%Y-%m-%d"]:
                        try:
                            pub_dt = datetime.strptime(pub_date[:25].strip(), fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        pub_dt = now
                    age_days = (now - pub_dt).days
                else:
                    age_days = 1
            except Exception:
                age_days = 1

            weight = self.decay_weights.get(min(age_days, 3), 0.3)
            weighted_scores.append(score * weight)
            weights.append(weight)

        if not weights:
            return {"weighted_score": 0.0, "sentiment": "Neutral", "confidence": 0.0}

        avg_weighted = sum(weighted_scores) / sum(weights)

        if avg_weighted > 0.1:
            sentiment = "Positive"
        elif avg_weighted < -0.1:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"

        return {
            "weighted_score": round(avg_weighted, 4),
            "sentiment": sentiment,
            "confidence": round(abs(avg_weighted), 4),
            "num_articles": len(articles),
        }