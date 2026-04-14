"""
News Text Preprocessor - Clean and prepare news text for sentiment analysis
"""

import re
from typing import List, Dict
from src.utils.logger import logger


class NewsPreprocessor:
    """Cleans news text for sentiment analysis."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean a single text string."""
        if not text:
            return ""

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s.,!?'-]", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    @classmethod
    def clean_articles(cls, articles: List[Dict]) -> List[Dict]:
        """Clean all articles."""
        cleaned = []
        for article in articles:
            cleaned_article = {
                "title": cls.clean_text(article.get("title", "")),
                "description": cls.clean_text(article.get("description", "")),
                "source": article.get("source", "Unknown"),
                "published": article.get("published", ""),
                "url": article.get("url", ""),
                "combined_text": cls.clean_text(
                    f"{article.get('title', '')} {article.get('description', '')}"
                ),
            }
            if cleaned_article["combined_text"]:
                cleaned.append(cleaned_article)

        logger.info(f"Cleaned {len(cleaned)}/{len(articles)} articles")
        return cleaned