"""
News Data Fetcher - Multiple sources with fallbacks
"""

import requests
import feedparser
import urllib.parse
from datetime import datetime, timedelta
from typing import List, Dict
from src.utils.logger import logger
import os
from dotenv import load_dotenv

load_dotenv()


class NewsDataFetcher:
    """Fetches financial news for Indian stocks with multiple fallbacks."""

    def __init__(self):
        self.newsapi_key = os.getenv("NEWS_API_KEY", "")
        if self.newsapi_key in ["", "your_newsapi_key_here", "YOUR_KEY_HERE"]:
            self.newsapi_key = ""
        logger.info("NewsDataFetcher initialized")

    def fetch_news(
        self,
        query: str,
        source: str = "auto",
        max_articles: int = 10,
        days_back: int = 7,
    ) -> List[Dict]:
        """
        Fetch news with automatic fallback.
        Tries multiple sources until one works.
        """
        logger.info(f"Fetching news for '{query}' from {source}")

        all_articles = []

        # Try multiple sources in order
        fetchers = [
            ("Google RSS", self._fetch_google_rss),
            ("Google RSS Alt", self._fetch_google_rss_alt),
            ("DuckDuckGo", self._fetch_duckduckgo_news),
            ("MoneyControl RSS", self._fetch_moneycontrol_rss),
            ("ET RSS", self._fetch_et_rss),
            ("RSS General", self._fetch_general_finance_rss),
        ]

        if self.newsapi_key:
            fetchers.insert(0, ("NewsAPI", self._fetch_newsapi))

        for name, fetcher in fetchers:
            try:
                articles = fetcher(query, max_articles, days_back)
                if articles:
                    logger.info(f"[OK] {name}: Got {len(articles)} articles for '{query}'")
                    all_articles.extend(articles)
                    if len(all_articles) >= max_articles:
                        break
                else:
                    logger.info(f"[SKIP] {name}: No articles found")
            except Exception as e:
                logger.warning(f"[FAIL] {name} failed: {e}")
                continue

        # Deduplicate by title
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            title_key = article.get("title", "").lower().strip()[:50]
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_articles.append(article)

        result = unique_articles[:max_articles]
        logger.info(f"Total unique articles: {len(result)}")
        return result

    def _fetch_google_rss(self, query: str, max_articles: int, days_back: int) -> List[Dict]:
        """Google News RSS - Method 1"""
        encoded = urllib.parse.quote(f"{query} stock")
        url = f"https://news.google.com/rss/search?q={encoded}&hl=en-IN&gl=IN&ceid=IN:en"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        feed = feedparser.parse(url, request_headers=headers)

        articles = []
        for entry in feed.entries[:max_articles]:
            articles.append({
                "title": entry.get("title", ""),
                "description": entry.get("summary", entry.get("title", "")),
                "source": entry.get("source", {}).get("title", "Google News"),
                "published": entry.get("published", ""),
                "url": entry.get("link", ""),
            })
        return articles

    def _fetch_google_rss_alt(self, query: str, max_articles: int, days_back: int) -> List[Dict]:
        """Google News RSS - Method 2 (different URL format)"""
        search_terms = [
            f"{query} share price",
            f"{query} stock market India",
            f"{query} NSE BSE",
        ]

        articles = []
        for term in search_terms:
            if len(articles) >= max_articles:
                break

            encoded = urllib.parse.quote(term)
            url = f"https://news.google.com/rss/search?q={encoded}+when:7d&hl=en-IN&gl=IN&ceid=IN:en"

            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:
                    articles.append({
                        "title": entry.get("title", ""),
                        "description": entry.get("summary", entry.get("title", "")),
                        "source": entry.get("source", {}).get("title", "Google News"),
                        "published": entry.get("published", ""),
                        "url": entry.get("link", ""),
                    })
            except Exception:
                continue

        return articles[:max_articles]

    def _fetch_duckduckgo_news(self, query: str, max_articles: int, days_back: int) -> List[Dict]:
        """DuckDuckGo News RSS"""
        encoded = urllib.parse.quote(f"{query} stock India")
        url = f"https://duckduckgo.com/?q={encoded}&t=h_&iar=news&ia=news&atb=v344-1"

        # DuckDuckGo instant answer API
        api_url = f"https://api.duckduckgo.com/?q={encoded}+stock+news&format=json&no_html=1"

        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            response = requests.get(api_url, headers=headers, timeout=10)
            data = response.json()

            articles = []
            for topic in data.get("RelatedTopics", [])[:max_articles]:
                text = topic.get("Text", "")
                if text:
                    articles.append({
                        "title": text[:150],
                        "description": text,
                        "source": "DuckDuckGo",
                        "published": datetime.now().isoformat(),
                        "url": topic.get("FirstURL", ""),
                    })
            return articles
        except Exception:
            return []

    def _fetch_moneycontrol_rss(self, query: str, max_articles: int, days_back: int) -> List[Dict]:
        """MoneyControl RSS feeds"""
        rss_urls = [
            "https://www.moneycontrol.com/rss/latestnews.xml",
            "https://www.moneycontrol.com/rss/marketreports.xml",
            "https://www.moneycontrol.com/rss/stocksnews.xml",
            "https://www.moneycontrol.com/rss/business.xml",
        ]

        query_words = query.lower().split()
        articles = []

        for rss_url in rss_urls:
            if len(articles) >= max_articles:
                break
            try:
                feed = feedparser.parse(rss_url)
                for entry in feed.entries:
                    title = entry.get("title", "").lower()
                    summary = entry.get("summary", "").lower()

                    # Match if ANY query word appears
                    if any(word in title or word in summary for word in query_words):
                        articles.append({
                            "title": entry.get("title", ""),
                            "description": entry.get("summary", ""),
                            "source": "MoneyControl",
                            "published": entry.get("published", ""),
                            "url": entry.get("link", ""),
                        })
            except Exception:
                continue

        return articles[:max_articles]

    def _fetch_et_rss(self, query: str, max_articles: int, days_back: int) -> List[Dict]:
        """Economic Times RSS"""
        rss_urls = [
            "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
            "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
            "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
        ]

        query_words = query.lower().split()
        articles = []

        for rss_url in rss_urls:
            if len(articles) >= max_articles:
                break
            try:
                feed = feedparser.parse(rss_url)
                for entry in feed.entries:
                    title = entry.get("title", "").lower()
                    summary = entry.get("summary", "").lower()

                    if any(word in title or word in summary for word in query_words):
                        articles.append({
                            "title": entry.get("title", ""),
                            "description": entry.get("summary", ""),
                            "source": "Economic Times",
                            "published": entry.get("published", ""),
                            "url": entry.get("link", ""),
                        })
            except Exception:
                continue

        return articles[:max_articles]

    def _fetch_general_finance_rss(self, query: str, max_articles: int, days_back: int) -> List[Dict]:
        """
        General financial news RSS as last fallback.
        Even if no stock-specific news, returns market news.
        """
        rss_urls = [
            "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
            "https://www.livemint.com/rss/markets",
            "https://www.business-standard.com/rss/markets-106.rss",
        ]

        articles = []
        for rss_url in rss_urls:
            if len(articles) >= max_articles:
                break
            try:
                feed = feedparser.parse(rss_url)
                for entry in feed.entries[:5]:
                    articles.append({
                        "title": entry.get("title", ""),
                        "description": entry.get("summary", entry.get("title", "")),
                        "source": "Market News",
                        "published": entry.get("published", ""),
                        "url": entry.get("link", ""),
                    })
            except Exception:
                continue

        return articles[:max_articles]

    def _fetch_newsapi(self, query: str, max_articles: int, days_back: int) -> List[Dict]:
        """NewsAPI.org (requires free API key)"""
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f"{query} stock",
            "from": from_date,
            "sortBy": "relevancy",
            "language": "en",
            "pageSize": max_articles,
            "apiKey": self.newsapi_key,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        articles = []
        for article in data.get("articles", []):
            articles.append({
                "title": article.get("title", ""),
                "description": article.get("description", ""),
                "source": article.get("source", {}).get("name", "NewsAPI"),
                "published": article.get("publishedAt", ""),
                "url": article.get("url", ""),
            })
        return articles