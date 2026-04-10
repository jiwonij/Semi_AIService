from datetime import datetime, timezone
from typing import Dict, List
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from ddgs import DDGS
from datetime import datetime
from dateutil import parser
from utils.config import (
    WEB_MIN_DOMAIN_COUNT,
    WEB_RECENCY_RATIO_THRESHOLD,
)


class WebSearchService:
    """
    실제 웹 검색을 수행한 뒤, 각 URL의 본문을 크롤링하여 구조화하는 서비스.
    """

    def __init__(self) -> None:
        self.max_results_per_query = 5
        self.request_timeout = 10
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }

    def run(self, search_queries: List[str]) -> Dict:
        print("\n[WebSearch] 크롤링 시작...")
        raw_documents = self._collect_documents(search_queries)
        print(f"[WebSearch] 크롤링 완료: {len(raw_documents)}개 문서 수집\n")
        metrics = self._evaluate_search_results(raw_documents)

        return {
            "raw_documents": raw_documents,
            "metrics": metrics,
        }

    def _collect_documents(self, search_queries: List[str]) -> List[Dict]:
        collected_documents: List[Dict] = []
        seen_urls = set()

        with DDGS() as ddgs:
            for query in search_queries:
                try:
                    results = ddgs.text(query, max_results=self.max_results_per_query)
                except Exception:
                    continue

                for item in results:
                    url = item.get("href") or item.get("url") or ""
                    if not url or url in seen_urls:
                        continue

                    seen_urls.add(url)

                    title = item.get("title", "")
                    snippet = item.get("body", "")
                    published_at = item.get("date") or ""
                    domain = self._extract_domain(url)

                    crawled_content = self._crawl_page_content(url)
                    final_content = crawled_content if crawled_content else snippet

                    if not final_content.strip():
                        continue

                    collected_documents.append(
                        {
                            "query": query,
                            "title": title,
                            "content": final_content,
                            "snippet": snippet,
                            "url": url,
                            "domain": domain,
                            "published_at": published_at,
                        }
                    )

        return collected_documents

    def _crawl_page_content(self, url: str) -> str:
        try:
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
        except Exception:
            return ""

        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type:
            return ""

        html = response.text
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        article = soup.find("article")
        main = soup.find("main")

        if article:
            text = article.get_text(separator=" ", strip=True)
        elif main:
            text = main.get_text(separator=" ", strip=True)
        else:
            body = soup.find("body")
            text = body.get_text(separator=" ", strip=True) if body else soup.get_text(separator=" ", strip=True)

        text = " ".join(text.split())

        if len(text) > 5000:
            text = text[:5000]

        return text

    def _evaluate_search_results(self, raw_documents: List[Dict]) -> Dict:
        if not raw_documents:
            return {
                "domain_diversity": 0,
                "recency_ratio": 0.0,
                "result_count": 0,
                "pass": False,
            }

        unique_domains = {doc["domain"] for doc in raw_documents if doc.get("domain")}
        domain_diversity = len(unique_domains)
        recency_ratio = self._compute_recency_ratio(raw_documents)

        return {
            "domain_diversity": domain_diversity,
            "recency_ratio": recency_ratio,
            "result_count": len(raw_documents),
            "pass": (
                domain_diversity >= WEB_MIN_DOMAIN_COUNT
                and (
                    recency_ratio >= WEB_RECENCY_RATIO_THRESHOLD
                    or recency_ratio == 0.0
                )
            ),
        }

    def _extract_domain(self, url: str) -> str:
        try:
            return urlparse(url).netloc.replace("www.", "")
        except Exception:
            return ""

    def _compute_recency_ratio(self, raw_documents: List[Dict]) -> float:
        now = datetime.now(timezone.utc)
        recent_count = 0
        dated_count = 0

        for doc in raw_documents:
            published_dt = self._parse_date(
                doc.get("published_at") 
                or doc.get("date")
                or doc.get("year")
            )
            if not published_dt:
                continue

            dated_count += 1
            if (now - published_dt).days <= 730:
                recent_count += 1

        if dated_count == 0:
            return 0.0

        return recent_count / dated_count

    def _parse_date(self, date_str):
        if not date_str:
            return None

        try:
            return parser.parse(date_str)
        except:
            return None

        return None
