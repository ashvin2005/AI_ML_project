import re
import time
import logging

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

try:
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
CHUNK_WORDS = 80
MAX_PAGES = 5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "this", "that", "these", "those", "it", "its",
    "said", "says", "would", "could", "should", "will", "also", "just",
    "more", "than", "about", "after", "before", "when", "where", "which",
    "who", "they", "them", "their", "there", "then", "into", "over",
    "each", "your", "some", "time", "year", "what", "news", "not", "can",
    "his", "her", "him", "she", "he", "we", "us", "our", "you", "do",
}


class FactCheckRetriever:
    def __init__(self):
        self._embedder = None

    def _load_embedder(self):
        if self._embedder is None and RAG_AVAILABLE:
            self._embedder = SentenceTransformer(EMBED_MODEL)
        return self._embedder

    def extract_keywords(self, text: str, n: int = 8) -> str:
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        freq = {}
        for w in words:
            if w not in STOPWORDS:
                freq[w] = freq.get(w, 0) + 1
        top = sorted(freq, key=freq.get, reverse=True)[:n]
        return " ".join(top)

    def _chunk_text(self, text: str, source_url: str) -> list:
        words = text.split()
        chunks = []
        for i in range(0, len(words), CHUNK_WORDS):
            chunk = " ".join(words[i: i + CHUNK_WORDS])
            if len(chunk.strip()) > 40:
                chunks.append({"text": chunk, "source": source_url})
        return chunks

    def _scrape(self, url: str) -> str:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=8)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
                tag.decompose()
            text = " ".join(p.get_text(separator=" ", strip=True) for p in soup.find_all("p"))
            return re.sub(r"\s+", " ", text).strip()[:8000]
        except Exception as e:
            logger.debug(f"Could not scrape {url}: {e}")
            return ""

    def _fact_check_urls(self, keywords: str) -> list:
        enc = requests.utils.quote(keywords)
        return [
            f"https://www.snopes.com/?s={enc}",
            f"https://www.factcheck.org/?s={enc}",
            f"https://www.politifact.com/search/?q={enc}",
            f"https://fullfact.org/search/?q={enc}",
            f"https://apnews.com/search?query={enc}",
        ]

    def _rag_retrieve(self, article_text: str, keywords: str) -> list:
        embedder = self._load_embedder()
        if embedder is None:
            return []

        chunks = []
        for url in self._fact_check_urls(keywords)[:MAX_PAGES]:
            raw = self._scrape(url)
            if raw:
                chunks.extend(self._chunk_text(raw, url))

        if not chunks:
            return []

        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode(
            texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
        ).astype("float32")

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        q_emb = embedder.encode([article_text[:512]], normalize_embeddings=True).astype("float32")
        k = min(TOP_K, len(chunks))
        scores, idxs = index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            c = chunks[idx]
            domain = re.search(r"https?://(?:www\.)?([^/]+)", c["source"])
            results.append({
                "title": domain.group(1) if domain else c["source"][:40],
                "snippet": c["text"][:300],
                "url": c["source"],
                "score": float(score),
            })
        return results

    def _duckduckgo_search(self, keywords: str, max_results: int = 5) -> list:
        results = []
        try:
            sites = "site:snopes.com OR site:factcheck.org OR site:reuters.com OR site:apnews.com OR site:politifact.com"
            enc = requests.utils.quote(f"{keywords} fact check {sites}")
            resp = requests.get(
                f"https://html.duckduckgo.com/html/?q={enc}",
                headers=HEADERS, timeout=10,
            )
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            items = soup.select(".result__body") or soup.select(".results_links") or soup.select(".web-result")
            for item in items[:max_results]:
                title_el = item.select_one(".result__title") or item.select_one(".result__a")
                snip_el = item.select_one(".result__snippet") or item.select_one(".result__extras")
                url_el = item.select_one(".result__url") or item.select_one(".result__extras__url")

                title = title_el.get_text(strip=True) if title_el else ""
                snippet = snip_el.get_text(strip=True) if snip_el else "No preview available."
                href = url_el.get_text(strip=True) if url_el else "N/A"

                if title:
                    results.append({"title": title, "snippet": snippet, "url": href})
        except Exception as e:
            logger.warning(f"DuckDuckGo search failed: {e}")

        return results if results else self._static_fallback(keywords)

    def _static_fallback(self, keywords: str) -> list:
        kw = requests.utils.quote(keywords)
        return [
            {"title": "Snopes", "snippet": "One of the oldest fact-checking sites.", "url": f"https://www.snopes.com/?s={kw}"},
            {"title": "PolitiFact", "snippet": "Rates the accuracy of political claims.", "url": f"https://www.politifact.com/search/?q={kw}"},
            {"title": "Reuters Fact Check", "snippet": "Reuters investigates viral misinformation.", "url": "https://www.reuters.com/fact-check/"},
            {"title": "AP Fact Check", "snippet": "Associated Press fact-checking team.", "url": "https://apnews.com/hub/ap-fact-check"},
            {"title": "FactCheck.org", "snippet": "Nonpartisan fact-checking of political claims.", "url": f"https://www.factcheck.org/?s={kw}"},
        ]

    def retrieve(self, text: str) -> dict:
        keywords = self.extract_keywords(text)

        if RAG_AVAILABLE:
            sources = self._rag_retrieve(text, keywords)
            if sources:
                return {"keywords_used": keywords, "sources": sources, "method": "rag"}

        sources = self._duckduckgo_search(keywords)
        return {"keywords_used": keywords, "sources": sources, "method": "duckduckgo" if sources else "static"}