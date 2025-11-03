"""Web search fallback retriever using Serper or Bing API.

Returns top 3 cleaned article texts when Wikipedia retrieval is insufficient.
"""

from __future__ import annotations

from typing import List, Dict

import os
import httpx
from bs4 import BeautifulSoup

try:
    from readability import Document  # type: ignore
    _HAS_READABILITY = True
except Exception:
    Document = None  # type: ignore
    _HAS_READABILITY = False


def _clean_html(html: str) -> str:
    if _HAS_READABILITY and Document is not None:
        try:
            doc = Document(html)
            summary_html = doc.summary(html_partial=True)
            soup = BeautifulSoup(summary_html, "html.parser")
            return soup.get_text(" ", strip=True)
        except Exception:
            pass
    # Fallback: simple text extraction
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return soup.get_text(" ", strip=True)


async def web_search_fallback(query: str) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    serper_key = os.getenv("SERPER_API_KEY")
    bing_key = os.getenv("BING_API_KEY")

    async with httpx.AsyncClient(timeout=15) as client:
        if serper_key:
            try:
                resp = await client.post(
                    "https://google.serper.dev/search",
                    headers={"X-API-KEY": serper_key, "Content-Type": "application/json"},
                    json={"q": query},
                )
                data = resp.json()
                for item in (data.get("organic", []) or [])[:3]:
                    url = item.get("link")
                    if not url:
                        continue
                    page = await client.get(url)
                    cleaned = _clean_html(page.text)
                    results.append({"url": url, "text": cleaned})
            except Exception:
                pass
        elif bing_key:
            try:
                resp = await client.get(
                    "https://api.bing.microsoft.com/v7.0/search",
                    headers={"Ocp-Apim-Subscription-Key": bing_key},
                    params={"q": query},
                )
                data = resp.json()
                for item in (data.get("webPages", {}).get("value", []) or [])[:3]:
                    url = item.get("url")
                    if not url:
                        continue
                    page = await client.get(url)
                    cleaned = _clean_html(page.text)
                    results.append({"url": url, "text": cleaned})
            except Exception:
                pass
    return results


__all__ = ["web_search_fallback"]


