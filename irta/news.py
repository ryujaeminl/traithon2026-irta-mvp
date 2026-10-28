"""WF2 기사 수집 — 언론사 공개 RSS + 기사 본문 추출.

실제 기사에는 정답 라벨이 없으므로 성능 측정(TEST-41~49)에는 쓰지 않고 시연·운영 시뮬레이션에만 쓴다.
본문은 노트북에만 저장하고 화면에는 근거 문장과 원문 링크만 보여 준다(재배포 안 함).
robots.txt를 확인하고, 요청 사이에 간격을 둔다.
"""
from __future__ import annotations

import calendar
import hashlib
import re
import time
import urllib.request
import urllib.robotparser
from functools import lru_cache
from html.parser import HTMLParser
from urllib.parse import urlparse

from .data import Article, norm_title

# 2026-10-28 점검: 매일경제·한겨레는 robots.txt 금지, 조선일보는 본문이 스크립트로 그려져 추출 불가,
# SBS는 피드 이용 조건이 '개인 리더 이용'이라 뺐다.
FEEDS = {
    "연합뉴스": "https://www.yna.co.kr/rss/news.xml",
    "연합뉴스 연예": "https://www.yna.co.kr/rss/entertainment.xml",
    "경향신문": "https://www.khan.co.kr/rss/rssdata/total_news.xml",
    "동아일보": "https://rss.donga.com/total.xml",
}
UA = "Mozilla/5.0 (TRAITHON2026 research prototype; mixnuts)"
MIN_SENT = 3
REQUEST_GAP_SEC = 1.0
_SENT_SPLIT = re.compile(r"(?<=[.?!。])\s+(?=\S)")


@lru_cache(maxsize=64)
def _robots(netloc: str) -> urllib.robotparser.RobotFileParser:
    rp = urllib.robotparser.RobotFileParser(f"https://{netloc}/robots.txt")
    try:
        rp.read()
    except Exception:  # robots.txt를 못 읽으면 막힌 것으로 본다
        rp.disallow_all = True
    return rp


def allowed(url: str) -> bool:
    return _robots(urlparse(url).netloc).can_fetch(UA, url)


def _get(url: str, timeout: float = 10.0) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


class _BlockText(HTMLParser):
    """블록 요소별로 그 안의 글자(<br>로 나뉜 조각 포함)를 모은다. <p> 문단은 부모 블록에도 더한다.
    가장 긴 블록을 본문으로 본다."""
    SKIP = {"script", "style", "noscript", "header", "footer", "nav", "aside", "figure", "figcaption", "form",
            "button", "iframe", "svg"}
    BLOCK = {"div", "article", "section", "p", "td", "main"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[int, str]] = []
        self.skip = 0
        self.blocks: dict[int, list[str]] = {}
        self.next_id = 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        if tag in self.BLOCK:
            self.next_id += 1
            self.stack.append((self.next_id, tag))
        elif tag == "br" and self.stack:
            self.blocks.setdefault(self.stack[-1][0], []).append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        if tag in self.BLOCK and self.stack:
            node, t = self.stack.pop()
            if t == "p" and self.stack:  # 끝난 문단을 부모 블록 본문에 한 줄로 더한다
                self.blocks.setdefault(self.stack[-1][0], []).extend(self.blocks.get(node, []) + ["\n"])

    def handle_data(self, data):
        if self.skip or not self.stack or not data.strip():
            return
        self.blocks.setdefault(self.stack[-1][0], []).append(data)


def extract_sentences(html: str) -> list[str]:
    p = _BlockText()
    p.feed(html)
    best = max(p.blocks.values(), key=lambda parts: len("".join(parts)), default=[])
    text = re.sub(r"[ \t ]+", " ", "".join(best))
    sents: list[str] = []
    for para in text.split("\n"):
        for s in _SENT_SPLIT.split(para.strip()):
            s = s.strip()
            # 기자 서명·저작권·사진 설명 같은 짧은 조각은 뺀다
            if len(s) >= 15 and "무단" not in s and "저작권" not in s and "@" not in s:
                sents.append(s)
    return sents


def fetch_feed(name: str, limit: int = 20) -> list[dict]:
    import feedparser

    d = feedparser.parse(FEEDS[name], agent=UA)
    out = []
    for e in d.entries[:limit]:
        content = (e.get("content") or [{}])[0].get("value", "")
        t = e.get("published_parsed") or e.get("updated_parsed")
        out.append({"source": name, "title": e.get("title", ""), "link": e.get("link", ""),
                    "published": e.get("published", ""), "ts": calendar.timegm(t) if t else 0,
                    "summary": e.get("summary", ""), "content": content})
    return out


def fetch_article(entry: dict) -> Article | None:
    """RSS 항목 → Article. 본문 문장이 MIN_SENT개 미만이면 None(판정 불가)."""
    sents: list[str] = []
    if entry.get("content"):
        sents = extract_sentences(f"<div>{entry['content']}</div>")
    if len(sents) < MIN_SENT and entry.get("link") and allowed(entry["link"]):
        time.sleep(REQUEST_GAP_SEC)
        try:
            sents = extract_sentences(_get(entry["link"]))
        except Exception:
            sents = []
    if len(sents) < MIN_SENT:
        return None
    aid = "rss:" + hashlib.sha1(entry["link"].encode()).hexdigest()[:12]
    return Article(id=aid, source="rss", category=entry["source"], title=norm_title(entry["title"]),
                   subtitle="", sentences=tuple(sents), url=entry["link"], published=entry.get("published", ""))
