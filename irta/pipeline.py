"""WF1~WF4 공통 실행 경로: 이슈 검색어로 기사 수집 → A.X 판정 → 등급 → 같은 이슈 비교 기사 → LOG-01 + 묶음 파일.

화면(app.py)·명령줄(run_news.py)·판정 서버(serve_judge.py)가 모두 이 함수를 쓴다.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict
from datetime import datetime
from typing import Callable

from .data import Article
from .grade import grade
from .logs import KST, write_log
from .news import fetch_article, fetch_feed

# 검색어에서 뺄 말(요청 문장의 군더더기)
STOP = {"이슈", "정리", "정리해줘", "정리해", "해줘", "알려줘", "관련", "뉴스", "기사", "이번", "이번주", "주", "최근", "요즘",
        "대해", "대한", "관해", "어떻게", "뭐", "무엇", "내용", "소식", "화제"}
JOSA = re.compile(r"(은|는|이|가|을|를|의|에|에서|와|과|도|로|으로)$")
RELATED_K = 2
RELATED_MIN_SIM = 0.12   # 같은 이슈로 볼 제목+첫 문장 3-gram 유사도 하한(초기 가정값)


def new_request_id() -> str:
    return "REQ-" + datetime.now(KST).strftime("%Y%m%d-%H%M%S")


def query_terms(query: str) -> list[str]:
    """'전기차 배터리 화재 이슈 정리해줘' → ['전기차', '배터리', '화재']"""
    terms = []
    for tok in re.findall(r"[가-힣A-Za-z0-9]+", query or ""):
        if len(tok) > 2:
            tok = JOSA.sub("", tok)
        if len(tok) >= 2 and tok not in STOP and tok not in terms:
            terms.append(tok)
    return terms


def match_score(entry: dict, terms: list[str]) -> int:
    """제목에 있으면 2점, 요약에만 있으면 1점. 검색어의 60% 이상이 나와야 후보(아니면 0)."""
    title, summ = entry["title"], entry.get("summary", "")
    hits = [t for t in terms if t in title or t in summ]
    if not terms or len(hits) < max(1, round(len(terms) * 0.6)):
        return 0
    return sum(2 if t in title else 1 for t in hits)


def public_row(row: dict) -> dict:
    """공개(Cloud) 전달용: 화면에 쓰는 근거·무관 의심 문장만 남기고 나머지 본문은 비운다. 모델 원출력은 뺀다."""
    j = {k: v for k, v in row["judgment"].items() if k != "raw"}
    keep = set(j["evidence"]) | set(j["irrelevant"])
    a = dict(row["article"])
    a["sentences"] = [s if i in keep else "" for i, s in enumerate(a["sentences"], 1)]
    return {**row, "article": a, "judgment": j}


def collect(feeds: list[str], top_n: int, query: str = "") -> list[Article]:
    """검색어가 있으면 그 이슈 기사(일치 점수 → 최신순), 없으면 최신순 상위 top_n건.
    본문을 못 가져오는 기사는 건너뛰고 다음 후보로 채운다."""
    entries = [e for name in feeds for e in fetch_feed(name)]
    terms = query_terms(query)
    if terms:
        scored = [(match_score(e, terms), e) for e in entries]
        entries = [e for s, e in sorted(scored, key=lambda x: (x[0], x[1]["ts"]), reverse=True) if s > 0]
    else:
        entries.sort(key=lambda e: e["ts"], reverse=True)
    arts: list[Article] = []
    seen: set[str] = set()
    for e in entries:
        if len(arts) >= top_n:
            break
        key = re.sub(r"\W+", "", e["title"])            # 같은 기사가 여러 분야 피드에 실린 경우
        if e["link"] in seen or key in seen:
            continue
        seen |= {e["link"], key}
        a = fetch_article(e)
        if a:
            arts.append(a)
    return arts


def _grams(text: str) -> set[str]:
    t = re.sub(r"\s+", "", text)
    return {t[i:i + 3] for i in range(max(len(t) - 2, 1))}


def attach_related(rows: list[dict]) -> None:
    """같은 묶음 안에서 제목+첫 문장이 가장 비슷한 다른 실제 기사 최대 2건을 비교 자료로 붙인다.
    크리에이터가 '다른 기사는 이렇게 썼다'를 보는 참고 정보이며 등급 계산에는 쓰지 않는다."""
    sig = [_grams(r["article"]["title"] + " " + next((s for s in r["article"]["sentences"] if s), "")) for r in rows]
    for i, r in enumerate(rows):
        cand = []
        for k, o in enumerate(rows):
            if k == i or o["article"].get("source") != "rss":
                continue
            sim = len(sig[i] & sig[k]) / (len(sig[i] | sig[k]) or 1)
            if sim >= RELATED_MIN_SIM:
                cand.append((sim, o))
        cand.sort(key=lambda x: x[0], reverse=True)
        r["related"] = [{"id": o["article"]["id"], "title": o["article"]["title"], "source": o["article"]["category"],
                         "url": o["article"].get("url", ""), "grade": o["grade"],
                         "lead": next((s for s in o["article"]["sentences"] if s), ""), "sim": round(sim, 2)}
                        for sim, o in cand[:RELATED_K]]


def judge_and_save(model, cfg: dict, items: list[tuple[Article, str, bool | None]], request_id: str,
                   safe_mode: bool = False, query: str = "",
                   progress: Callable[[int, int], None] | None = None) -> list[dict]:
    """items: (기사, 종류 rss|aihub-demo|check, 정답 또는 None). 묶음 파일은 runs/bundles/<요청 ID>.jsonl."""
    from .model import judge_article

    theta = cfg["rule"]["theta"]
    rows = []
    for i, (a, kind, answer) in enumerate(items, 1):
        j = judge_article(model, a, run_detect=True)
        g = grade(j.label, j.c, len(j.irrelevant), j.truncated, j.error, safe_mode, theta)
        write_log(cfg, "LOG-01", {
            "request_id": request_id, "article_id": a.id, "category": a.category, "check_article": kind == "check",
            "label": j.label, "c": j.c, "label_mass": j.label_mass, "irrelevant": list(j.irrelevant),
            "truncated": j.truncated, "error": j.error, "grade": g.grade, "reason": g.reason,
            "theta": theta, "state": "안전 모드" if safe_mode else "정상", "seconds": j.seconds})
        # answer(정답)는 점검용·사례 기사에만 있고, 화면은 크리에이터가 답한 뒤에만 보여 준다
        rows.append({"article": asdict(a), "judgment": asdict(j), "grade": g.grade, "reason": g.reason,
                     "row": g.row, "kind": kind, "answer": answer, "query": query})
        if progress:
            progress(i, len(items))
    attach_related(rows)
    out = cfg["paths"]["runs"] / "bundles" / f"{request_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return rows
