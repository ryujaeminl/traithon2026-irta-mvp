"""WF2~WF4 공통 실행 경로: 기사 수집(상위 N건) → A.X 판정 → 등급 → LOG-01 + 묶음 파일.

화면(app.py)·명령줄(run_news.py)·판정 서버(serve_judge.py)가 모두 이 함수를 쓴다.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime

from .data import Article
from .grade import grade
from .logs import KST, write_log
from .news import fetch_article, fetch_feed


def new_request_id() -> str:
    return "REQ-" + datetime.now(KST).strftime("%Y%m%d-%H%M%S")


def public_row(row: dict) -> dict:
    """공개(Cloud) 전달용: 화면에 쓰는 근거·무관 의심 문장만 남기고 나머지 본문은 비운다. 모델 원출력은 뺀다."""
    j = {k: v for k, v in row["judgment"].items() if k != "raw"}
    keep = set(j["evidence"]) | set(j["irrelevant"])
    a = dict(row["article"])
    a["sentences"] = [s if i in keep else "" for i, s in enumerate(a["sentences"], 1)]
    return {**row, "article": a, "judgment": j}


def collect(feeds: list[str], top_n: int) -> list[Article]:
    """고른 언론사 RSS를 합쳐 최신순 상위 top_n건. 본문을 못 가져오는 기사는 건너뛰고 다음 기사로 채운다."""
    entries = [e for name in feeds for e in fetch_feed(name, limit=max(top_n, 10))]
    entries.sort(key=lambda e: e["ts"], reverse=True)
    arts: list[Article] = []
    seen: set[str] = set()
    for e in entries:
        if len(arts) >= top_n:
            break
        if e["link"] in seen:
            continue
        seen.add(e["link"])
        a = fetch_article(e)
        if a:
            arts.append(a)
    return arts


def judge_and_save(model, cfg: dict, items: list[tuple[Article, str, bool | None]], request_id: str,
                   safe_mode: bool = False) -> list[dict]:
    """items: (기사, 종류 rss|aihub-demo|check, 정답 또는 None). 묶음 파일은 runs/bundles/<요청 ID>.jsonl."""
    from .model import judge_article

    theta = cfg["rule"]["theta"]
    out = cfg["paths"]["runs"] / "bundles" / f"{request_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(out, "w", encoding="utf-8") as fh:
        for a, kind, answer in items:
            j = judge_article(model, a, run_detect=True)
            g = grade(j.label, j.c, len(j.irrelevant), j.truncated, j.error, safe_mode, theta)
            write_log(cfg, "LOG-01", {
                "request_id": request_id, "article_id": a.id, "category": a.category, "check_article": kind == "check",
                "label": j.label, "c": j.c, "label_mass": j.label_mass, "irrelevant": list(j.irrelevant),
                "truncated": j.truncated, "error": j.error, "grade": g.grade, "reason": g.reason,
                "theta": theta, "state": "안전 모드" if safe_mode else "정상", "seconds": j.seconds})
            # answer(정답)는 점검용·사례 기사에만 있고, 화면은 크리에이터가 답한 뒤에만 보여 준다
            row = {"article": asdict(a), "judgment": asdict(j), "grade": g.grade, "reason": g.reason,
                   "row": g.row, "kind": kind, "answer": answer}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
    return rows
