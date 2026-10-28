"""운영 로그 LOG-01~06 (BK07 §4.1.5, DATA-05). JSONL에 덧붙이기만 한다.

LOG-01 판정·등급 / LOG-02 크리에이터 결정 / LOG-03 운영자 점검 / LOG-04 경보·상태 전환 / LOG-06 구성 승인.
모든 줄에 시각(KST)과 CFG 버전을 붙인다. 자유 기재란은 저장 전에 전화번호·이메일을 가린다.
실명·연락처·요청 원문은 저장하지 않는다(보존 90일, 초기 가정값).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))
_PHONE = re.compile(r"\d{2,3}[-. ]?\d{3,4}[-. ]?\d{4}")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")


def mask(text: str) -> str:
    return _PHONE.sub("[전화번호]", _EMAIL.sub("[이메일]", text or ""))


def now() -> str:
    return datetime.now(KST).isoformat(timespec="seconds")


def write_log(cfg: dict, name: str, record: dict) -> dict:
    """name: 'LOG-01' 등. 기록이 실패하면 예외를 그대로 올린다 — 기록 없는 판정은 요약에 쓰지 않는다(BK07)."""
    row = {"ts": now(), "cfg": cfg["version"], **record}
    path = cfg["paths"]["runs"] / "logs" / f"{name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row


def read_log(cfg: dict, name: str) -> list[dict]:
    path = cfg["paths"]["runs"] / "logs" / f"{name}.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(ln) for ln in fh if ln.strip()]
