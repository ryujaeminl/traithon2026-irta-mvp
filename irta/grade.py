"""MDL-04 신뢰 등급 산정기 — RULE-v0.1 결정표 (BK05 §4.3).

위에서부터 처음 맞는 줄을 적용한다.
 ① 안전 모드: 높음을 내지 않는다. 낚시·c≥θ는 낮음, 나머지는 중간
 ② MDL-01·02 오류·시간 초과·형식 위반 → 중간
 ③ c < θ → 중간
 ④ 낚시, c ≥ θ → 낮음
 ⑤ 정상, c ≥ θ, 무관 문장 1개 이상 → 중간(판단 엇갈림)
 ⑥ 정상, c ≥ θ, 본문 절단 → 중간
 ⑦ 그 밖 → 높음
"""
from __future__ import annotations

from dataclasses import dataclass

HIGH, MID, LOW = "높음", "중간", "낮음"

# 사유 코드 → 크리에이터에게 보여 줄 쉬운 말 (BK05 §4.3, 문구 확정은 BK03 TXT)
REASON_TEXT = {
    "낚시 확신": "제목이 본문 내용과 다를 가능성이 높음",
    "확신 부족": "AI가 확신하지 못함",
    "판단 엇갈림": "본문에 주제와 무관한 문장이 있음",
    "본문 절단": "기사가 길어 일부만 확인함",
    "처리 오류": "자동 판단 실패",
    "안전 모드": "안전 모드 중이라 모든 기사를 직접 확인해야 함",
}


@dataclass(frozen=True)
class Grade:
    grade: str
    reason: str | None   # 높음이면 None
    row: int             # 적용된 결정표 줄 ①~⑦ (TEST-48 추적용)


def grade(label: str | None, c: float | None, n_irrelevant: int, truncated: bool,
          error: str | None, safe_mode: bool, theta: float) -> Grade:
    confident = c is not None and c >= theta
    if safe_mode:
        if error is None and label == "낚시" and confident:
            return Grade(LOW, "낚시 확신", 1)
        return Grade(MID, "안전 모드", 1)
    if error is not None or label not in ("낚시", "정상") or c is None:
        return Grade(MID, "처리 오류", 2)
    if not confident:
        return Grade(MID, "확신 부족", 3)
    if label == "낚시":
        return Grade(LOW, "낚시 확신", 4)
    if n_irrelevant >= 1:
        return Grade(MID, "판단 엇갈림", 5)
    if truncated:
        return Grade(MID, "본문 절단", 6)
    return Grade(HIGH, None, 7)


def certainty_word(c: float | None, theta: float, low: float) -> str:
    """확실성 3단계 (TLD-01-c). 확신도 원값은 크리에이터 화면에 내지 않는다."""
    if c is None:
        return "불확실"
    if c >= theta:
        return "확실함"
    return "다소 불확실" if c >= low else "불확실"
