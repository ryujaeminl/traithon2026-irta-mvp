"""평가 지표 (BK05 TEST-41·42·45, MDL-02 공통 계획). 모델 없이 계산한다.

- 신뢰구간: Wilson 95% (표본이 작은 셀에서 정규근사보다 안전, BK04 DF-03과 같은 방식)
- ECE: 10구간 가중 평균 |정답률 − 평균 확신도|
- 온도 스케일링: z = ln p낚시 − ln p정상, p'(낚시) = σ(z/T). T 하나를 선택용 절반에서 NLL 최소로 맞춘다
- 낚시 비율 π 보정: P(낚시|높음) = π·r_높|낚 / (π·r_높|낚 + (1−π)·r_높|정), 중간 비율 = π·r_중|낚 + (1−π)·r_중|정
"""
from __future__ import annotations

import math

Z = 1.959964


def wilson(k: int, n: int) -> tuple[float, float, float]:
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + Z * Z / n
    mid = (p + Z * Z / (2 * n)) / den
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return p, mid - half, mid + half


def log_odds(pf: float, pn: float) -> float:
    eps = 1e-12
    return math.log(max(pf, eps)) - math.log(max(pn, eps))


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x)) if x >= 0 else math.exp(x) / (1 + math.exp(x))


def calibrated_c(z: float, t: float) -> float:
    """보정된 확신도 = 두 레이블 중 큰 쪽 확률."""
    return sigmoid(abs(z) / t)


def nll(zs: list[float], ys: list[int], t: float) -> float:
    eps = 1e-12
    return -sum(math.log(max(sigmoid(z / t) if y else 1 - sigmoid(z / t), eps)) for z, y in zip(zs, ys)) / len(zs)


def fit_temperature(zs: list[float], ys: list[int]) -> float:
    """T ∈ [0.25, 100] 로그 격자 탐색 후 근방 세분. 매개변수 하나라 격자로 충분하다."""
    grid = [10 ** (k / 40) for k in range(-24, 81)]
    best = min(grid, key=lambda t: nll(zs, ys, t))
    fine = [best * 10 ** (k / 400) for k in range(-10, 11)]
    return min(fine, key=lambda t: nll(zs, ys, t))


def ece(confs: list[float], correct: list[bool], bins: int = 10) -> float:
    n = len(confs)
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confs) if (lo < c <= hi) or (b == 0 and c == 0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            conf = sum(confs[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - conf)
    return total


def pi_correct(pi: float, r_hf: float, r_hn: float, r_mf: float, r_mn: float) -> tuple[float, float]:
    """(높음 중 낚시 비율, 중간 비율). 베이즈 정리 / 전확률 공식."""
    den = pi * r_hf + (1 - pi) * r_hn
    return (pi * r_hf / den if den else float("nan")), pi * r_mf + (1 - pi) * r_mn


def prf(pred: set[int], gold: set[int]) -> tuple[int, int, int]:
    """(맞힘, 예측 수, 정답 수) — 문장 단위로 합산해 P·R·F1을 낸다."""
    return len(pred & gold), len(pred), len(gold)


def f1(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)
