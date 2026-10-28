"""지표 계산을 손으로 계산한 값과 대조한다."""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irta.metrics import calibrated_c, ece, f1, fit_temperature, pi_correct, sigmoid, wilson  # noqa: E402


def test_wilson_known_value():
    p, lo, hi = wilson(3, 10)
    assert p == 0.3 and abs(lo - 0.1078) < 1e-3 and abs(hi - 0.6032) < 1e-3


def test_pi_correct_matches_hand_calculation():
    # 설명용 예: 낚시 10%·정상 80%가 높음 → π=20%면 3.0%, π=50%면 11.1%
    assert abs(pi_correct(0.2, 0.10, 0.80, 0, 0)[0] - 0.02 / 0.66) < 1e-9
    assert abs(pi_correct(0.5, 0.10, 0.80, 0, 0)[0] - 0.05 / 0.45) < 1e-9
    assert abs(pi_correct(0.2, 0, 0, 0.5, 0.1)[1] - (0.2 * 0.5 + 0.8 * 0.1)) < 1e-9


def test_ece_zero_when_calibrated_and_large_when_overconfident():
    assert ece([0.75] * 4, [True, True, True, False]) < 1e-9
    assert abs(ece([0.99] * 10, [True] * 5 + [False] * 5) - 0.49) < 1e-9


def test_temperature_recovers_known_scale():
    rng = random.Random(0)
    true_t, zs, ys = 4.0, [], []
    for _ in range(4000):
        z = rng.uniform(-20, 20)
        zs.append(z)
        ys.append(int(rng.random() < sigmoid(z / true_t)))
    assert abs(fit_temperature(zs, ys) - true_t) / true_t < 0.15
    assert abs(calibrated_c(-8.0, 4.0) - sigmoid(2.0)) < 1e-12


def test_f1():
    assert f1(2, 4, 2) == (0.5, 1.0, 2 * 0.5 / 1.5)
    assert f1(0, 0, 3) == (0.0, 0.0, 0.0)
    assert not math.isnan(f1(0, 2, 0)[2])
