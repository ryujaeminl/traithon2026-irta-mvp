"""판정 결과 → TEST-41·42·45와 MDL-02 지표 보고 (BK05). CPU에서 돈다.

θ와 온도 T는 선택용(theta_select) 절반에서만 정하고, 확인용(confirm) 절반은 한 번만 확인한다.
usage: python scripts/report_eval.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irta.data import load_cfg  # noqa: E402
from irta.grade import HIGH, MID, grade  # noqa: E402
from irta.metrics import calibrated_c, ece, f1, fit_temperature, log_odds, pi_correct, prf, wilson  # noqa: E402
from scripts.run_eval import out_dir  # noqa: E402

PIS = (0.10, 0.20, 0.50)
THETAS = [round(0.50 + 0.01 * k, 2) for k in range(50)]
MAX_HIGH_FAKE, MAX_MID = 0.05, 0.30


def pct(x: float) -> str:
    return "—" if x != x else f"{100 * x:.1f}%"


def ci(k: int, n: int) -> str:
    p, lo, hi = wilson(k, n)
    return f"{pct(p)} ({k}/{n}, 95% CI {pct(lo)}~{pct(hi)})" if n else "표본 없음"


def load(path: Path) -> list[dict]:
    rows = []
    for ln in path.open(encoding="utf-8"):
        r = json.loads(ln)
        j, lab = r["judgment"], r["label"]
        ok = j["error"] is None and j["p_f"] is not None
        rows.append({**r, "y": lab["clickbait"], "method": lab["method"] if lab["clickbait"] else "정상",
                     "pred": j["label"], "ok": ok, "z": log_odds(j["p_f"], j["p_n"]) if ok else None})
    return rows


def grade_row(r: dict, theta: float, t: float) -> str:
    j = r["judgment"]
    c = calibrated_c(r["z"], t) if r["ok"] else None
    return grade(j["label"], c, len(j["irrelevant"]), j["truncated"], j["error"], False, theta).grade


def rates(rows: list[dict], theta: float, t: float) -> dict:
    out = {}
    for grp in ("Auto", "Direct", "정상"):
        sub = [r for r in rows if r["method"] == grp]
        g = [grade_row(r, theta, t) for r in sub]
        n = len(g) or 1
        out[grp] = (g.count(HIGH) / n, g.count(MID) / n, len(sub))
    return out


def pi_table(rt: dict, pi: float) -> tuple[float, float]:
    """낚시 쪽은 Auto/Direct 중 나쁜 쪽(BK05 §4.3.3)."""
    hn, mn, _ = rt["정상"]
    worst = [pi_correct(pi, rt[m][0], hn, rt[m][1], mn) for m in ("Auto", "Direct")]
    return max(w[0] for w in worst), max(w[1] for w in worst)


def main() -> None:
    cfg = load_cfg()
    d = out_dir(cfg)
    rows = load(d / "judgments.jsonl")
    sel = [r for r in rows if r["role"] == "theta_select"]
    chk = [r for r in rows if r["role"] == "confirm"]
    L: list[str] = [f"# 판정 평가 보고 — {cfg['version']} / {cfg['model']['precision']}",
                    "",
                    f"> 노트북 예비 실행 여부: {'예(4bit, 최종 수치 아님)' if cfg['model']['precision'] == 'nf4' else '아니오'}"
                    f" · 표본 {len(rows)}건(선택용 {len(sel)} / 확인용 {len(chk)}) · 형식·모델 오류 "
                    f"{sum(not r['ok'] for r in rows)}건", ""]

    p1 = [r for r in rows if r["part"] == "Part1" and r["ok"]]
    L += ["## TEST-41 판정 성능 (Part1, MDL-01)", "",
          f"- 사람 작성 낚시 FNR: {ci(sum(r['pred'] != '낚시' for r in p1 if r['method'] == 'Direct'), sum(r['method'] == 'Direct' for r in p1))} — 기준 ≤ 30%",
          f"- 자동 생성 낚시 FNR: {ci(sum(r['pred'] != '낚시' for r in p1 if r['method'] == 'Auto'), sum(r['method'] == 'Auto' for r in p1))}",
          f"- 정상 기사 FPR: {ci(sum(r['pred'] == '낚시' for r in p1 if r['method'] == '정상'), sum(r['method'] == '정상' for r in p1))} — 기준 ≤ 10%",
          ""]
    by_pat = defaultdict(lambda: [0, 0])
    for r in p1:
        if r["method"] == "Direct":
            by_pat[r["processPattern"]][0] += r["pred"] != "낚시"
            by_pat[r["processPattern"]][1] += 1
    L += ["| 패턴 | 미탐/건수 |", "|---|---|"] + [f"| {p} | {k}/{n} |" for p, (k, n) in sorted(by_pat.items())] + [""]

    ok_sel = [r for r in sel if r["ok"]]
    t = fit_temperature([r["z"] for r in ok_sel], [int(r["y"]) for r in ok_sel]) if ok_sel else 1.0
    L += ["## TEST-42 보정과 θ 선택", "", f"- 온도 T = {t:.3f} (선택용 {len(ok_sel)}건에서 NLL 최소)"]
    for name, sub in (("확인용 전체", [r for r in chk if r["ok"]]),
                      ("확인용 자동 생성", [r for r in chk if r["ok"] and r["method"] in ("Auto", "정상")]),
                      ("확인용 사람 작성", [r for r in chk if r["ok"] and r["method"] in ("Direct", "정상")])):
        if sub:
            before = ece([calibrated_c(r["z"], 1.0) for r in sub], [(r["pred"] == "낚시") == r["y"] for r in sub])
            after = ece([calibrated_c(r["z"], t) for r in sub], [(r["pred"] == "낚시") == r["y"] for r in sub])
            L.append(f"- ECE {name}: 보정 전 {before:.3f} → 보정 후 {after:.3f} (기준 ≤ 0.10, {len(sub)}건)")
    feasible = []
    for th in THETAS:
        rt = rates(sel, th, t)
        h20, m20 = pi_table(rt, 0.20)
        if h20 <= MAX_HIGH_FAKE and m20 <= MAX_MID:
            feasible.append((m20, th))
    L.append("")
    if feasible:
        theta = min(feasible)[1]
        L.append(f"- 선택된 θ = {theta} (π=20%에서 높음 중 낚시 ≤ 5%·중간 ≤ 30%를 만족하는 θ 중 중간 비율 최소)")
    else:
        theta = cfg["rule"]["theta"]
        L.append(f"- **두 조건을 함께 만족하는 θ 없음** — DEC 기록 필요. 아래 표는 현재 θ = {theta} 기준")
    L += ["", "| 세트 | π | 높음 중 낚시 | 중간 비율 | 판정 |", "|---|---|---|---|---|"]
    for name, sub in (("선택용", sel), ("확인용", chk)):
        rt = rates(sub, theta, t)
        for pi in PIS:
            h, m = pi_table(rt, pi)
            L.append(f"| {name} | {int(pi * 100)}% | {pct(h)} | {pct(m)} | "
                     f"{'통과' if h <= MAX_HIGH_FAKE and m <= MAX_MID else '미달'} |")
    rt = rates(chk, theta, t)
    L += ["", "클래스별 등급 비율(확인용, θ 적용): " + " / ".join(
        f"{k} 높음 {pct(v[0])}·중간 {pct(v[1])} (n={v[2]})" for k, v in rt.items()), ""]

    L += ["## TEST-45 카테고리 격차 (Part1)", "", "| 카테고리 | Direct FNR | FPR |", "|---|---|---|"]
    gaps = {"fnr": [], "fpr": []}
    for cat in sorted({r["category"] for r in p1}):
        dd = [r for r in p1 if r["category"] == cat and r["method"] == "Direct"]
        nn = [r for r in p1 if r["category"] == cat and r["method"] == "정상"]
        fnr = sum(r["pred"] != "낚시" for r in dd) / len(dd) if dd else float("nan")
        fpr = sum(r["pred"] == "낚시" for r in nn) / len(nn) if nn else float("nan")
        gaps["fnr"].append(fnr)
        gaps["fpr"].append(fpr)
        L.append(f"| {cat} | {pct(fnr)} ({len(dd)}) | {pct(fpr)} ({len(nn)}) |")
    for k, v in gaps.items():
        v = [x for x in v if x == x]
        if v:
            L.append(f"- {k.upper()} 격차: {pct(max(v) - min(v))} (기준 ≤ 15%p)")
    L.append("")

    p2 = [r for r in rows if r["part"] == "Part2" and r["ok"]]
    L += ["## MDL-02 무관 문장 (Part2) — '마지막 문장 지목' 기준선과 비교", "",
          "| 그룹 | MDL-02 P / R / F1 | 기준선 P / R / F1 |", "|---|---|---|"]
    for grp in ("Auto", "Direct", "Direct+정상"):
        sub = [r for r in p2 if (r["method"] in ("Direct", "정상") if grp == "Direct+정상" else r["method"] == grp)]
        m = [0, 0, 0]
        b = [0, 0, 0]
        for r in sub:
            gold = set(r["label"]["irrelevant"])
            for acc, pred in ((m, set(r["judgment"]["irrelevant"])), (b, {r["label"]["n_sent"]})):
                for i, v in enumerate(prf(pred, gold)):
                    acc[i] += v
        fm, fb = f1(*m), f1(*b)
        L.append(f"| {grp} (n={len(sub)}) | {fm[0]:.2f} / {fm[1]:.2f} / {fm[2]:.2f} | "
                 f"{fb[0]:.2f} / {fb[1]:.2f} / {fb[2]:.2f} |")
    L += ["", "기준: Direct+정상에서 MDL-02 F1 > 기준선 F1 (BK05 §5.2, 초기 가정값)."]

    (d / "report.md").write_text("\n".join(L), encoding="utf-8")
    (d / "report.json").write_text(json.dumps({"temperature": t, "theta": theta, "n": len(rows)}, indent=1),
                                   encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
