"""EVAL-AUTO-v0.1 판정 배치 (BK05 TEST-41·42·45 입력). GPU에서 돌리고, 끊겨도 이어서 돈다.

Part1은 MDL-01만, Part2는 MDL-01+02 (BK05 §4.3.2 θ 선택 데이터).
--per-cell N: (Part × 카테고리 × 셀 × 선택용/확인용)마다 앞 N건만 — 노트북 예비 실행용 층화 표본.
usage: python scripts/run_eval.py --per-cell 5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irta.data import load_article, load_cfg, load_manifest  # noqa: E402
from irta.model import AXModel, judge_article  # noqa: E402


def out_dir(cfg: dict) -> Path:
    return cfg["paths"]["runs"] / "eval" / f"{cfg['version']}-{cfg['model']['precision']}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="EVAL-AUTO-v0.1")
    ap.add_argument("--per-cell", type=int, default=0, help="0이면 전체")
    args = ap.parse_args()

    cfg = load_cfg()
    rows = load_manifest(cfg, args.eval_set)
    if args.per_cell:
        seen: dict = {}
        picked = []
        for r in sorted(rows, key=lambda r: int(r["rank_in_cell"])):
            k = (r["part"], r["category"], r["cell"], r["role"])
            if seen.get(k, 0) < args.per_cell:
                seen[k] = seen.get(k, 0) + 1
                picked.append(r)
        rows = picked
    path = out_dir(cfg) / "judgments.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if path.exists():
        done = {json.loads(ln)["newsID"] for ln in path.open(encoding="utf-8") if ln.strip()}
    todo = [r for r in rows if r["newsID"] not in done]
    print(f"대상 {len(rows)}건, 남은 {len(todo)}건 → {path}")
    if not todo:
        return

    model = AXModel(cfg)
    t0 = time.perf_counter()
    with path.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(todo, 1):
            art, lab = load_article(cfg, r)
            j = judge_article(model, art, run_detect=(r["part"] == "Part2"))
            keep = {k: r[k] for k in ("eval_set", "role", "part", "category", "cell", "newsID", "processPattern")}
            rec = {**keep, "label": {"clickbait": lab.clickbait, "method": lab.method,
                                     "irrelevant": list(lab.irrelevant), "n_sent": len(art.sentences)},
                   "judgment": {k: v for k, v in asdict(j).items() if k != "raw"}, "raw": j.raw}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if i % 20 == 0 or i == len(todo):
                el = time.perf_counter() - t0
                print(f"  {i}/{len(todo)} · 평균 {el / i:.1f}s/건 · 남은 약 {el / i * (len(todo) - i) / 60:.0f}분",
                      flush=True)


if __name__ == "__main__":
    main()
