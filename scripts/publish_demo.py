"""노트북에서 판정한 묶음을 Streamlit Cloud 시연용으로 공개 폴더(demo/bundles)에 옮긴다.

Cloud에는 GPU가 없어 A.X 판정을 못 하므로, 노트북(또는 대회 GPU)에서 run_news.py로 판정한 결과를 저장소에 올린다.
공개 범위를 줄인다:
- 기본은 실제 RSS 기사만. AI-Hub 기사(낚시 사례·점검용)는 --include-aihub를 줄 때만 넣는다(이용 범위 확인 후).
- 본문은 화면에 쓰는 근거·무관 의심 문장만 남기고 나머지 문장은 빈 문자열로 지운다(원문은 링크로 연결).
- 모델 원출력(raw)은 뺀다.
usage: python scripts/publish_demo.py [REQ-...] [--include-aihub]   (요청 ID를 빼면 가장 최근 묶음)
       git add demo/bundles && git commit -m "demo: 시연 묶음 갱신" && git push   → Cloud가 자동 재배포
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irta.data import ROOT, load_cfg  # noqa: E402


def minimize(row: dict) -> dict:
    j = {k: v for k, v in row["judgment"].items() if k != "raw"}
    keep = set(j["evidence"]) | set(j["irrelevant"])
    a = dict(row["article"])
    a["sentences"] = [s if i in keep else "" for i, s in enumerate(a["sentences"], 1)]
    return {**row, "article": a, "judgment": j}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("request_id", nargs="?")
    ap.add_argument("--include-aihub", action="store_true")
    args = ap.parse_args()

    src_dir = load_cfg()["paths"]["runs"] / "bundles"
    src = src_dir / f"{args.request_id}.jsonl" if args.request_id else max(src_dir.glob("*.jsonl"))
    rows = [json.loads(ln) for ln in src.open(encoding="utf-8") if ln.strip()]
    kept = [minimize(r) for r in rows if args.include_aihub or r["article"]["source"] == "rss"]
    out = ROOT / "demo" / "bundles" / src.name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    print(f"{src.name}: {len(rows)}건 중 {len(kept)}건 공개 → {out}")


if __name__ == "__main__":
    main()
