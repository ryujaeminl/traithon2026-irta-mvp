"""WF1~WF4 실전 실행: 언론사 RSS 기사 수집 → A.X 판정(MDL-01·02) → 등급(MDL-04) → LOG-01 기록.

결과 묶음은 runs/bundles/<요청 ID>.jsonl에 저장하고 화면(app.py)이 읽는다.
실제 기사에는 정답이 없으므로 이 결과는 성능 수치가 아니다(시연·운영 시뮬레이션 전용).
--aihub-demo K: 사람이 낚시로 고쳐 쓴 AI-Hub 검증 기사 등 K건을 함께 넣는다(시연용, 평가 세트와 본문 겹침 없음).
--check-ratio: 확인 목록에 섞을 점검용 기사 비율(BK06 HO-01, 초기 가정 10%). 정답은 응답 후에만 보여 준다.
usage: python scripts/run_news.py --per-feed 3 --aihub-demo 4 [--feeds 연합뉴스 경향신문]
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import math  # noqa: E402

from irta.data import load_cfg, sample_validation  # noqa: E402
from irta.grade import certainty_word, grade  # noqa: E402
from irta.logs import KST, write_log  # noqa: E402
from irta.model import AXModel, judge_article  # noqa: E402
from irta.news import FEEDS, fetch_article, fetch_feed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-feed", type=int, default=3)
    ap.add_argument("--feeds", nargs="*", default=list(FEEDS))
    ap.add_argument("--aihub-demo", type=int, default=0)
    ap.add_argument("--check-ratio", type=float, default=0.10)
    args = ap.parse_args()

    cfg = load_cfg()
    theta, low = cfg["rule"]["theta"], cfg["rule"]["certainty_low"]
    request_id = "REQ-" + datetime.now(KST).strftime("%Y%m%d-%H%M%S")
    arts = []
    for name in args.feeds:
        for e in fetch_feed(name, limit=args.per_feed * 3):
            a = fetch_article(e)
            if a:
                arts.append(a)
            if sum(x.category == name for x in arts) >= args.per_feed:
                break
    items = [(a, "rss", None) for a in arts]
    seed = int(request_id[-6:])
    if args.aihub_demo:
        k = args.aihub_demo
        demo = (sample_validation(cfg, "Part1", "Clickbait", "Direct", math.ceil(k / 2), seed)
                + sample_validation(cfg, "Part2", "Clickbait", "Direct", 1, seed + 1)
                + sample_validation(cfg, "Part1", "NonClickbait", "Auto", max(k - math.ceil(k / 2) - 1, 0), seed + 2))
        items += [(a, "aihub-demo", lab.clickbait) for a, lab in demo]
    n_check = max(1, round(len(items) * args.check_ratio)) if args.check_ratio else 0
    checks = (sample_validation(cfg, "Part1", "Clickbait", "Direct", math.ceil(n_check / 2), seed + 3)
              + sample_validation(cfg, "Part1", "NonClickbait", "Auto", n_check // 2, seed + 4))
    items += [(a, "check", lab.clickbait) for a, lab in checks]
    print(f"{request_id}: 실제 기사 {len(arts)}건 + AI-Hub 사례 {len(items) - len(arts) - len(checks)}건 "
          f"+ 점검용 {len(checks)}건, 모델 로드 중…")

    model = AXModel(cfg)
    out = cfg["paths"]["runs"] / "bundles" / f"{request_id}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for a, kind, answer in items:
            j = judge_article(model, a, run_detect=True)
            g = grade(j.label, j.c, len(j.irrelevant), j.truncated, j.error, False, theta)
            write_log(cfg, "LOG-01", {
                "request_id": request_id, "article_id": a.id, "category": a.category, "check_article": kind == "check",
                "label": j.label, "c": j.c, "label_mass": j.label_mass, "irrelevant": list(j.irrelevant),
                "truncated": j.truncated, "error": j.error, "grade": g.grade, "reason": g.reason,
                "theta": theta, "state": "정상", "seconds": j.seconds})
            fh.write(json.dumps({"article": asdict(a), "judgment": asdict(j), "grade": g.grade,
                                 "reason": g.reason, "row": g.row}, ensure_ascii=False) + "\n")
            print(f"[{g.grade}/{g.reason or '-'} · {certainty_word(j.c, theta, low)}] {kind} {a.category} | {a.title[:45]}"
                  f" | 근거 {list(j.evidence)} 무관 {list(j.irrelevant)} | {j.seconds}s")
    print("저장:", out)


if __name__ == "__main__":
    main()
