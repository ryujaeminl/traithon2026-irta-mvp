"""WF1~WF4 실전 실행: 언론사 RSS 기사 수집 → A.X 판정(MDL-01·02) → 등급(MDL-04) → LOG-01 기록.

결과 묶음은 runs/bundles/<요청 ID>.jsonl에 저장하고 화면(app.py)이 읽는다.
실제 기사에는 정답이 없으므로 이 결과는 성능 수치가 아니다(시연·운영 시뮬레이션 전용).
--aihub-demo K: 사람이 낚시로 고쳐 쓴 AI-Hub 검증 기사 등 K건을 함께 넣는다(시연용, 평가 세트와 본문 겹침 없음).
--check-ratio: 확인 목록에 섞을 점검용 기사 비율(BK06 HO-01, 초기 가정 10%). 정답은 응답 후에만 보여 준다.
usage: python scripts/run_news.py --top-n 10 --aihub-demo 4 [--feeds 연합뉴스 경향신문]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irta.data import load_cfg, sample_validation  # noqa: E402
from irta.grade import certainty_word  # noqa: E402
from irta.model import AXModel  # noqa: E402
from irta.news import FEEDS  # noqa: E402
from irta.pipeline import collect, judge_and_save, new_request_id  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-n", type=int, default=10, help="고른 언론사를 합친 최신순 상위 N건")
    ap.add_argument("--feeds", nargs="*", default=list(FEEDS))
    ap.add_argument("--aihub-demo", type=int, default=0)
    ap.add_argument("--check-ratio", type=float, default=0.10)
    args = ap.parse_args()

    cfg = load_cfg()
    theta, low = cfg["rule"]["theta"], cfg["rule"]["certainty_low"]
    request_id = new_request_id()
    arts = collect(args.feeds, args.top_n)
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
    rows = judge_and_save(model, cfg, items, request_id)
    for r in rows:
        a, j = r["article"], r["judgment"]
        print(f"[{r['grade']}/{r['reason'] or '-'} · {certainty_word(j['c'], theta, low)}] {r['kind']} {a['category']} "
              f"| {a['title'][:45]} | 근거 {j['evidence']} 무관 {j['irrelevant']} | {j['seconds']}s")
    out = cfg["paths"]["runs"] / "bundles" / f"{request_id}.jsonl"
    print("저장:", out)


if __name__ == "__main__":
    main()
