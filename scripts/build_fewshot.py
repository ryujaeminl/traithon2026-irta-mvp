"""PRM-v0.1 few-shot 예시 생성 (BK05 MDL-01·02).

- Training에서만 뽑는다. 두 평가 세트와 본문 정규화 해시가 같은 기사는 뺀다(EVD-04-04-03).
- 판정 예시 4건: 사람 작성 낚시 2(서로 다른 패턴) + 정상 2. 근거 번호는 낚시는 referSentenceInfo의 Y 문장,
  정상은 1번 문장(리드)으로 둔다.
- 무관 문장 예시 2건: 사람 작성 삽입 기사 1건의 삽입 문장을 본문 중간으로 옮긴 것 + 정상 1건.
  데이터의 삽입 문장은 모두 끝에 있어(EVD-04-04-01) 위치 지름길을 막으려고 중간 예시를 넣는다.
usage: python scripts/build_fewshot.py
"""
from __future__ import annotations

import json
import random
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irta import prompts  # noqa: E402
from irta.data import body_hash, load_cfg, load_manifest, parse_record  # noqa: E402
from irta.model import FEWSHOT_PATH  # noqa: E402

SEED = 20261009
CATS = ["EC", "ET", "GB", "IS", "LC", "PO", "SO"]
MIN_S, MAX_S = 5, 9


def records(cfg: dict, zname: str):
    z = zipfile.ZipFile(cfg["paths"]["aihub_root"] / "Training" / "02.라벨링데이터" / zname)
    names = sorted(n for n in z.namelist() if n.endswith(".json"))
    for n in random.Random(SEED).sample(names, min(400, len(names))):
        yield json.loads(z.read(n).decode("utf-8-sig"))


def pick(cfg: dict, zname: str, banned: set[str], k: int, ok=lambda rec: True, max_s: int = MAX_S) -> list[dict]:
    out, patterns = [], set()
    for rec in records(cfg, zname):
        src = rec["sourceDataInfo"]
        n = len(src.get("sentenceInfo") or [])
        if body_hash(src.get("newsContent")) in banned or not MIN_S <= n <= max_s or not ok(rec):
            continue
        if src.get("processType") == "D" and src.get("processPattern") in patterns:
            continue  # 사람 작성 낚시는 패턴이 겹치지 않게
        patterns.add(src.get("processPattern"))
        out.append(rec)
        if len(out) == k:
            return out
    raise SystemExit(f"{zname}: 조건에 맞는 기사 {k}건을 못 찾음")


def main() -> None:
    cfg = load_cfg()
    banned = {r["body_sha256_norm"] for s in ("EVAL-v0.1", "EVAL-AUTO-v0.1") for r in load_manifest(cfg, s)}
    rng = random.Random(SEED)
    c1, c2 = rng.sample(CATS, 2)
    shots: list[dict] = []

    judge = [(pick(cfg, f"TL_Part1_Clickbait_Direct_{c1}.zip", banned, 2), "낚시"),
             (pick(cfg, f"TL_Part1_NonClickbait_Auto_{c2}.zip", banned, 2), "정상")]
    for recs, verdict in judge:
        for rec in recs:
            art, _ = parse_record(rec, "Part1")
            refer = [q["sentenceNo"] for q in rec["labeledDataInfo"].get("referSentenceInfo") or []
                     if q.get("referSentenceyn") == "Y"]
            ev = ", ".join(map(str, (refer or [1])[:2]))
            shots.append({"task": "judge", "news_id": art.id, "assistant": f"{verdict}\n근거: {ev}",
                          "user": prompts.judge_user(art.title, art.subtitle, art.sentences)})

    # 사람 작성 삽입은 늘 2문장 이상의 연속 묶음이다 — 묶음째 본문 중간으로 옮긴다
    ins = pick(cfg, f"TL_Part2_Clickbait_Direct_{c1}.zip", banned, 1, max_s=14)[0]
    art, label = parse_record(ins, "Part2")
    block = [art.sentences[i - 1] for i in label.irrelevant]
    rest = [s for i, s in enumerate(art.sentences, 1) if i not in label.irrelevant]
    pos = len(rest) // 2                      # 처음·끝이 아닌 중간
    sents = rest[:pos] + block + rest[pos:]
    nums = ", ".join(str(pos + 1 + k) for k in range(len(block)))
    shots.append({"task": "detect", "news_id": art.id, "assistant": f"무관: {nums}",
                  "user": prompts.detect_user(art.title, sents)})
    norm = pick(cfg, f"TL_Part2_NonClickbait_Auto_{c2}.zip", banned, 1, max_s=14)[0]
    art, _ = parse_record(norm, "Part2")
    shots.append({"task": "detect", "news_id": art.id, "assistant": "무관: 없음",
                  "user": prompts.detect_user(art.title, art.sentences)})

    FEWSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FEWSHOT_PATH.write_text(json.dumps(shots, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(shots)}건 저장: {FEWSHOT_PATH} (판정 카테고리 낚시 {c1} / 정상 {c2})")


if __name__ == "__main__":
    main()
