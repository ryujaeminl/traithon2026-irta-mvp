"""실행 환경 첫 점검 — 새 GPU(노트북·대회 서버)에서 가장 먼저 돌린다.

1) 대화 템플릿이 응답 시작 직전에서 끝나는지, 레이블 토큰 id가 CFG 값과 같은지
2) 판정 몇 건이 실제로 돌고 형식을 지키는지 (θ 선택용 표본에서 셀별 1건, 조정·평가에는 쓰지 않음)
3) 같은 입력 2회 반복 시 같은 결과인지 (temperature 0, BK05 가정 7)
usage: python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irta.data import load_article, load_cfg, load_manifest  # noqa: E402
from irta.grade import certainty_word, grade  # noqa: E402
from irta.model import AXModel, judge_article  # noqa: E402


def main() -> None:
    cfg = load_cfg()
    t0 = time.perf_counter()
    m = AXModel(cfg)
    torch = m.torch
    print(f"모델 로드 {time.perf_counter() - t0:.0f}초, 정밀도 {cfg['model']['precision']}, "
          f"GPU 메모리 {torch.cuda.memory_allocated() / 2**30:.2f}GB")

    prompt = m.tok.apply_chat_template([{"role": "user", "content": "안녕"}], add_generation_prompt=True,
                                       tokenize=False)
    print("템플릿 끝:", repr(prompt[-40:]))
    for word, want in cfg["model"]["label_token_ids"].items():
        got = m.tok(prompt + word, add_special_tokens=False)["input_ids"][-1]
        alone = m.tok(word, add_special_tokens=False)["input_ids"]
        print(f"  '{word}': 응답 첫머리 토큰 {got} (CFG {want}) {'OK' if got == want and alone == [want] else '불일치!'}")

    rows = load_manifest(cfg, "EVAL-AUTO-v0.1")
    picked, seen = [], set()
    for r in rows:
        key = (r["part"], r["cell"])
        if r["role"] == "theta_select" and key not in seen:
            seen.add(key)
            picked.append(r)
    theta, low = cfg["rule"]["theta"], cfg["rule"]["certainty_low"]
    for r in picked:
        art, lab = load_article(cfg, r)
        j = judge_article(m, art, run_detect=(r["part"] == "Part2"))
        g = grade(j.label, j.c, len(j.irrelevant), j.truncated, j.error, False, theta)
        print(f"[{r['part']} {r['cell']}] 정답 {'낚시' if lab.clickbait else '정상'} → 판정 {j.label} "
              f"c={j.c and round(j.c, 3)} 합={j.label_mass and round(j.label_mass, 3)} 근거 {j.evidence} "
              f"무관 {j.irrelevant}(정답 {lab.irrelevant}) → {g.grade}/{g.reason} {certainty_word(j.c, theta, low)} "
              f"{j.seconds}s {j.error or ''}")
        print("   제목:", art.title[:60], "| 출력:", repr(j.raw.get("judge", ""))[:60])

    art, _ = load_article(cfg, picked[0])
    a, b = judge_article(m, art, False), judge_article(m, art, False)
    print("반복 동일:", (a.label, a.c, a.evidence) == (b.label, b.c, b.evidence), a.c, b.c)
    print(f"최대 GPU 메모리 {torch.cuda.max_memory_allocated() / 2**30:.2f}GB")


if __name__ == "__main__":
    main()
