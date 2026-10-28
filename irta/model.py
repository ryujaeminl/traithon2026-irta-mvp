"""A.X 4.0 Light 실행과 MDL-01(판정)·MDL-02(무관 문장)·MDL-03(요약) (BK05).

확신도 c = max(p낚시, p정상) / (p낚시 + p정상). p는 응답 첫 위치의 원시 logits softmax에서 꺼낸
두 레이블 토큰 확률이다(EVD-05-04-11). 두 확률의 합(label_mass)이 작으면 출력이 형식을 벗어나려 한 것이다.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from . import prompts
from .data import ROOT, Article

FEWSHOT_PATH = ROOT / "prompts" / prompts.VERSION / "fewshot.json"


class AXModel:
    """노트북(nf4 4bit)과 대회 GPU(bf16) 모두 같은 코드로 돈다. 정밀도는 CFG가 정한다."""

    def __init__(self, cfg: dict):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        m = cfg["model"]
        self.cfg = cfg
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(m["id"], revision=m["revision"])
        kw: dict = {"revision": m["revision"], "device_map": "cuda:0"}
        if m["precision"] == "nf4":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True)
        else:
            kw["torch_dtype"] = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(m["id"], **kw).eval()
        self.label_ids = dict(m["label_token_ids"])
        self.fewshot = json.loads(FEWSHOT_PATH.read_text(encoding="utf-8")) if FEWSHOT_PATH.exists() else []
        self._prefixes: dict = {}

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text, add_special_tokens=False)["input_ids"])

    def _ids(self, messages: list[dict], gen: bool):
        text = self.tok.apply_chat_template(messages, add_generation_prompt=gen, tokenize=False)
        return self.tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to("cuda:0")

    def _prefix_cache(self, prefix: list[dict]):
        """지시문+few-shot처럼 매번 같은 앞부분은 한 번만 계산해 KV 캐시로 둔다. shortcut: 메모리에 접두부 종류만큼 쌓임(지금 3종)."""
        key = json.dumps(prefix, ensure_ascii=False)
        if key not in self._prefixes:
            ids = self._ids(prefix, gen=False)
            with self.torch.inference_mode():
                cache = self.model(ids, use_cache=True).past_key_values
            self._prefixes[key] = (ids, cache)
        return self._prefixes[key]

    def chat(self, messages: list[dict], max_new_tokens: int) -> tuple[str, "object"]:
        """탐욕적 디코딩. (생성 텍스트, 첫 위치 원시 logits) 반환. 결과는 캐시 사용 여부와 같아야 한다."""
        import copy

        torch = self.torch
        ids = self._ids(messages, gen=True)
        kw: dict = {}
        if len(messages) > 1:
            p_ids, cache = self._prefix_cache(messages[:-1])
            n = p_ids.shape[1]
            if n < ids.shape[1] and torch.equal(ids[0, :n], p_ids[0]):
                kw["past_key_values"] = copy.deepcopy(cache)
        with torch.inference_mode():
            out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                      do_sample=False, output_logits=True, return_dict_in_generate=True,
                                      pad_token_id=self.tok.eos_token_id, **kw)
        text = self.tok.decode(out.sequences[0, ids.shape[1]:], skip_special_tokens=True)
        return text, out.logits[0][0].float()

    def messages(self, system: str, task: str, user: str) -> list[dict]:
        msgs = [{"role": "system", "content": system}]
        for ex in self.fewshot:
            if ex["task"] == task:
                msgs += [{"role": "user", "content": ex["user"]}, {"role": "assistant", "content": ex["assistant"]}]
        return msgs + [{"role": "user", "content": user}]


# ---------- 파서 (모델 없이 테스트 가능) ----------

def parse_judge(text: str, n_sent: int) -> tuple[str | None, list[int]]:
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    first = re.sub(r"^[^\w가-힣]+", "", lines[0]) if lines else ""
    label = next((lb for lb in ("낚시", "정상") if first.startswith(lb)), None)
    ev_line = next((ln for ln in lines[1:] if ln.startswith("근거")), "")
    nums = [int(x) for x in re.findall(r"\d+", ev_line)]
    return label, [n for n in dict.fromkeys(nums) if 1 <= n <= n_sent][:2]


def parse_detect(text: str, n_sent: int) -> list[int] | None:
    """None = 형식 위반."""
    m = re.search(r"무관\s*[:：]\s*(.*)", text)
    if not m:
        return None
    if "없음" in m.group(1):
        return []
    nums = [int(x) for x in re.findall(r"\d+", m.group(1))]
    return [n for n in dict.fromkeys(nums) if 1 <= n <= n_sent]


def truncate(model: AXModel, sentences: tuple[str, ...]) -> tuple[tuple[str, ...], bool]:
    budget, used, kept = model.cfg["decoding"]["body_max_tokens"], 0, []
    for s in sentences:
        used += model.n_tokens(s) + 4
        if used > budget:
            return tuple(kept), True
        kept.append(s)
    return tuple(kept), False


# ---------- MDL-01 / MDL-02 ----------

@dataclass(frozen=True)
class Judgment:
    article_id: str
    label: str | None
    c: float | None
    label_mass: float | None
    p_f: float | None          # '낚시' 토큰 확률(보정용 원값, TEST-42)
    p_n: float | None          # '정상' 토큰 확률
    evidence: tuple[int, ...]
    irrelevant: tuple[int, ...]
    truncated: bool
    error: str | None
    seconds: float
    raw: dict = field(default_factory=dict)


def judge_article(model: AXModel, art: Article, run_detect: bool = True) -> Judgment:
    t0 = time.perf_counter()
    sents, truncated = truncate(model, art.sentences)
    label = c = mass = pf = pn = None
    evidence: list[int] = []
    irrelevant: list[int] = []
    error = None
    raw: dict = {}
    try:
        text, logits = model.chat(model.messages(prompts.JUDGE_SYSTEM, "judge",
                                                 prompts.judge_user(art.title, art.subtitle, sents)),
                                  model.cfg["decoding"]["judge_max_new_tokens"])
        p = model.torch.softmax(logits, -1)
        pf, pn = (p[model.label_ids[k]].item() for k in ("낚시", "정상"))
        mass = pf + pn
        c = max(pf, pn) / mass if mass > 0 else None
        parsed, evidence = parse_judge(text, len(sents))
        label = ("낚시" if pf > pn else "정상") if parsed else None
        raw["judge"] = text
        if parsed is None:
            error = "형식 위반(판정)"
        if run_detect and error is None:
            dtext, _ = model.chat(model.messages(prompts.DETECT_SYSTEM, "detect",
                                                 prompts.detect_user(art.title, sents)),
                                  model.cfg["decoding"]["judge_max_new_tokens"])
            raw["detect"] = dtext
            found = parse_detect(dtext, len(sents))
            if found is None:
                error = "형식 위반(무관 문장)"
            else:
                irrelevant = found
    except Exception as exc:  # 모델 오류는 등급 규칙 ②로 중간 처리한다(TEST-48)
        error = f"모델 오류: {type(exc).__name__}"
    return Judgment(art.id, label, c, mass, pf, pn, tuple(evidence), tuple(irrelevant), truncated, error,
                    round(time.perf_counter() - t0, 3), raw)


# ---------- MDL-03 요약 + 출처 검사 규칙 v0.1 ----------

def grams(s: str, n: int = 3) -> set[str]:
    t = re.sub(r"\s+", "", s)
    return {t[i:i + n] for i in range(max(len(t) - n + 1, 1))} if t else set()


def overlap(s: str, ref: set[str]) -> float:
    g = grams(s)
    return len(g & ref) / len(g) if g else 0.0


def check_sources(lines: list[str], sources: dict[int, list[str]], excluded: list[str],
                  excl_max: float, support_min: float) -> tuple[list[dict], list[dict]]:
    """반환: (남긴 문장, 지운 문장). 각 항목은 text, sources, weak/why."""
    kept, dropped = [], []
    excl_grams = [grams(e) for e in excluded]
    for ln in lines:
        tags = [int(x) for x in re.findall(r"\[출처\s*(\d+)\]", ln)]
        text = re.sub(r"\s*\[출처\s*\d+\]", "", ln).strip()
        if not text:
            continue
        valid = [t for t in dict.fromkeys(tags) if t in sources]
        if not valid:
            dropped.append({"text": text, "why": "출처 없음"})
            continue
        if any(overlap(text, eg) >= excl_max for eg in excl_grams if eg):
            dropped.append({"text": text, "why": "제외 문장 인용"})
            continue
        ref = set().union(*(grams(" ".join(sources[t])) for t in valid))
        kept.append({"text": text, "sources": valid, "weak": overlap(text, ref) < support_min})
    return kept, dropped


def summarize(model: AXModel, sources: dict[int, list[str]], excluded: list[str]) -> dict:
    """sources: {출처 번호: 허용된 본문 문장}. 실패하면 error를 채우고 화면은 기사 목록만 보여 준다."""
    s = model.cfg["summary"]
    try:
        text, _ = model.chat([{"role": "system", "content": prompts.SUMMARY_SYSTEM},
                              {"role": "user", "content": prompts.summary_user(sorted(sources.items()))}],
                             model.cfg["decoding"]["summary_max_new_tokens"])
    except Exception as exc:
        return {"sentences": [], "dropped": [], "error": f"모델 오류: {type(exc).__name__}", "raw": ""}
    lines = [ln.strip(" -•\t") for ln in text.splitlines() if ln.strip()]
    kept, dropped = check_sources(lines, sources, excluded, s["excluded_overlap_max"], s["support_overlap_min"])
    return {"sentences": kept, "dropped": dropped, "error": None if kept else "요약 문장 없음", "raw": text}
