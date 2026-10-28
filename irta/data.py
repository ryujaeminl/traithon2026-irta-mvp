"""구성(CFG) 읽기와 기사 데이터 (BK04 DATA-01·02·04, DATA-05 입력).

기사(Article)에는 모델 입력에 허용된 항목만 담고, 정답(Label)은 따로 둔다.
useType·process*·referSentenceInfo·원 제목(Part1 낚시의 newsTitle)·파일명은 정답을 드러내므로
Article에 넣지 않는다(BK04 입력 금지 항목, EVD-04-04-02).
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import tomllib
import unicodedata
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NONWORD = re.compile(r"[\W_]+")


def load_cfg(name: str = "CFG-v0.1") -> dict:
    with open(ROOT / "configs" / f"{name}.toml", "rb") as fh:
        cfg = tomllib.load(fh)
    cfg["paths"] = {k: (ROOT / v).resolve() for k, v in cfg["paths"].items()}
    return cfg


# norm_title·body_hash는 evidence/scripts/dq_checks.py와 같은 정의다. 매니페스트의 body_sha256_norm과
# 같은 값을 내야 본문 해시 중복 제거(EVD-04-04-03)가 성립하므로 바꾸지 않는다.
def norm_title(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").replace('\\"', '"').replace("\\'", "'")).strip()


def body_hash(text: str) -> str:
    return hashlib.sha256(NONWORD.sub("", unicodedata.normalize("NFC", text or "")).encode()).hexdigest()


@dataclass(frozen=True)
class Article:
    """모델과 화면에 넘기는 기사. 정답 관련 항목은 없다."""
    id: str
    source: str                 # "aihub" | "rss"
    category: str
    title: str
    subtitle: str
    sentences: tuple[str, ...]  # 본문 문장(1번부터)
    url: str = ""
    published: str = ""

    @property
    def body(self) -> str:
        return " ".join(self.sentences)


@dataclass(frozen=True)
class Label:
    """평가 전용 정답. 모델 입력에 절대 넘기지 않는다."""
    news_id: str
    part: str                   # "Part1" | "Part2"
    clickbait: bool
    method: str                 # "Auto" | "Direct"
    pattern: str
    irrelevant: tuple[int, ...] = field(default=())  # Part2 삽입(N) 문장 번호(1부터)


def _sentences(raw: list[dict]) -> tuple[str, ...]:
    return tuple((s.get("sentenceContent") or "").strip() for s in sorted(raw, key=lambda s: s["sentenceNo"]))


def parse_record(rec: dict, part: str) -> tuple[Article, Label]:
    src, lab = rec["sourceDataInfo"], rec["labeledDataInfo"]
    clickbait = str(lab.get("clickbaitClass")) == "0"
    if part == "Part1":
        title = lab["newTitle"]            # 낚시면 가공된 제목, 정상이면 원 제목과 같다
        sents = _sentences(src["sentenceInfo"])
        irrelevant: tuple[int, ...] = ()
    else:
        title = src["newsTitle"]
        psi = sorted(lab["processSentenceInfo"], key=lambda s: s["sentenceNo"])
        sents = _sentences(psi)            # 삽입 문장은 processSentenceInfo에만 있다(EVD-04-04-01)
        irrelevant = tuple(s["sentenceNo"] for s in psi if s.get("subjectConsistencyYn") == "N")
    art = Article(id=src["newsID"], source="aihub", category=src["newsCategory"],
                  title=norm_title(title), subtitle=norm_title(src.get("newsSubTitle") or ""), sentences=sents)
    label = Label(news_id=src["newsID"], part=part, clickbait=clickbait,
                  method="Direct" if src.get("processType") == "D" else "Auto",
                  pattern=str(src.get("processPattern")), irrelevant=irrelevant)
    return art, label


@lru_cache(maxsize=None)
def _zip(path: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(path)


def load_manifest(cfg: dict, eval_set: str = "EVAL-AUTO-v0.1") -> list[dict]:
    path = cfg["paths"]["manifests"] / f"EVD-04-04-09_{eval_set}_manifest.csv"
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_article(cfg: dict, row: dict) -> tuple[Article, Label]:
    """매니페스트 한 줄(label_zip, entry, part)로 Validation 라벨 zip에서 기사를 꺼낸다."""
    zpath = cfg["paths"]["aihub_root"] / "Validation" / "02.라벨링데이터" / row["label_zip"]
    rec = json.loads(_zip(str(zpath)).read(row["entry"]).decode("utf-8-sig"))
    return parse_record(rec, row["part"])
