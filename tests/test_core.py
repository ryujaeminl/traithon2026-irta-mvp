"""모델 없이 도는 단위 시험: 결정표(TEST-48 일부), 파서, 출처 검사, 입력 금지 항목."""
from __future__ import annotations

import itertools
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irta.data import Article, parse_record  # noqa: E402
from irta.grade import HIGH, LOW, MID, certainty_word, grade  # noqa: E402
from irta.logs import mask  # noqa: E402
from irta.model import check_sources, parse_detect, parse_judge  # noqa: E402

T = 0.85


def test_decision_table_all_combinations():
    """판정 × c 구간 × 무관 문장 × 절단 × 오류 × 안전 모드 전 조합에서 높음은 ⑦번 줄 조건일 때만 나온다."""
    for label, c, n_irr, trunc, err, safe in itertools.product(
            ["낚시", "정상", None], [None, 0.5, 0.849, 0.85, 0.99], [0, 2], [False, True],
            [None, "모델 오류: X"], [False, True]):
        g = grade(label, c, n_irr, trunc, err, safe, T)
        high_ok = (not safe and err is None and label == "정상" and c is not None and c >= T
                   and n_irr == 0 and not trunc)
        assert (g.grade == HIGH) == high_ok, (label, c, n_irr, trunc, err, safe, g)
        if err is not None:
            assert g.grade == MID          # 오류는 언제나 중간 (안전 모드 포함)


def test_decision_table_rows():
    assert grade("낚시", 0.9, 0, False, None, False, T).grade == LOW
    assert grade("낚시", 0.9, 0, False, None, True, T).grade == LOW      # 안전 모드에서도 낮음은 유지
    assert grade("정상", 0.9, 0, False, None, True, T).reason == "안전 모드"
    assert grade("정상", 0.9, 1, False, None, False, T).reason == "판단 엇갈림"
    assert grade("정상", 0.9, 0, True, None, False, T).reason == "본문 절단"
    assert grade("정상", 0.84, 0, False, None, False, T).reason == "확신 부족"
    assert grade(None, None, 0, False, "형식 위반(판정)", False, T).reason == "처리 오류"


def test_certainty_words():
    assert certainty_word(0.9, T, 0.6) == "확실함"
    assert certainty_word(0.7, T, 0.6) == "다소 불확실"
    assert certainty_word(0.55, T, 0.6) == "불확실"


def test_parse_judge():
    assert parse_judge("낚시\n근거: 2, 5", 6) == ("낚시", [2, 5])
    assert parse_judge("정상\n근거: 1, 9, 3", 5) == ("정상", [1, 3])      # 범위 밖 번호 제거, 최대 2개
    assert parse_judge("판정: 낚시", 5)[0] is None                       # 첫머리 형식 위반
    assert parse_judge("", 5) == (None, [])


def test_parse_detect():
    assert parse_detect("무관: 없음", 8) == []
    assert parse_detect("무관: 7, 8, 12", 8) == [7, 8]
    assert parse_detect("모르겠다", 8) is None


def test_check_sources():
    src = {1: ["정부는 내년 예산을 3조원 늘리기로 했다."], 2: ["시장은 새 버스 노선을 발표했다."]}
    lines = ["정부는 내년 예산을 3조원 늘리기로 했다 [출처1]",
             "출처 없는 문장이다",
             "카카오가 코인 300만개를 기부했다 [출처2]",
             "시장은 우주 정거장을 짓겠다고 밝혔다 [출처9]"]
    kept, dropped = check_sources(lines, src, ["카카오가 코인 300만개를 기부하였다."], 0.5, 0.3)
    assert [k["sources"] for k in kept] == [[1]] and not kept[0]["weak"]
    assert {d["why"] for d in dropped} == {"출처 없음", "제외 문장 인용"}


def test_article_has_no_label_fields():
    """정답을 드러내는 항목(BK04 입력 금지)이 Article에 없어야 한다."""
    names = {f.name for f in fields(Article)}
    for banned in ("useType", "processType", "processPattern", "processLevel", "referSentenceInfo",
                   "clickbaitClass", "newsTitle", "label", "pattern", "irrelevant"):
        assert banned not in names


def test_parse_record_part1_uses_new_title_and_part2_inserted_sentences():
    rec1 = {"sourceDataInfo": {"newsID": "SO_1", "newsCategory": "사회", "newsTitle": "원 제목",
                               "newsSubTitle": "", "processType": "D", "processPattern": "14",
                               "sentenceInfo": [{"sentenceNo": 2, "sentenceContent": "둘"},
                                                {"sentenceNo": 1, "sentenceContent": "하나"}]},
            "labeledDataInfo": {"newTitle": "낚시 제목", "clickbaitClass": "0"}}
    a, lab = parse_record(rec1, "Part1")
    assert a.title == "낚시 제목" and a.sentences == ("하나", "둘") and lab.clickbait and lab.method == "Direct"
    rec2 = {"sourceDataInfo": {"newsID": "EC_2", "newsCategory": "경제", "newsTitle": "제목",
                               "processType": "A", "processPattern": "99", "sentenceInfo": []},
            "labeledDataInfo": {"clickbaitClass": "0", "processSentenceInfo": [
                {"sentenceNo": 1, "sentenceContent": "본문", "subjectConsistencyYn": "Y"},
                {"sentenceNo": 2, "sentenceContent": "삽입", "subjectConsistencyYn": "N"}]}}
    a, lab = parse_record(rec2, "Part2")
    assert a.sentences == ("본문", "삽입") and lab.irrelevant == (2,)


def test_mask():
    assert mask("연락 010-1234-5678, a.b@c.com") == "연락 [전화번호], [이메일]"
