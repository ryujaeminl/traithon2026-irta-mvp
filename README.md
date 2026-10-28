# 이슈 리서치 신뢰 등급 에이전트 — MVP (TRAITHON 2026, 팀 믹스넛츠)

크리에이터가 이슈 리서치에 쓸 기사마다 A.X 4.0 Light로 낚시성(제목–본문 불일치, 본문 내 무관 문장)을 판정하고,
신뢰 등급(높음/중간/낮음)에 따라 요약 반영·확인·제외를 정하는 에이전트입니다. 설계 근거는 BaseKit 01~07(작성본)에 있습니다.

## 구성

| 경로 | 내용 | BaseKit |
|---|---|---|
| `configs/CFG-v0.1.toml` | 승인 구성: 모델 리비전, 정밀도, θ, 디코딩 | BK05 CFG, BK07 OPS-05 |
| `irta/model.py` | A.X 실행, MDL-01 판정·확신도 c, MDL-02 무관 문장, MDL-03 요약 + 출처 검사 | BK05 |
| `irta/grade.py` | MDL-04 결정표 ①~⑦, 사유 코드, 확실성 3단계 | BK05 §4.3 |
| `irta/data.py` | AI-Hub 기사 로더(정답 노출 항목 차단), 평가 세트 매니페스트 | BK04 |
| `irta/news.py` | 언론사 RSS 수집·본문 추출(robots.txt 준수) | WF2 |
| `irta/logs.py` | LOG-01~06 기록(가명, 마스킹) | BK07 |
| `irta/texts.py` | 고지 문안 TXT-v0.1 | BK03 |
| `app.py` | 크리에이터 화면 + 운영자 콘솔(Streamlit) | BK03·06·07 |
| `scripts/` | 실행 점검, 평가, 보고, 지시문 조정, 실시간 뉴스 판정 | BK05 TEST-41~48 |

## 준비

- Python 3.13, CUDA GPU(노트북 6GB는 `precision = "nf4"`, 대회 GPU는 `"bf16"`으로 바꾸고 새 CFG 버전으로 기록)
- `pip install torch transformers accelerate bitsandbytes streamlit feedparser pytest`
- 데이터는 저장소에 없습니다. 대회 폴더 구조(`../146.낚시성 기사 탐지 데이터`, `../Basekit v1.1 작성본/evidence`)를 그대로 두고 `mvp/`를 그 옆에 둡니다.
- few-shot 파일은 AI-Hub 본문을 담고 있어 저장소에 넣지 않습니다. 처음에 `python scripts/build_fewshot.py`로 만듭니다.

## 실행

```bash
python scripts/smoke_test.py                              # GPU·레이블 토큰·반복 일치 점검
python scripts/run_eval.py --per-cell 5                   # EVAL-AUTO-v0.1 층화 표본 판정 (0이면 전체)
python scripts/report_eval.py                             # TEST-41·42·45, MDL-02 기준선 비교 보고
python scripts/run_news.py --per-feed 2 --aihub-demo 4    # 실제 RSS 기사 + 낚시 사례 + 점검용 기사 판정
streamlit run app.py                                      # 크리에이터 화면·운영자 콘솔
python -m pytest -q tests                                 # 단위 시험
```

시험·리허설은 `IRTA_RUNS=<임시 폴더>`로 기록 폴더를 분리해 운영 로그를 오염시키지 않습니다.

## 원칙

- 결과를 지어내지 않는다: 노트북 4bit 결과는 예비 수치이며, 최종 수치는 대회 GPU(bf16) 재실행으로 낸다.
- 평가 세트(EVAL-v0.1·EVAL-AUTO-v0.1)는 지시문·few-shot 조정에 쓰지 않는다. 조정은 Training 표본으로만 한다.
- 기사 내용의 사실 여부(팩트체크)는 판정하지 않는다.
