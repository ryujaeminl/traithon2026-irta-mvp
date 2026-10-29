"""이슈 리서치 신뢰 등급 에이전트 — 크리에이터 화면 + 운영자 콘솔 (BK03 TLD, BK06 HO-01~04, BK07 OPS).

usage: streamlit run app.py
판정은 scripts/run_news.py가 미리 만든 묶음(runs/bundles)을 읽는다. 요약(MDL-03)만 버튼을 눌렀을 때 A.X로 만든다.
"""
from __future__ import annotations

import hashlib
import html
import json
import random
import time

from pathlib import Path

import streamlit as st

from irta.data import load_cfg
from irta.grade import HIGH, LOW, MID, REASON_TEXT, certainty_word, grade
from irta.logs import mask, now, read_log, write_log
from irta.metrics import calibrated_c, log_odds
from irta.news import FEEDS
from irta.texts import VERSION as TXT_VERSION
from irta.texts import T

st.set_page_config(page_title="이슈 리서치 신뢰 등급 에이전트", page_icon="🔎", layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button { font-family: 'Noto Sans KR', sans-serif; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding-top: 2rem; max-width: 1100px; }
h1, h2, h3 { letter-spacing: -0.02em; }
[data-testid="stVerticalBlockBorderWrapper"] { background: #FFFFFF; border-radius: 14px !important;
  border: 1px solid #E6E8EE !important; box-shadow: 0 1px 3px rgba(20, 30, 60, .06); }
.badge { display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: .82rem; font-weight: 700;
  margin-right: 8px; vertical-align: 2px; }
.b-높음 { background: #E7F6EC; color: #17663A; } .b-중간 { background: #FFF4DB; color: #8A5A00; }
.b-낮음 { background: #FDECEC; color: #A42323; } .b-정보 { background: #EEF2FD; color: #2F4BB5; }
.chip { display: inline-block; padding: 2px 8px; border-radius: 6px; background: #F1F3F7; color: #4A5163;
  font-size: .78rem; margin-left: 6px; }
.title { font-size: 1.08rem; font-weight: 700; color: #1F2430; line-height: 1.5; }
.quote { border-left: 4px solid #2F5BEA; background: #F5F7FF; padding: 10px 14px; border-radius: 0 10px 10px 0;
  margin: 8px 0; color: #2A3042; line-height: 1.65; }
.quote.warn { border-left-color: #D99000; background: #FFF9EC; }
.qnum { color: #6B7385; font-size: .78rem; margin-left: 6px; }
.small { color: #6B7385; font-size: .82rem; }
.banner { background: #FFFFFF; border: 1px solid #DCE3FB; border-radius: 14px; padding: 14px 18px; margin-bottom: 12px; }
.banner b { color: #2F5BEA; }
</style>""", unsafe_allow_html=True)


def esc(text: str) -> str:
    return html.escape(text or "")


BADGE = {"높음": "✔ 높음", "중간": "⚠ 중간", "낮음": "✕ 낮음"}


def badge(grade_: str, label: str | None = None) -> str:
    return f'<span class="badge b-{grade_}">{label or BADGE.get(grade_, grade_)}</span>'


def card_title(a: dict, grade_: str, label: str | None = None) -> None:
    st.markdown(f'{badge(grade_, label)}<span class="title">{esc(a["title"])}</span>'
                f'<span class="chip">{esc(a["category"])}</span>', unsafe_allow_html=True)
cfg = load_cfg()
RUNS = cfg["paths"]["runs"]
STATE = RUNS / "state.json"
ss = st.session_state


# ---------- 운영 상태 · 보정 ----------

def op_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"state": "정상"}


def set_state(new: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(new, ensure_ascii=False), encoding="utf-8")


def temperature() -> float:
    """TEST-42에서 맞춘 온도 T가 있으면 쓴다(없으면 보정 전 확률)."""
    p = RUNS / "eval" / f"{cfg['version']}-{cfg['model']['precision']}" / "report.json"
    return json.loads(p.read_text(encoding="utf-8"))["temperature"] if p.exists() else 1.0


def regrade(row: dict, safe: bool, t: float) -> dict:
    j = row["judgment"]
    c = calibrated_c(log_odds(j["p_f"], j["p_n"]), t) if j.get("p_f") is not None else j["c"]
    g = grade(j["label"], c, len(j["irrelevant"]), j["truncated"], j["error"], safe, cfg["rule"]["theta"])
    return {**row, "c": c, "grade": g.grade, "reason": g.reason}


def gpu_ready() -> bool:
    """Streamlit Cloud처럼 GPU가 없는 곳에서는 판정·요약을 이 화면에서 할 수 없다. IRTA_NO_GPU=1이면 GPU를 쓰지 않는다."""
    import os
    if os.environ.get("IRTA_NO_GPU"):
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@st.cache_resource(show_spinner="A.X 모델을 불러오는 중…")
def get_model():
    from irta.model import AXModel
    return AXModel(cfg)




def _find_secret(name: str) -> str | None:
    """Streamlit secrets에서 키를 찾는다. 최상위·[섹션] 안·소문자 표기 모두 허용."""
    try:
        sec = st.secrets.to_dict()
    except Exception as exc:  # secrets.toml이 없거나 문법 오류
        ss["secret_err"] = type(exc).__name__
        return None

    def walk(d: dict) -> str | None:
        for k, v in d.items():
            if isinstance(v, dict):
                found = walk(v)
                if found:
                    return found
            elif k.upper() == name and str(v).strip():
                return str(v).strip()
        return None
    return walk(sec)


def remote_api() -> tuple[str, str] | None:
    """판정 서버 주소·토큰 (Streamlit Cloud secrets 또는 환경변수)."""
    import os
    url = _find_secret("IRTA_API_URL") or os.environ.get("IRTA_API_URL")
    tok = _find_secret("IRTA_API_TOKEN") or os.environ.get("IRTA_API_TOKEN")
    ss["api_missing"] = [n for n, v in (("IRTA_API_URL", url), ("IRTA_API_TOKEN", tok)) if not v]
    return (url.rstrip("/"), tok) if url and tok else None


def _api(url: str, tok: str, path: str, body: dict | None = None) -> dict:
    import urllib.request
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(f"{url}{path}", data=data, method="POST" if body is not None else "GET",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {tok}"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def search_news(query: str, feeds: list[str], top_n: int) -> None:
    """WF1·WF2 이슈 리서치 요청. GPU가 있으면 여기서 판정, 없으면 판정 서버(접수→조회), 둘 다 없으면 목록만."""
    import urllib.error

    from irta.pipeline import collect, judge_and_save, new_request_id

    ss["unjudged"] = None          # 이전 '판정 대기' 목록이 새 결과를 가리지 않게 지운다
    if gpu_ready():
        with st.spinner("이슈 기사를 찾는 중…"):
            arts = collect(feeds, top_n, query)
        if not arts:
            st.sidebar.warning("최근 기사 중 이 이슈와 맞는 기사를 찾지 못했습니다. 검색어를 바꿔 보세요.")
            return
        bar = st.progress(0.0, text=f"A.X 판정 0/{len(arts)}")
        rid = new_request_id()
        judge_and_save(get_model(), cfg, [(a, "rss", None) for a in arts], rid, query=query,
                       progress=lambda i, n: bar.progress(i / n, text=f"A.X 판정 {i}/{n}"))
        ss["pick"] = rid
        return
    api = remote_api()
    if api:
        url, tok = api
        try:
            jid = _api(url, tok, "/judge", {"feeds": feeds, "top_n": top_n, "query": query})["job_id"]
            bar = st.progress(0.0, text="판정 서버에 요청했습니다…")
            for _ in range(600):                     # 최대 약 20분
                time.sleep(2)
                job = _api(url, tok, f"/jobs/{jid}")
                if job["status"] != "running":
                    break
                n = max(job.get("total") or 1, 1)
                bar.progress(min(job.get("done", 0) / n, 1.0), text=f"{job['stage']} {job.get('done', 0)}/{n}")
        except urllib.error.HTTPError as exc:
            st.sidebar.error(f"판정 서버 오류 {exc.code}" + (" — 다른 판정이 진행 중입니다." if exc.code == 429 else
                             " — 토큰이 맞지 않습니다." if exc.code == 401 else ""))
            return
        except Exception as exc:
            st.sidebar.error(f"판정 서버에 연결하지 못했습니다({type(exc).__name__}). 노트북 서버·터널이 켜져 있는지 확인하세요.")
            return
        if job["status"] != "done":
            st.sidebar.error(job.get("stage", "판정 실패"))
            return
        res = job["result"]
        if not res["rows"]:
            st.sidebar.warning("최근 기사 중 이 이슈와 맞는 기사를 찾지 못했습니다. 검색어를 바꿔 보세요.")
            return
        out = RUNS / "bundles" / f"{res['request_id']}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + chr(10) for r in res["rows"]), encoding="utf-8")
        ss["pick"] = res["request_id"]
        return
    with st.spinner("기사 목록을 가져오는 중…"):
        ss["unjudged"] = collect(feeds, top_n, query)


# ---------- 사이드바 ----------

st.sidebar.markdown("### 🔎 이슈 리서치\n**신뢰 등급 에이전트**")
st.sidebar.caption(T("START"))
user = st.sidebar.text_input("크리에이터 가명 ID", value="creator-01")
# 노트북·GPU에서 판정한 묶음(runs/bundles) + 저장소에 올린 시연 묶음(demo/bundles, Streamlit Cloud용)
DEMO = Path(__file__).resolve().parent / "demo" / "bundles"
with st.sidebar.container(border=True):
    st.markdown("**이슈 리서치 요청**")
    query = st.text_input("어떤 이슈를 정리할까요?", placeholder="예: 구글 과징금, 전기차 배터리 화재",
                          help="비워 두면 최신 기사 Top N을 가져옵니다.")
    feeds_sel = st.multiselect("언론사", list(FEEDS), default=list(FEEDS))
    top_n = st.select_slider("가져올 기사 수 (Top N)", [5, 10, 20, 30], value=10)
    mode = "이 컴퓨터 GPU로 판정" if gpu_ready() else ("판정 서버로 판정" if remote_api() else "판정 서버 미연결 — 목록만")
    st.caption(f"판정 방식: {mode}")
    if mode.startswith("판정 서버 미연결"):
        why = ss.get("secret_err") or ", ".join(ss.get("api_missing") or [])
        st.caption(f"Secrets에서 찾지 못한 값: {why} — 앱 Settings → Secrets에 최상위로 넣고 저장한 뒤 'Reboot app'을 누르세요.")
    if st.button("이슈 기사 찾기", type="primary", use_container_width=True, disabled=not feeds_sel):
        search_news(query.strip(), feeds_sel, top_n)
bundles = sorted([*(RUNS / "bundles").glob("*.jsonl"), *DEMO.glob("*.jsonl")], key=lambda p: p.stem, reverse=True)
if ss.get("unjudged") is not None:
    st.title("가져온 기사 (판정 대기)")
    st.warning("판정 서버가 연결되지 않아 A.X 판정을 하지 못했습니다. 아래는 기사 목록뿐이며 신뢰 등급이 없습니다.")
    for a in ss["unjudged"]:
        st.markdown(f"- {esc(a.title)} <span class='chip'>{esc(a.category)}</span> [원문 ↗]({a.url})",
                    unsafe_allow_html=True)
    if st.button("목록 닫기"):
        ss["unjudged"] = None
        st.rerun()
    st.stop()
if not bundles:
    st.info("판정된 기사 묶음이 없습니다. 왼쪽 '기사 찾기'를 누르거나 `python scripts/run_news.py --top-n 10`을 실행하세요.")
    st.stop()
names = [p.stem for p in bundles]
pick = names.index(ss["pick"]) if ss.get("pick") in names else 0
bundle = st.sidebar.selectbox("리서치 요청(묶음)", bundles, index=pick, format_func=lambda p: p.stem)
page = st.sidebar.radio("화면", ["리서치 결과", "요약", "운영자 콘솔", "실험 설정"])
ss.setdefault("condition", "B 근거 먼저")
state = op_state()
safe = state["state"] == "안전 모드"
t_cal = temperature()
rows = [regrade(json.loads(ln), safe, t_cal) for ln in bundle.open(encoding="utf-8") if ln.strip()]
req = bundle.stem
dec = ss.setdefault(f"dec-{req}", {})          # 기사 ID → 결정 기록
if safe:
    st.warning(T("SAFE-MODE", why=state.get("why", "-")))


def log_decision(art_id: str, row: dict, decision: str, reason: str = "") -> None:
    if dec.get(art_id, {}).get("decision") == decision:   # 같은 결정 중복 기록 방지(다시 누름·재실행)
        return
    shown = ss.get(f"shown-{req}-{art_id}", time.time())
    rec = {"request_id": req, "creator": user, "article_id": art_id, "grade": row["grade"],
           "certainty": certainty_word(row["c"], cfg["rule"]["theta"], cfg["rule"]["certainty_low"]),
           "q1": ss.get(f"q1-{req}-{art_id}"), "decision": decision, "reason": mask(reason),
           "seconds": round(ss.get(f"q1t-{req}-{art_id}", time.time()) - shown, 1),
           "opened_ai_before_answer": ss.get(f"open-before-{req}-{art_id}", False),
           "overrode_ai": (decision in ("사용", "복원")) and row["judgment"]["label"] == "낚시",
           "condition": ss["condition"], "text_version": TXT_VERSION,
           "check_article": row.get("kind") == "check", "check_correct": check_correct(row, ss.get(f"q1-{req}-{art_id}"))}
    write_log(cfg, "LOG-02", rec)
    dec[art_id] = rec


def check_correct(row: dict, q1: str | None) -> bool | None:
    """점검용 기사 정답 여부(KPI-10). '예'=제목 주장이 본문에서 확인됨=정상, '아니오'=낚시, '모르겠음'=오답."""
    if row.get("kind") != "check":
        return None
    return (q1 == "예" and row["answer"] is False) or (q1 == "아니오" and row["answer"] is True)


def evidence_block(row: dict) -> None:
    a, j = row["article"], row["judgment"]
    parts = [f'<div class="quote">「{esc(a["sentences"][n - 1])}」<span class="qnum">본문 {n}번째 문장</span></div>'
             for n in j["evidence"]]
    parts += [f'<div class="quote warn">주제와 무관해 보이는 문장 — 「{esc(a["sentences"][n - 1])}」'
              f'<span class="qnum">본문 {n}번째 문장</span></div>' for n in j["irrelevant"]]
    st.markdown("".join(parts), unsafe_allow_html=True)
    if a.get("url"):
        st.markdown(f"[원문 보기 ↗]({a['url']})")
    if row.get("related"):
        items = "".join(f'<div class="small">· {esc(o["source"])} — <a href="{esc(o["url"])}" target="_blank">'
                        f'{esc(o["title"])}</a><br>&nbsp;&nbsp;「{esc(o["lead"][:120])}」</div>' for o in row["related"])
        st.markdown(f'<div class="banner"><b>같은 이슈의 다른 기사</b> (참고용 · 등급에는 쓰지 않음){items}</div>',
                    unsafe_allow_html=True)


def explain(row: dict) -> list[str]:
    """이 기사가 왜 이 등급인지, 판정 내용으로 풀어 쓴 문장들(숫자·전문 용어 없음, BK03)."""
    j, a, reason = row["judgment"], row["article"], row["reason"]
    theta, low = cfg["rule"]["theta"], cfg["rule"]["certainty_low"]

    def quote(nums: list[int]) -> str:
        qs = [f"{n}번째 문장 「{a['sentences'][n - 1][:80]}」" for n in nums if 0 < n <= len(a["sentences"])
              and a["sentences"][n - 1]]
        return ", ".join(qs)

    out: list[str] = []
    if reason == "안전 모드":
        out.append(T("WHY-안전 모드", why=state.get("why", "-")))
    elif reason == "처리 오류":
        err = j.get("error") or ""
        out.append(T("WHY-처리 오류-무관" if "무관" in err else "WHY-처리 오류-판정" if "형식" in err
                     else "WHY-처리 오류-모델"))
    elif reason == "확신 부족":
        out.append(T("WHY-확신 부족-갈피") if (row["c"] or 0) < low else T("WHY-확신 부족-약함", label=j["label"]))
        out.append(T("WHY-확신 부족-확인"))
    elif reason == "판단 엇갈림":
        where = quote(j["irrelevant"][:2])
        out.append(T("WHY-판단 엇갈림", where=f"({where})" if where else ""))
    elif reason == "본문 절단":
        out.append(T("WHY-본문 절단"))
    elif reason == "낚시 확신":
        out.append(T("WHY-낚시 확신"))
    else:
        out.append(T("WHY-높음"))
    if j.get("evidence") and reason not in ("처리 오류",):
        ev = quote(j["evidence"])
        if ev:
            out.append(T("WHY-근거", ev=ev))
    return out


def ai_panel(row: dict) -> None:
    word = certainty_word(row["c"], cfg["rule"]["theta"], cfg["rule"]["certainty_low"])
    extra = T("TLD-01-c-엇갈림") if row["reason"] == "판단 엇갈림" else ""
    why = "".join(f"<li>{esc(t)}</li>" for t in explain(row))
    st.markdown(f'{badge(row["grade"], T("TLD-01-a-" + row["grade"]))} <span class="small">AI 판단 · '
                f'{esc(T("TLD-01-c", word=word) + extra)}</span><ul class="small" style="margin-top:6px">{why}</ul>',
                unsafe_allow_html=True)


# ---------- 리서치 결과 (HO-01·02, TLD-01·03) ----------

def page_results() -> None:
    st.title("리서치 결과")
    q = next((r.get("query") for r in rows if r.get("query")), "")
    st.markdown(f'<span class="small">{"이슈: " + esc(q) + " · " if q else ""}{esc(req)} · 기사 {len(rows)}건</span>',
                unsafe_allow_html=True)
    # 점검용 기사는 등급과 무관하게 확인 목록에 섞고(구별되지 않게 순서를 섞음), 요약에는 넣지 않는다(BK06 HO-01)
    mids = sorted([r for r in rows if r["grade"] == MID or r.get("kind") == "check"],
                  key=lambda r: hashlib.sha1((req + r["article"]["id"]).encode()).hexdigest())
    highs = [r for r in rows if r["grade"] == HIGH and r.get("kind") != "check"]
    lows = [r for r in rows if r["grade"] == LOW and r.get("kind") != "check"]
    m1, m2, m3 = st.columns(3)
    m1.metric("✔ 요약에 반영 (높음)", len(highs))
    m2.metric("⚠ 확인 필요 (중간)", len(mids))
    m3.metric("✕ 제외 (낮음)", len(lows))
    st.markdown(f'<div class="banner">{esc(T("TLD-01-e"))}</div>', unsafe_allow_html=True)

    st.subheader("확인 필요 (중간)")
    st.caption(T("TLD-03-a-보류"))
    for r in mids:
        a = r["article"]
        aid = a["id"]
        with st.container(border=True):
            ss.setdefault(f"shown-{req}-{aid}", time.time())
            card_title(a, MID, "⚠ 중간 · 확인 필요")
            if ss["condition"].startswith("A"):
                ai_panel(r)                       # TEST-55 대조 조건: 등급 먼저
            evidence_block(r)

            def answered(aid=aid):
                ss[f"q1t-{req}-{aid}"] = time.time()

            st.radio(T("Q1"), ["예", "아니오", "모르겠음"], index=None, key=f"q1-{req}-{aid}", horizontal=True,
                     on_change=answered)
            if ss["condition"].startswith("B"):
                def opened(aid=aid):
                    if ss.get(f"q1-{req}-{aid}") is None:
                        ss[f"open-before-{req}-{aid}"] = True

                if st.toggle("AI 판단 펼치기", key=f"open-{req}-{aid}", on_change=opened):
                    ai_panel(r)
            q1 = ss.get(f"q1-{req}-{aid}")
            reason = ""
            if q1 == "아니오":
                reason = st.text_input("'아니오'라고 답한 기사를 쓰려면 사유 1줄을 적어 주세요", key=f"why-{req}-{aid}")
            c1, c2, c3 = st.columns([1, 1, 4])
            if c1.button("사용", key=f"use-{req}-{aid}", disabled=q1 is None or (q1 == "아니오" and not reason.strip())):
                log_decision(aid, r, "사용", reason)
            if c2.button("사용 안 함", key=f"skip-{req}-{aid}", disabled=q1 is None):
                log_decision(aid, r, "미사용")
            if aid in dec:
                c3.success(f"기록됨: {dec[aid]['decision']}")
                if r.get("kind") == "check":
                    st.info(f"이 기사는 확인 정확도를 재는 점검용 기사였습니다. 정답: {'낚시' if r['answer'] else '정상'} "
                            f"— 회원님 응답 {'정답' if dec[aid]['check_correct'] else '오답'}. 요약에는 들어가지 않습니다.")
            st.markdown(f'<div class="small">{esc(T("TLD-01-d"))}</div>', unsafe_allow_html=True)

    st.subheader("요약에 반영됨 (높음)")
    for r in highs:
        with st.expander(f"✔ {r['article']['title']}  ·  {r['article']['category']}"):
            st.markdown(badge(HIGH, T("TLD-01-a-높음")), unsafe_allow_html=True)
            evidence_block(r)
            st.caption(T("TLD-01-d"))

    st.subheader("제외 기사 (낮음)")
    st.caption(T("TLD-03-a"))
    for r in lows:
        a, aid = r["article"], r["article"]["id"]
        with st.container(border=True):
            card_title(a, LOW, "✕ 낮음 · 요약에서 제외")
            why = " ".join(explain(r))
            st.markdown(f'<div class="small">제외 사유: {esc(why)}</div>', unsafe_allow_html=True)
            evidence_block(r)
            if aid in dec:
                st.success("낮음에서 복원됨")
                continue
            why = st.text_area(T("TLD-03-c"), key=f"rwhy-{req}-{aid}")
            st.caption(T("TLD-03-d"))
            if st.button("복원 확정", key=f"rest-{req}-{aid}"):
                if not why.strip():
                    st.error(T("TLD-03-오류"))
                else:
                    log_decision(aid, r, "복원", why)
                    st.rerun()


# ---------- 요약 (MDL-03, TLD-02·04) ----------

def page_summary() -> None:
    from irta.model import summarize

    st.header("요약 초안")
    rows_s = [r for r in rows if r.get("kind") != "check"]          # 점검용 기사는 요약 제외
    used_mid = [r for r in rows_s if r["grade"] == MID and dec.get(r["article"]["id"], {}).get("decision") == "사용"]
    restored = [r for r in rows_s if r["grade"] == LOW and dec.get(r["article"]["id"], {}).get("decision") == "복원"]
    highs = [] if safe else [r for r in rows_s if r["grade"] == HIGH]   # 안전 모드: 자동 반영 없음
    use = (highs + used_mid + restored)[:cfg["summary"]["bundle_max_articles"]]
    excluded = len(rows_s) - len(use)
    st.markdown(f"**[AI 요약 초안]** {T('TLD-02-a')}  \n"
                + T("TLD-02-a-건수", h=len(highs), m=len(used_mid), r=len(restored), x=excluded))
    st.caption(T("TLD-02-b"))
    if not use:
        st.info("요약에 쓸 기사가 없습니다. 확인 필요 기사를 확인하거나 높음 기사가 있는 묶음을 고르세요.")
        return
    if not gpu_ready():
        st.info("이 화면은 GPU가 없는 환경(시연 서버)이라 A.X 요약을 만들 수 없습니다. 대체 처리로 기사 목록·등급·출처만 보여 드립니다.")
        for n, r in enumerate(use, 1):
            a = r["article"]
            st.markdown(f"- [출처{n} · {esc(a['category'])} · {r['grade']}] {esc(a['title'])}"
                        + (f" — [원문]({a['url']})" if a.get("url") else ""))
        return
    if st.button("요약 생성 (A.X)"):
        sources, excl, meta = {}, [], {}
        for n, r in enumerate(use, 1):
            a, irr = r["article"], set(r["judgment"]["irrelevant"])
            sources[n] = [s for i, s in enumerate(a["sentences"], 1) if i not in irr]
            excl += [a["sentences"][i - 1] for i in irr]
            meta[n] = r
        ss[f"sum-{req}"] = (summarize(get_model(), sources, excl), meta, len(excl))
    if f"sum-{req}" in ss:
        res, meta, n_excl = ss[f"sum-{req}"]
        if res["error"]:
            st.error(f"요약 실패({res['error']}) — 기사 목록·등급·출처만 제공합니다.")
            for n, r in meta.items():
                st.markdown(f"- [출처{n}] {r['article']['title']} · {r['grade']}")
            return
        for s in res["sentences"]:
            tags = []
            for n in s["sources"]:
                r = meta[n]
                who = (" — 회원님 확인" if r["grade"] == MID else
                       " — 낮음에서 복원" if r["grade"] == LOW else "")
                tags.append(f"[출처{n} · {r['article']['category']} · {r['grade']}{who}]")
            st.markdown(f"{s['text']} {' '.join(tags)}" + (" ⚠️근거 약함" if s["weak"] else ""))
        if n_excl:
            st.caption(T("TLD-02-d", n=n_excl))
        if res["dropped"]:
            with st.expander(f"출처 검사로 지운 문장 {len(res['dropped'])}개"):
                for d in res["dropped"]:
                    st.write(f"- {d['text']} ({d['why']})")
        links = ", ".join(meta[n]["article"].get("url") or meta[n]["article"]["id"] for n in meta)
        st.text_area("내보내기용 표기 문구(TLD-04 초안)", T("TLD-04-draft", sources=links))


# ---------- 운영자 콘솔 (HO-03·04, OPS-01·03) ----------

def page_operator() -> None:
    st.header("운영자 콘솔")
    st.caption("크리에이터 화면에는 보이지 않는 정보(확신도 원값, 복원 사유)를 다룹니다. 크리에이터는 가명으로만 표시합니다.")
    l1, l2, l3 = read_log(cfg, "LOG-01"), read_log(cfg, "LOG-02"), read_log(cfg, "LOG-03")
    grades = [r["grade"] for r in l1]
    n = len(grades) or 1
    decided = [r for r in l2 if r["decision"] in ("사용", "미사용")]
    restores = [r for r in l2 if r["decision"] == "복원"]
    fast = sum(r["seconds"] < 3 for r in decided)
    audited = [r for r in l3 if r.get("kind") == "높음 표본"]
    fake = sum(r["verdict"] == "낚시" for r in audited)
    checks = [r for r in l2 if r.get("check_article")]
    c = st.columns(6)
    c[0].metric("판정 기사", len(grades))
    c[1].metric("중간 비율 (KPI-05)", f"{grades.count(MID) / n:.0%}", help="경보 > 30%")
    c[2].metric("3초 미만 확인 (KPI-09)", f"{fast / len(decided):.0%}" if decided else "—", help="경보 > 20%")
    c[3].metric("낮음 복원 수 (KPI-08)", len(restores))
    c[4].metric("표본 감사 낚시 (KPI-12)", f"{fake}/{len(audited)}" if audited else "—", help="40건 중 3건 이상이면 TRG-04")
    c[5].metric("점검용 기사 정답률 (KPI-10)",
                f"{sum(bool(r['check_correct']) for r in checks) / len(checks):.0%}" if checks else "—", help="경보 < 70%")

    st.subheader("안전 모드 (HO-04 · OPS-03)")
    st.write(f"현재 상태: **{state['state']}**" + (f" — {state.get('why')} ({state.get('by')}, {state.get('ts')})"
                                                     if safe else ""))
    if not safe:
        why = st.text_input("전환 사유 (예: TRG-04 높음 표본 낚시 3/40)")
        op = st.text_input("운영 담당(ROLE-02) 가명")
        if st.button("안전 모드 전환", disabled=not (why and op)):
            set_state({"state": "안전 모드", "why": why, "by": op, "ts": now()})
            write_log(cfg, "LOG-04", {"event": "안전 모드 진입", "why": why, "by": op,
                                      "state_before": "정상", "state_after": "안전 모드"})
            st.rerun()
    else:
        approver = st.text_input("재개 승인자(ROLE-03, 전환자와 달라야 함) 가명")
        basis = st.text_input("재개 근거 (재평가 결과 EVD 번호 등)")
        if st.button("재개 승인", disabled=not (approver and basis and approver != state.get("by"))):
            set_state({"state": "정상"})
            write_log(cfg, "LOG-04", {"event": "재개", "why": basis, "by": approver,
                                      "state_before": "안전 모드", "state_after": "정상"})
            st.rerun()

    st.subheader("복원 기록 전수 검토 (HO-03)")
    for r in restores:
        with st.container(border=True):
            st.write(f"{r['ts']} · {r['creator']} · {r['article_id']} · 사유: {r['reason']}")
            res = st.selectbox("판단", ["유지", "등급 재조정 권고"], key=f"aud-{r['ts']}-{r['article_id']}")
            if st.button("점검 기록", key=f"audb-{r['ts']}-{r['article_id']}"):
                write_log(cfg, "LOG-03", {"kind": "복원", "article_id": r["article_id"], "creator": r["creator"],
                                          "verdict": res, "notice": T("TLD-05-draft", article=r["article_id"], result=res)})

    st.subheader("높음 등급 무작위 표본 감사 (주 5%, 최소 40건)")
    high_rows = list({r["article_id"]: r for r in l1 if r["grade"] == HIGH}.values())  # 같은 기사 재판정은 최신 1건
    k = min(len(high_rows), max(40, round(len(high_rows) * 0.05)))
    sample = random.Random(20261009).sample(high_rows, k) if high_rows else []
    done = {r["article_id"] for r in audited}
    for r in sample:
        if r["article_id"] in done:
            continue
        cols = st.columns([4, 1, 1])
        cols[0].write(f"{r['article_id']} · {r['category']} · c={r['c'] and round(r['c'], 3)}")
        for col, verdict in ((cols[1], "정상"), (cols[2], "낚시")):
            if col.button(verdict, key=f"hs-{r['article_id']}-{verdict}"):
                write_log(cfg, "LOG-03", {"kind": "높음 표본", "article_id": r["article_id"], "verdict": verdict})
                st.rerun()


def page_settings() -> None:
    st.header("실험 설정 (TEST-55)")
    ss["condition"] = st.radio("확인 카드 조건", ["B 근거 먼저", "A 등급 먼저"],
                               index=0 if ss["condition"].startswith("B") else 1)
    st.write(f"θ = {cfg['rule']['theta']}, 온도 T = {t_cal:.3f}, 구성 {cfg['version']} / {cfg['model']['precision']}")
    st.caption(T("TLD-01-자세히"))


{"리서치 결과": page_results, "요약": page_summary, "운영자 콘솔": page_operator, "실험 설정": page_settings}[page]()
