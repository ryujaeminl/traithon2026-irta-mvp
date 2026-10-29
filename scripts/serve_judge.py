"""판정 서버 — GPU가 있는 곳(노트북·대회 GPU)에서 띄우고, Streamlit Cloud 화면이 버튼으로 호출한다.

판정은 수십 초~수 분 걸려 터널(약 100초 제한)이 끊으므로 '접수 → 조회' 방식이다.
POST /judge   {"feeds": [...], "top_n": 10, "query": "구글 과징금"} → 202 {"job_id": ...}
GET  /jobs/<id> → {"status": "running|done|error", "stage": ..., "done": 3, "total": 10, "result": {...}}
GET  /health  → {"ok": true, "cfg": ..., "precision": ...}
모든 /judge·/jobs 요청은 헤더 Authorization: Bearer <IRTA_API_TOKEN>가 필요하다.

보안: 토큰 불일치 401, 언론사는 FEEDS 안에서만, top_n 1~30, 검색어 100자 이하. 판정은 한 번에 1건(GPU 1장).
기본은 127.0.0.1에만 열리고, 밖에서 쓰려면 터널로 이 포트만 노출한다.
usage: set IRTA_API_TOKEN=<24자 이상 무작위>  →  python scripts/serve_judge.py [--port 8765]
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irta.data import load_cfg  # noqa: E402
from irta.model import AXModel  # noqa: E402
from irta.news import FEEDS  # noqa: E402
from irta.pipeline import collect, judge_and_save, new_request_id, public_row  # noqa: E402

MAX_TOP_N = 30
MAX_BODY = 4096
MAX_QUERY = 100
JOB_KEEP_SEC = 3600


class Handler(BaseHTTPRequestHandler):
    cfg: dict = {}
    model = None
    token = ""
    gpu = threading.Lock()
    jobs: dict[str, dict] = {}

    def log_message(self, fmt, *args):  # 요청 줄만 간단히(토큰은 헤더라 찍히지 않음)
        sys.stderr.write("%s %s\n" % (time.strftime("%H:%M:%S"), fmt % args))

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self) -> bool:
        got = self.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        return bool(self.token) and hmac.compare_digest(got, self.token)

    def do_GET(self) -> None:
        if self.path == "/health":
            return self._send(200, {"ok": True, "cfg": self.cfg["version"], "precision": self.cfg["model"]["precision"]})
        if self.path.startswith("/jobs/"):
            if not self._authorized():
                return self._send(401, {"error": "unauthorized"})
            job = self.jobs.get(self.path.removeprefix("/jobs/"))
            return self._send(200, job) if job else self._send(404, {"error": "no such job"})
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/judge":
            return self._send(404, {"error": "not found"})
        if not self._authorized():
            return self._send(401, {"error": "unauthorized"})
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            return self._send(400, {"error": "bad body"})
        try:
            req = json.loads(self.rfile.read(n))
            feeds = [f for f in req.get("feeds", []) if f in FEEDS]
            top_n = int(req.get("top_n", 10))
            query = str(req.get("query", ""))[:MAX_QUERY]
        except (ValueError, TypeError, AttributeError):
            return self._send(400, {"error": "bad json"})
        if not feeds or not 1 <= top_n <= MAX_TOP_N:
            return self._send(400, {"error": f"feeds는 {list(FEEDS)} 중에서, top_n은 1~{MAX_TOP_N}"})
        if any(j["status"] == "running" for j in self.jobs.values()):
            return self._send(429, {"error": "다른 판정이 진행 중입니다. 잠시 후 다시 시도하세요."})
        self._gc()
        jid = uuid.uuid4().hex
        self.jobs[jid] = {"status": "running", "stage": "기사 수집", "done": 0, "total": top_n, "started": time.time()}
        threading.Thread(target=self._run, args=(jid, feeds, top_n, query), daemon=True).start()
        self._send(202, {"job_id": jid})

    @classmethod
    def _gc(cls) -> None:
        now = time.time()
        for k in [k for k, j in cls.jobs.items() if j["status"] != "running" and now - j["started"] > JOB_KEEP_SEC]:
            del cls.jobs[k]

    @classmethod
    def _run(cls, jid: str, feeds: list[str], top_n: int, query: str) -> None:
        job = cls.jobs[jid]
        try:
            arts = collect(feeds, top_n, query)
            job.update(stage="A.X 판정", total=len(arts))

            def prog(i: int, n: int) -> None:
                job["done"] = i

            with cls.gpu:
                rid = new_request_id()
                rows = judge_and_save(cls.model, cls.cfg, [(a, "rss", None) for a in arts], rid, query=query,
                                      progress=prog)
            job.update(status="done", stage="완료", result={"request_id": rid, "rows": [public_row(r) for r in rows]})
        except Exception as exc:  # 서버는 계속 살아 있어야 한다
            job.update(status="error", stage=f"판정 실패: {type(exc).__name__}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    token = os.environ.get("IRTA_API_TOKEN", "")
    if len(token) < 24:
        raise SystemExit("IRTA_API_TOKEN 환경변수에 24자 이상의 무작위 문자열을 넣으세요.")
    Handler.cfg = load_cfg()
    Handler.model = AXModel(Handler.cfg)
    Handler.token = token
    print(f"판정 서버 http://{args.host}:{args.port}  ({Handler.cfg['version']} / {Handler.cfg['model']['precision']})",
          flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
