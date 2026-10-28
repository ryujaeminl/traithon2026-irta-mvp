"""판정 서버 — GPU가 있는 곳(노트북·대회 GPU)에서 띄우고, Streamlit Cloud 화면이 버튼으로 호출한다.

POST /judge  {"feeds": ["연합뉴스", ...], "top_n": 10}  헤더 Authorization: Bearer <IRTA_API_TOKEN>
  → {"request_id": ..., "rows": [공개용으로 줄인 묶음 행 ...]}   (근거·무관 문장만 남김, raw 출력 제외)
GET  /health → {"ok": true, "cfg": ..., "precision": ...}

보안: 토큰이 없거나 틀리면 401. 언론사는 FEEDS 목록 안에서만, top_n은 1~30만 받는다. 한 번에 한 요청만 처리한다.
기본은 127.0.0.1에만 열리고, 밖에서 쓰려면 터널(cloudflared 등)로 이 포트만 노출한다.
usage: set IRTA_API_TOKEN=<긴 무작위 문자열>  →  python scripts/serve_judge.py [--port 8765]
"""
from __future__ import annotations

import argparse
import hmac
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from irta.data import load_cfg  # noqa: E402
from irta.model import AXModel  # noqa: E402
from irta.news import FEEDS  # noqa: E402
from irta.pipeline import collect, judge_and_save, new_request_id, public_row  # noqa: E402

MAX_TOP_N = 30
MAX_BODY = 4096


class Handler(BaseHTTPRequestHandler):
    cfg: dict = {}
    model = None
    token = ""
    lock = threading.Lock()

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
            self._send(200, {"ok": True, "cfg": self.cfg["version"], "precision": self.cfg["model"]["precision"]})
        else:
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
        except (ValueError, TypeError, AttributeError):
            return self._send(400, {"error": "bad json"})
        if not feeds or not 1 <= top_n <= MAX_TOP_N:
            return self._send(400, {"error": f"feeds는 {list(FEEDS)} 중에서, top_n은 1~{MAX_TOP_N}"})
        if not self.lock.acquire(blocking=False):
            return self._send(429, {"error": "다른 판정이 진행 중입니다. 잠시 후 다시 시도하세요."})
        try:
            rid = new_request_id()
            rows = judge_and_save(self.model, self.cfg, [(a, "rss", None) for a in collect(feeds, top_n)], rid)
            self._send(200, {"request_id": rid, "rows": [public_row(r) for r in rows]})
        except Exception as exc:  # 서버는 계속 살아 있어야 한다
            self._send(500, {"error": f"판정 실패: {type(exc).__name__}"})
        finally:
            self.lock.release()


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
    print(f"판정 서버 http://{args.host}:{args.port}  ({Handler.cfg['version']} / {Handler.cfg['model']['precision']})")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
