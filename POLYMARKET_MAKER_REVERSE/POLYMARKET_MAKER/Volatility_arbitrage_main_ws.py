# Volatility_arbitrage_main_ws.py
# -*- coding: utf-8 -*-
"""
最小 WS 连接器（只负责连接与订阅，不做格式化/节流/查询展示）。
外部可传入 on_event 回调来处理每条事件。支持 verbose 开关（默认关闭，不输出）。

用法：
  from Volatility_arbitrage_main_ws import ws_watch_by_ids
  ws_watch_by_ids([YES_id, NO_id], label="...", on_event=handler, verbose=False)

依赖：pip install websocket-client
"""
from __future__ import annotations

import json, time, threading, ssl
from typing import Callable, List, Optional, Any, Dict

try:
    import websocket  # websocket-client
except Exception:
    raise RuntimeError("缺少依赖，请先安装： pip install websocket-client")

WS_BASE = "wss://ws-subscriptions-clob.polymarket.com"
CHANNEL = "market"

_REST_RATE_LIMIT_SEC = 1.0
_last_rest_call_ts = 0.0


def _enforce_rest_rate_limit() -> None:
    global _last_rest_call_ts
    now = time.monotonic()
    elapsed = now - _last_rest_call_ts
    remaining = _REST_RATE_LIMIT_SEC - elapsed
    if remaining > 0:
        time.sleep(remaining)
    _last_rest_call_ts = time.monotonic()

def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%H:%M:%S")

def ws_watch_by_ids(
    asset_ids: List[str],
    label: str = "",
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    *,
    on_state: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    verbose: bool = False,
    stop_event: Optional[threading.Event] = None,
):
    """
    只负责：连接 → 订阅 → 将 WS 事件回调给 on_event（逐条 dict）。
    - asset_ids: 订阅的 token_ids（字符串）
    - label: 可选，仅用于启动打印（不参与逻辑）
    - on_event: 回调函数，参数是一条事件（dict）。若服务端下发 list，将按条回调。
    - on_state: 连接状态通知，state in {open, closed, error, silence}
    - verbose: 默认 False。为 True 时打印 OPEN/SUB/ERROR/CLOSED 及无回调时的事件。
    """
    ids = [str(x) for x in asset_ids if x]
    if not ids:
        raise ValueError("asset_ids 为空")

    if verbose and label:
        print(f"[INIT] 订阅: {label}")
    if verbose:
        for i, tid in enumerate(ids):
            print(f"  - token_id[{i}] = {tid}")

    stop_event = stop_event or threading.Event()

    reconnect_delay = 1
    max_reconnect_delay = 60
    silence_timeout = 600  # 秒，超过则主动重连以避免卡死

    headers = [
        "Origin: https://polymarket.com",
        "User-Agent: Mozilla/5.0",
    ]

    while not stop_event.is_set():
        ping_stop = {"v": False}
        silence_guard_stop = {"v": False}

        def _notify(state: str, info: Optional[Dict[str, Any]] = None) -> None:
            if on_state is None:
                return
            payload = info or {}
            try:
                on_state(state, payload)
            except Exception:
                pass

        last_event_ts = time.monotonic()

        def on_open(ws):
            nonlocal reconnect_delay, last_event_ts
            if verbose:
                print(f"[{_now()}][WS][OPEN] -> {WS_BASE+'/ws/'+CHANNEL}")
            payload = {"type": CHANNEL, "assets_ids": ids}
            ws.send(json.dumps(payload))
            reconnect_delay = 1
            last_event_ts = time.monotonic()
            _notify("open", {"label": label, "asset_ids": ids})

            # 文本心跳 PING（与底层 ping 帧并行存在）
            def _ping():
                while not ping_stop["v"] and not stop_event.is_set():
                    try:
                        ws.send("PING")
                        time.sleep(10)
                    except Exception:
                        break

            threading.Thread(target=_ping, daemon=True).start()

            def _silence_guard():
                while not silence_guard_stop["v"] and not stop_event.is_set():
                    time.sleep(5)
                    if stop_event.is_set() or silence_guard_stop["v"]:
                        break
                    if time.monotonic() - last_event_ts < silence_timeout:
                        continue
                    if verbose:
                        print(
                            f"[{_now()}][WS][SILENCE] {label or ids} {silence_timeout}s 无消息，主动重连。"
                        )
                    _notify(
                        "silence",
                        {"label": label, "asset_ids": ids, "timeout": silence_timeout},
                    )
                    try:
                        ws.close()
                    except Exception:
                        pass
                    break

            threading.Thread(target=_silence_guard, daemon=True).start()

        def on_message(ws, message):
            nonlocal last_event_ts
            # 忽略非 JSON 文本（如 PONG）
            try:
                data = json.loads(message)
            except Exception:
                return

            last_event_ts = time.monotonic()

            # 无回调：仅在 verbose=True 时打印，否则静默
            if on_event is None:
                if verbose:
                    print(f"[{_now()}][WS][EVENT] {data}")
                return

            # 逐条回调
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        try:
                            on_event(item)
                        except Exception:
                            pass
            elif isinstance(data, dict):
                try:
                    on_event(data)
                except Exception:
                    pass

        def on_error(ws, error):
            if verbose:
                print(f"[{_now()}][WS][ERROR] {error}")
            _notify("error", {"label": label, "error": str(error)})

        def on_close(ws, status_code, msg):
            ping_stop["v"] = True
            silence_guard_stop["v"] = True
            if verbose:
                print(f"[{_now()}][WS][CLOSED] {status_code} {msg}")
            _notify(
                "closed",
                {"label": label, "status_code": status_code, "message": msg},
            )

        wsa = websocket.WebSocketApp(
            WS_BASE + "/ws/" + CHANNEL,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            header=headers,
        )

        try:
            wsa.run_forever(
                sslopt={"cert_reqs": ssl.CERT_REQUIRED},
                ping_interval=25,
                ping_timeout=10,
            )
        except Exception as exc:
            ping_stop["v"] = True
            silence_guard_stop["v"] = True
            if verbose:
                print(f"[{_now()}][WS][EXCEPTION] {exc}")
            _notify("error", {"label": label, "error": str(exc)})
        finally:
            ping_stop["v"] = True
            silence_guard_stop["v"] = True

        if stop_event.is_set():
            break

        if verbose:
            print(f"[{_now()}][WS] 连接结束，{reconnect_delay}s 后重试…")
        time.sleep(reconnect_delay)
        reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

# --- 仅供独立运行调试 ---
def _parse_cli(argv: List[str]) -> Optional[str]:
    for i, a in enumerate(argv):
        if a == "--source" and i + 1 < len(argv):
            return argv[i + 1].strip()
        if a.startswith("--source="):
            return a.split("=", 1)[1].strip()
    return None

def _resolve_ids_via_rest(source: str):
    import urllib.parse, requests, json
    GAMMA_API = "https://gamma-api.polymarket.com/markets"

    def _is_url(s: str) -> bool:
        return s.startswith("http://") or s.startswith("https://")

    def _extract_market_slug(url: str):
        p = urllib.parse.urlparse(url)
        parts = [x for x in p.path.split("/") if x]
        if len(parts) >= 2 and parts[0] == "event":
            return parts[-1]
        if len(parts) >= 2 and parts[0] == "market":
            return parts[1]
        return None

    if _is_url(source):
        slug = _extract_market_slug(source)
        if not slug:
            raise ValueError("无法从 URL 解析出 market slug")
        _enforce_rest_rate_limit()
        r = requests.get(GAMMA_API, params={"limit": 1, "slug": slug}, timeout=10)
        r.raise_for_status()
        arr = r.json()
        if not (isinstance(arr, list) and arr):
            raise ValueError("gamma-api 未找到该市场")
        m = arr[0]
        title = m.get("question") or slug
        token_ids_raw = m.get("clobTokenIds", "[]")
        token_ids = json.loads(token_ids_raw) if isinstance(token_ids_raw, str) else (token_ids_raw or [])
        return [x for x in token_ids if x], title

    if "," in source:
        a, b = [x.strip() for x in source.split(",", 1)]
        title = "manual-token-ids"
        return [x for x in (a, b) if x], title

    raise ValueError("未识别的输入。")

if __name__ == "__main__":
    import sys
    src = _parse_cli(sys.argv[1:])
    if not src:
        print('请输入 Polymarket 市场 URL：')
        src = input().strip()
        if not src:
            raise SystemExit(1)
    ids, label = _resolve_ids_via_rest(src)

    # 独立运行调试：开启 verbose 以便观察
    def _dbg(ev): print(ev)
    ws_watch_by_ids(ids, label=label, on_event=_dbg, verbose=True)
