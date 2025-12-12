#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customize_fliter.py  ·  基于 Volatility_fliter_EOA.py 的 REST-only 极简版
- 仅用 REST /books 批量获取买一/卖一（bestBid/bestAsk），完全移除 WS 逻辑
- 保留：时间切片（突破500）、早筛后回补、流式逐个输出/详细块、诊断样本等
- 新增：高亮参数（HIGHLIGHT_*）支持命令行自定义
"""

from __future__ import annotations

import argparse
import datetime as dt
import math
import time
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import requests
except Exception as e:
    print("[FATAL] 缺少 requests，请先 pip install requests", file=sys.stderr)
    sys.exit(
        "[EXIT] 已终止：未安装 requests，安装方法示例：pip install requests"
    )

# ===============================
# 参数集中区（集中管理默认口径）
# ===============================
# 宽口径（通过筛选）与 argparse 默认值的对齐建议：把下面四项与 argparse 默认保持一致
DEFAULT_MIN_END_HOURS: float = 1.0
DEFAULT_MAX_END_DAYS: int   = 2
DEFAULT_GAMMA_WINDOW_DAYS: int = 2
DEFAULT_GAMMA_MIN_WINDOW_HOURS: int = 1
DEFAULT_LEGACY_END_DAYS: int  = 730

# 高亮（严格口径）集中参数
HIGHLIGHT_MAX_HOURS: float   = 48.0
HIGHLIGHT_ASK_MIN: float     = 0.96
HIGHLIGHT_ASK_MAX: float     = 0.995
HIGHLIGHT_MIN_TOTAL_VOLUME: float = 10000.0  # 总交易量≥此值（USDC）
HIGHLIGHT_MAX_ASK_DIFF: float = 0.10         # 同一 token 点差 |ask - bid| ≤ 此阈值（YES 或 NO 任一侧满足即可）

# 价格反转检测默认参数
REVERSAL_ENABLED: bool = True
REVERSAL_P1: float = 0.35                 # 旧段最高价上限
REVERSAL_P2: float = 0.80                 # 近段最高价下限
REVERSAL_WINDOW_HOURS: float = 2.0        # 近段窗口（小时）
REVERSAL_LOOKBACK_DAYS: float = 5.0       # 旧段回溯天数
REVERSAL_SHORT_INTERVAL: str = "6h"      # 短窗口 interval 触发
REVERSAL_SHORT_FIDELITY: int = 15         # 短窗口 fidelity
REVERSAL_LONG_FIDELITY: int = 60          # 长窗口 fidelity


# -------------------------------
# 小工具
# -------------------------------

def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)

def _parse_dt(s: Optional[str]) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        if s.endswith('Z'):
            s = s[:-1] + '+00:00'
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None

def _coerce_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        if isinstance(x, str):
            x2 = x.replace(',', '').strip()
            if x2 == '':
                return None
            return float(x2)
    except Exception:
        return None
    return None

def _coerce_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ('true', 'yes', 'y', '1'):
            return True
        if s in ('false', 'no', 'n', '0'):
            return False
    if isinstance(x, (int, float)):
        return bool(x)
    return None

def _fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "-"
    try:
        return f"{x:,.2f}"
    except Exception:
        return str(x)

def _hours_until(t: Optional[dt.datetime]) -> Optional[float]:
    if not t:
        return None
    delta = (t - _now_utc()).total_seconds() / 3600.0
    return round(delta, 1)

def _infer_binary_from_raw(raw: Dict[str, Any]) -> bool:
    if not isinstance(raw, dict):
        return False
    op = raw.get("outcomePrices")
    if isinstance(op, list) and len(op) == 2:
        return True
    for k in ("outcomes", "contracts"):
        v = raw.get(k)
        if isinstance(v, list) and len(v) == 2:
            return True
    for k in ("binary", "isBinary"):
        bv = raw.get(k)
        if isinstance(bv, bool) and bv:
            return True
        if isinstance(bv, str) and bv.lower() in ("true","yes","y","1"):
            return True
    return False

# -------------------------------
# 数据结构
# -------------------------------

@dataclass
class OutcomeSnapshot:
    name: str
    token_id: Optional[str] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None


@dataclass
class MarketSnapshot:
    slug: str
    title: str
    raw: Dict[str, Any] = field(default_factory=dict)
    event_id: Optional[str] = None
    yes: OutcomeSnapshot = field(default_factory=lambda: OutcomeSnapshot(name='YES'))
    no: OutcomeSnapshot = field(default_factory=lambda: OutcomeSnapshot(name='NO'))
    liquidity: Optional[float] = None
    volume24h: Optional[float] = None
    totalVolume: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    active: Optional[bool] = None
    closed: Optional[bool] = None
    resolved: Optional[bool] = None
    acceptingOrders: Optional[bool] = None
    end_time: Optional[dt.datetime] = None
    reversal_hit: bool = False
    reversal_side: Optional[str] = None
    reversal_detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HighlightedOutcome:
    """被严格筛选条件命中的市场-方向组合。"""

    market: MarketSnapshot
    outcome: OutcomeSnapshot
    hours_to_end: float


@dataclass
class FilterResult:
    """封装供自动化脚本复用的筛选结果。"""

    total_markets: int
    candidates: List[MarketSnapshot]
    chosen: List[MarketSnapshot]
    rejected: List[Tuple[MarketSnapshot, str]]
    highlights: List[HighlightedOutcome]
    merged_event_count: int = 0
    missing_event_id_count: int = 0

# -------------------------------
# Gamma 抓取（时间切片 · 突破500）
# -------------------------------

_GAMMA_HOST = os.environ.get("GAMMA_HOST", "https://gamma-api.polymarket.com")

def _gamma_fetch(params: Dict[str, str]) -> List[Dict[str, Any]]:
    url = f"{_GAMMA_HOST}/markets"
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "data" in data:
            return data["data"]
        return []
    except Exception:
        return []

def fetch_markets_windowed(
    end_min: dt.datetime,
    end_max: dt.datetime,
    *,
    window_days: int = 14,
    min_window_hours: int = DEFAULT_GAMMA_MIN_WINDOW_HOURS,
) -> List[Dict[str, Any]]:
    all_mkts: List[Dict[str, Any]] = []
    seen: set = set()
    one_sec = dt.timedelta(seconds=1)
    min_window = dt.timedelta(hours=max(1, int(min_window_hours)))

    def _process_interval(start: dt.datetime, end: dt.datetime) -> None:
        params = {
            "limit": "500",
            "order": "endDate",
            "ascending": "true",
            "active": "true",
            "closed": "false",
            "end_date_min": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "end_date_max": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        chunk = _gamma_fetch(params)

        for m in chunk:
            mid = m.get("id") or m.get("slug")
            if mid and mid not in seen:
                seen.add(mid)
                all_mkts.append(m)

        if not chunk:
            return

        duration = end - start
        hit_limit = len(chunk) >= 500

        if hit_limit and duration > min_window:
            mid_point = start + dt.timedelta(seconds=duration.total_seconds() / 2)
            left_end = min(mid_point, end)
            right_start = left_end + one_sec
            if start < left_end:
                _process_interval(start, left_end)
            if right_start <= end:
                _process_interval(right_start, end)
            return

        if hit_limit:
            last_end = _parse_dt(chunk[-1].get("endDate") or chunk[-1].get("end_time") or chunk[-1].get("endTime"))
            if last_end is not None:
                next_start = last_end + one_sec
                if next_start <= end:
                    _process_interval(next_start, end)

    cur = end_min
    while cur <= end_max:
        sub_end = min(cur + dt.timedelta(days=window_days), end_max)
        _process_interval(cur, sub_end)
        cur = sub_end + one_sec

    return all_mkts

# -------------------------------
# 解析 + 旧格式检测
# -------------------------------

def _is_arch_legacy_nonclob(raw: Dict[str, Any], legacy_end_days: int) -> bool:
    title = (raw.get("question") or raw.get("title") or "").strip()
    slug  = (raw.get("slug") or "").strip()
    end   = _parse_dt(raw.get("endDate") or raw.get("end_time") or raw.get("endTime"))
    clob_ids = raw.get("clobTokenIds") or raw.get("clob_token_ids") or raw.get("clobTokens")

    if title.upper().startswith("ARCH") or slug.lower().startswith("arch"):
        return True
    if not clob_ids:
        return True
    if end is not None and legacy_end_days and legacy_end_days > 0:
        try:
            hours = _hours_until(end)
            if hours is not None and hours < -24.0 * float(legacy_end_days):
                return True
        except Exception:
            pass
    return False

def _extract_event_id(raw: Dict[str, Any]) -> Optional[str]:
    """尽可能宽松地提取事件标识，用于同事件只选一个市场的去重。

    部分行情接口会将事件信息放在二级字段（如 event: {id/slug/...}），
    或使用不同的命名风格。这里统一兜底解析，避免因为缺少 event_id
    导致同一事件的多个市场同时通过筛选。
    """

    def _stringify(v: Any) -> Optional[str]:
        if v is None:
            return None
        try:
            s = str(v).strip()
            return s or None
        except Exception:
            return None

    def _from_list(obj: Any) -> Optional[Dict[str, Any]]:
        if isinstance(obj, list) and obj:
            first = obj[0]
            if isinstance(first, dict):
                return first
        return None

    def _extract_from_mapping(mapping: Dict[str, Any]) -> Optional[str]:
        for k in (
            "id",
            "eventId",
            "slug",
            "eventSlug",
            "title",
            "eventTitle",
            "name",
            "eventName",
        ):
            v = _stringify(mapping.get(k))
            if v:
                return v
        return None

    # Gamma /markets 会把事件信息放进 events/templates 列表里，优先从这里取
    events_obj = _from_list(raw.get("events"))
    if events_obj:
        v = _extract_from_mapping(events_obj)
        if v:
            return v

    templates_obj = _from_list(raw.get("templates"))
    if templates_obj:
        v = _extract_from_mapping(templates_obj)
        if v:
            return v

    # 直接暴露在顶层的 eventId/slug 等
    direct_keys = (
        "eventId",
        "event_id",
        "eventID",
        "eventSlug",
        "event_slug",
        "eventTitle",
        "event_title",
        "eventName",
        "event_name",
    )
    for k in direct_keys:
        v = _stringify(raw.get(k))
        if v:
            return v

    nested = raw.get("event") or raw.get("eventInfo") or raw.get("event_info")
    if isinstance(nested, dict):
        for k in ("id", "eventId", "slug", "eventSlug", "title", "name"):
            v = _stringify(nested.get(k))
            if v:
                return v
    elif isinstance(nested, str):
        v = _stringify(nested)
        if v:
            return v

    # 部分接口直接在顶层放字符串事件名
    v = _stringify(raw.get("event"))
    if v:
        return v

    return None

def _parse_market(raw: Dict[str, Any]) -> MarketSnapshot:
    title = raw.get("question") or raw.get("title") or ""
    slug  = raw.get("slug") or ""
    ms = MarketSnapshot(slug=slug, title=title, raw=raw)

    ms.event_id = _extract_event_id(raw)
    ms.active = _coerce_bool(raw.get("active"))
    ms.closed = _coerce_bool(raw.get("closed"))
    ms.resolved = _coerce_bool(raw.get("resolved"))
    ms.acceptingOrders = _coerce_bool(raw.get("acceptingOrders"))
    ms.end_time = _parse_dt(raw.get("endDate") or raw.get("end_time") or raw.get("endTime"))

    ms.liquidity = _coerce_float(raw.get("liquidity") or raw.get("liquidity_num") or raw.get("liquidityNum") or raw.get("liquidityUsd") or raw.get("totalLiquidity"))
    ms.volume24h = _coerce_float(raw.get("volume24h") or raw.get("volume24Hr") or raw.get("volume24Hour") or raw.get("volume_24h") or raw.get("lastDayVolume"))
    ms.totalVolume = _coerce_float(raw.get("totalVolume") or raw.get("volume") or raw.get("volume_num") or raw.get("volumeNum"))

    tags = raw.get("tags") or raw.get("tagNames") or raw.get("categories") or []
    if isinstance(tags, list):
        ms.tags = [str(t) for t in tags]
    elif isinstance(tags, str):
        ms.tags = [tags]

    clob_ids = raw.get("clobTokenIds") or raw.get("clob_token_ids") or raw.get("clobTokens")
    if isinstance(clob_ids, str):
        try:
            import json as _json
            clob_ids = _json.loads(clob_ids)
        except Exception:
            clob_ids = None
    if isinstance(clob_ids, list) and len(clob_ids) >= 2:
        try:
            ms.yes.token_id = str(clob_ids[0])
            ms.no.token_id  = str(clob_ids[1])
        except Exception:
            pass

    return ms

# -------------------------------
# 早筛（不拉价格，先确定是否需要回补）
# -------------------------------

def _is_binary(ms: MarketSnapshot) -> bool:
    return bool(ms.yes.token_id and ms.no.token_id)

def _early_filter_reason(ms: MarketSnapshot, min_end_hours: float, legacy_end_days: int) -> Tuple[bool, str]:
    if _is_arch_legacy_nonclob(ms.raw, legacy_end_days):
        if not _is_binary(ms) and _infer_binary_from_raw(ms.raw):
            return False, "二元（旧格式；缺 clobTokenIds）"
        return False, "归档/旧格式（非 CLOB）"
    if not _is_binary(ms):
        if _infer_binary_from_raw(ms.raw):
            return False, "二元（旧格式；缺 clobTokenIds）"
        return False, "非二元市场"
    if min_end_hours is not None and min_end_hours > 0:
        h = _hours_until(ms.end_time)
        if h is None or h < min_end_hours:
            return False, f"剩余时间不足（{h}h）"
    return True, "候选（待回补报价）"

# -------------------------------
# REST /books 批量回补（直接取买一/卖一）
# -------------------------------

_POLY_HOST = os.environ.get("POLY_HOST", "https://clob.polymarket.com").rstrip("/")

def _rest_books_backfill(
    candidates: List[MarketSnapshot],
    batch_size: int = 200,
    timeout: float = 10.0,
) -> None:
    # 仅对仍缺买卖价的 token 做回补（任一侧有价即可跳过）
    missing: List[str] = []
    index: Dict[str, Tuple[MarketSnapshot, str]] = {}
    seen = set()

    for ms in candidates:
        for side, snap in (('YES', ms.yes), ('NO', ms.no)):
            tid = snap.token_id
            if not tid:
                continue
            if (snap.bid is None and snap.ask is None) and tid not in seen:
                seen.add(tid)
                missing.append(tid)
                index[tid] = (ms, side)

    if not missing:
        return

    url = f"{_POLY_HOST}/books"
    headers = {"Content-Type": "application/json"}

    def best_from_levels(levels: List[Dict[str, Any]], is_bid: bool) -> Optional[float]:
        if not isinstance(levels, list) or not levels:
            return None
        prices = []
        for lv in levels:
            p = _coerce_float((lv or {}).get("price"))
            if p is not None:
                prices.append(p)
        if not prices:
            return None
        return (max(prices) if is_bid else min(prices))

    for i in range(0, len(missing), batch_size):
        batch_idx = i // batch_size + 1
        total_batches = math.ceil(len(missing) / batch_size) if batch_size else 0
        batch = missing[i:i+batch_size]
        body = [{"token_id": tid} for tid in batch]
        print(
            f"[HEARTBEAT] /books 回补进度：批次 {batch_idx}/{total_batches}，token 数={len(batch)}",
            flush=True,
        )
        try:
            r = requests.post(url, json=body, headers=headers, timeout=timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            print(f"[WARN] REST /books 回补失败：{e}", file=sys.stderr)
            continue

        if not isinstance(data, list):
            continue
        for ob in data:
            try:
                tid = str(ob.get("asset_id") or ob.get("token_id") or "")
                if not tid or tid not in index:
                    continue
                ms, side = index[tid]
                bids = ob.get("bids") or []
                asks = ob.get("asks") or []
                bb = best_from_levels(bids, is_bid=True)
                aa = best_from_levels(asks, is_bid=False)
                if side == 'YES':
                    if ms.yes.bid is None and bb is not None: ms.yes.bid = bb
                    if ms.yes.ask is None and aa is not None: ms.yes.ask = aa
                else:
                    if ms.no.bid is None and bb is not None: ms.no.bid = bb
                    if ms.no.ask is None and aa is not None: ms.no.ask = aa
            except Exception:
                continue

# -------------------------------
# 价格历史与反转检测
# -------------------------------

def _ts_from_dt(value: dt.datetime) -> float:
    try:
        return value.timestamp()
    except Exception:
        return 0.0


def _normalize_ts(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            v = float(value)
            if v > 1e12:  # 毫秒时间戳
                v = v / 1000.0
            return v
        if isinstance(value, str):
            v = value.strip()
            if not v:
                return None
            try:
                num = float(v)
                if num > 1e12:
                    num = num / 1000.0
                return num
            except Exception:
                dt_obj = _parse_dt(v)
                if dt_obj:
                    return _ts_from_dt(dt_obj)
                return None
    except Exception:
        return None
    return None


def _extract_price(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except Exception:
            return None
    return None


def _price_points_from_history(raw: Any) -> List[Tuple[float, float]]:
    data = raw
    if isinstance(raw, dict):
        for key in ("prices", "data", "items", "result"):
            if isinstance(raw.get(key), list):
                data = raw.get(key)
                break

    if not isinstance(data, list):
        return []

    points: List[Tuple[float, float]] = []
    for item in data:
        ts: Optional[float] = None
        price: Optional[float] = None

        if isinstance(item, dict):
            for tk in ("timestamp", "time", "ts", "t", "date"):
                ts = _normalize_ts(item.get(tk))
                if ts is not None:
                    break
            for pk in ("price", "mid", "value", "avg", "close", "last", "p"):
                price = _extract_price(item.get(pk))
                if price is not None:
                    break
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            ts = _normalize_ts(item[0])
            price = _extract_price(item[1])

        if ts is None or price is None:
            continue
        points.append((ts, price))

    points.sort(key=lambda x: x[0])
    return points


def _fetch_prices_history_once(
    token_id: str,
    params: Dict[str, Any],
    *,
    timeout: float = 10.0,
) -> List[Tuple[float, float]]:
    url = f"{_POLY_HOST}/prices-history"
    try:
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return _price_points_from_history(r.json())
    except Exception as e:
        print(f"[WARN] 获取价格历史失败 token={token_id} params={params}: {e}", file=sys.stderr)
        return []


def fetch_prices_history(
    token_id: str,
    *,
    start_ts: Optional[float] = None,
    end_ts: Optional[float] = None,
    interval: Optional[str] = None,
    fidelity: Optional[int] = None,
    timeout: float = 10.0,
) -> List[Tuple[float, float]]:
    params_base: Dict[str, Any] = {"market": token_id}
    if interval:
        params_base["interval"] = interval
    if fidelity is not None:
        params_base["fidelity"] = fidelity

    # 长窗口切片抓取，避免单次请求过大被拒绝
    if start_ts is not None and end_ts is not None:
        slice_seconds = 7 * 86400
        window_start = int(start_ts)
        window_end_all = int(end_ts)
        slice_index = 1
        collected: List[Tuple[float, float]] = []

        while window_start < window_end_all:
            window_end = min(window_start + slice_seconds, window_end_all)
            params = params_base | {"startTs": window_start, "endTs": window_end}
            print(
                f"[HEARTBEAT] 价格切片 {slice_index} / ?：token={token_id}, "
                f"{dt.datetime.utcfromtimestamp(window_start)} -> {dt.datetime.utcfromtimestamp(window_end)}",
                flush=True,
            )
            collected.extend(
                _fetch_prices_history_once(token_id, params, timeout=timeout)
            )
            slice_index += 1
            window_start = window_end
            if window_start < window_end_all:
                time.sleep(0.5)

        return collected

    # 常规请求（无起止时间）
    return _fetch_prices_history_once(token_id, params_base, timeout=timeout)


def _max_price_in_range(points: List[Tuple[float, float]], start_ts: float, end_ts: float) -> Optional[float]:
    vals = [p for ts, p in points if start_ts <= ts <= end_ts]
    if not vals:
        return None
    try:
        return float(max(vals))
    except Exception:
        return None


def detect_reversal(
    token_id: Optional[str],
    *,
    p1: float = REVERSAL_P1,
    p2: float = REVERSAL_P2,
    window_hours: float = REVERSAL_WINDOW_HOURS,
    lookback_days: float = REVERSAL_LOOKBACK_DAYS,
    short_interval: str = REVERSAL_SHORT_INTERVAL,
    short_fidelity: int = REVERSAL_SHORT_FIDELITY,
    long_fidelity: int = REVERSAL_LONG_FIDELITY,
) -> Tuple[bool, Dict[str, Any]]:
    if not token_id:
        return False, {"reason": "缺少 token_id"}

    now = _now_utc()
    now_ts = _ts_from_dt(now)
    recent_start_ts = now_ts - float(window_hours) * 3600.0
    lookback_start_ts = now_ts - float(lookback_days) * 86400.0

    short_points = fetch_prices_history(
        token_id,
        interval=short_interval,
        fidelity=short_fidelity,
    )
    max_recent_short = _max_price_in_range(short_points, recent_start_ts, now_ts)
    if max_recent_short is None or max_recent_short < p2:
        return False, {
            "reason": "短窗口未触发",
            "max_recent_short": max_recent_short,
            "points_short": len(short_points),
        }

    long_points = fetch_prices_history(
        token_id,
        start_ts=lookback_start_ts,
        end_ts=now_ts,
        fidelity=long_fidelity,
    )

    max_recent = _max_price_in_range(long_points, recent_start_ts, now_ts)
    max_old = _max_price_in_range(long_points, lookback_start_ts, recent_start_ts)

    hit = bool(max_recent is not None and max_recent >= p2 and max_old is not None and max_old < p1)
    detail = {
        "points_long": len(long_points),
        "points_short": len(short_points),
        "max_recent": max_recent,
        "max_old": max_old,
        "max_recent_short": max_recent_short,
        "window_hours": window_hours,
        "lookback_days": lookback_days,
        "p1": p1,
        "p2": p2,
    }
    if not hit:
        detail["reason"] = "长窗口未满足反转"
    return hit, detail

# -------------------------------
# 最终筛选（在回补后判断报价）
# -------------------------------

def _final_pass_reason(ms: MarketSnapshot, require_quotes: bool) -> Tuple[bool, str]:
    hit = _blacklist_hit(ms)
    if hit:
        return False, f"命中黑名单：{hit}"
    if require_quotes:
        yes_ok = (ms.yes.bid is not None or ms.yes.ask is not None)
        no_ok  = (ms.no.bid is not None or ms.no.ask is not None)
        if not (yes_ok or no_ok):
            return False, "缺少买卖价（空簿/超时）"
    return True, "OK"

# -------------------------------
# 打印
# -------------------------------

DEFAULT_BLACKLIST_TERMS = [
    "Bitcoin",
    "BTC",
    "ETH",
    "Ethereum",
    "Sol",
    "Solana",
    "Doge",
    "Dogecoin",
    "BNB",
    "Binance",
    "Cardano",
    "ADA",
    "XRP",
    "Ripple",
    "Matic",
    "Polygon",
    "Crypto",
    "Cryptocurrency",
    "Blockchain",
    "Token",
    "NFT",
    "DeFi",
    "vs",
    "odds",
    "score",
    "spread",
    "moneyline",
    "win",
    "Esports",
    "CS2",
    "Cup",
    "Arsenal",
    "Liverpool",
    "Chelsea",
    "EPL",
    "PGA",
    "Tour Championship",
    "Scottie Scheffler",
    "Vitality",
    "MOUZ",
    "Falcons",
    "The MongolZ",
    "AL",
    "Houston",
    "Chicago",
    "New York",
    "Will Elon Musk post",
    "dota2",
    "a dozen eggs",
]

def _build_blacklist_patterns(terms: Iterable[str]) -> List[Tuple[str, re.Pattern[str]]]:
    patterns: List[Tuple[str, re.Pattern[str]]] = []
    for term in terms:
        tl = term.lower()
        if len(tl) <= 3 and tl.isalpha():
            pat = re.compile(rf"\b{re.escape(tl)}\b", re.IGNORECASE)
        else:
            pat = re.compile(re.escape(term), re.IGNORECASE)
        patterns.append((term, pat))
    return patterns

BLACKLIST_TERMS: List[str] = list(DEFAULT_BLACKLIST_TERMS)
BLACKLIST_PATTERNS = _build_blacklist_patterns(BLACKLIST_TERMS)


def set_blacklist_terms(terms: Iterable[str]) -> None:
    """Refresh blacklist terms from a configurable source."""

    global BLACKLIST_TERMS, BLACKLIST_PATTERNS

    cleaned: List[str] = []
    seen = set()
    for term in terms:
        if term is None:
            continue
        t = str(term).strip()
        if not t or t.lower() in seen:
            continue
        seen.add(t.lower())
        cleaned.append(t)

    BLACKLIST_TERMS = cleaned or list(DEFAULT_BLACKLIST_TERMS)
    BLACKLIST_PATTERNS = _build_blacklist_patterns(BLACKLIST_TERMS)
    print(f"[CONFIG] 黑名单关键词已更新：{len(BLACKLIST_TERMS)} 条。")


def _print_snapshot(idx: int, total: int, ms: MarketSnapshot):
    print(f"[TRACE] [{idx}/{total}] 原始市场：slug={ms.slug} | 标题={ms.title}")
    st = " ".join([
        f"active={'是' if ms.active else '-'}",
        f"resolved={'是' if ms.resolved else '-'}",
        f"closed={'是' if ms.closed else '-'}",
        f"acceptingOrders={'是' if ms.acceptingOrders else '-'}",
    ])
    print(f"[TRACE]   状态：{st}")
    print(f"[TRACE]   金额：liquidity={_fmt_money(ms.liquidity)} volume24h={_fmt_money(ms.volume24h)} totalVolume={_fmt_money(ms.totalVolume)}")
    raw_end = ms.end_time.isoformat() if ms.end_time else "-"
    print(f"[TRACE]   时间：raw_end={raw_end}")
    print(f"[TRACE]   解析结果：")
    print(f"[TRACE]     {ms.slug} | {ms.title}")
    if (ms.yes.token_id is None or ms.no.token_id is None):
        print(f"[TRACE]       [HINT] 未能解析 clobTokenIds（疑似旧格式）。")
    def _fmt_side(s: OutcomeSnapshot) -> str:
        b = "-" if s.bid is None else f"{s.bid:.4f}"
        a = "-" if s.ask is None else f"{s.ask:.4f}"
        return f"{s.name}[{s.token_id}] bid={b} ask={a}"
    print(f"[TRACE]       {_fmt_side(ms.yes)}")
    print(f"[TRACE]       {_fmt_side(ms.no)}")
    h = _hours_until(ms.end_time)
    print(f"[TRACE]       liquidity={_fmt_money(ms.liquidity)}  volume24h={_fmt_money(ms.volume24h)}  ends_in={h}h  tags={','.join(ms.tags) if ms.tags else '-'}")

def _print_singleline(ms: MarketSnapshot, reason: str):
    yb = "-" if ms.yes.bid is None else f"{ms.yes.bid:.4f}"
    ya = "-" if ms.yes.ask is None else f"{ms.yes.ask:.4f}"
    nb = "-" if ms.no.bid is None else f"{ms.no.bid:.4f}"
    na = "-" if ms.no.ask is None else f"{ms.no.ask:.4f}"
    h  = _hours_until(ms.end_time)
    print(f"[RES] {ms.slug} | {ms.title} | YES {yb}/{ya} NO {nb}/{na} | ends_in={h}h | {reason}", flush=True)


def _blacklist_hit(ms: MarketSnapshot) -> Optional[str]:
    parts = [ms.title or "", ms.slug or ""]
    if ms.tags:
        parts.append(" ".join(ms.tags))
    haystack = " ".join(filter(None, parts))
    for term, pat in BLACKLIST_PATTERNS:
        if pat.search(haystack):
            return term
    return None



def _highlight_outcomes(
    ms: MarketSnapshot,
    max_hours: Optional[float] = None,
    min_total_volume: Optional[float] = None,
    max_ask_diff: Optional[float] = None,
    require_reversal: bool = True,
) -> List[Tuple[OutcomeSnapshot, float]]:
    """
    高亮（严格口径）筛选条件：
    - 剩余时间 ≤ max_hours（默认 HIGHLIGHT_MAX_HOURS）
    - 总交易量 totalVolume ≥ min_total_volume（默认 HIGHLIGHT_MIN_TOTAL_VOLUME）
    - 同一 token 的点差 |ask - bid| ≤ max_ask_diff（YES 或 NO 任一满足即可；默认 HIGHLIGHT_MAX_ASK_DIFF）
    - 价格反转命中（默认强制）
    - 不命中黑名单
    """
    mh = HIGHLIGHT_MAX_HOURS if max_hours is None else max_hours
    mv = HIGHLIGHT_MIN_TOTAL_VOLUME if min_total_volume is None else min_total_volume
    mdiff = HIGHLIGHT_MAX_ASK_DIFF if max_ask_diff is None else max_ask_diff

    hours = _hours_until(ms.end_time)
    if hours is None or hours < 0 or hours > mh:
        return []
    if _blacklist_hit(ms):
        return []

    if require_reversal and not ms.reversal_hit:
        return []

    # 交易量要求
    tv = getattr(ms, "totalVolume", None)
    if tv is None:
        try:
            tv = float(ms.raw.get("totalVolume") or ms.raw.get("volume") or 0)
        except Exception:
            tv = 0.0
    if tv < mv:
        return []

    matches: List[Tuple[OutcomeSnapshot, float]] = []
    candidates: List[OutcomeSnapshot] = []
    if ms.reversal_side and ms.reversal_side.upper() == "NO":
        candidates = [ms.no, ms.yes]
    elif ms.reversal_side and ms.reversal_side.upper() == "YES":
        candidates = [ms.yes, ms.no]
    else:
        candidates = [ms.yes, ms.no]

    for snap in candidates:
        spread_ok = True
        if mdiff is not None and snap.bid is not None and snap.ask is not None:
            spread_ok = abs(float(snap.ask) - float(snap.bid)) <= mdiff
        if spread_ok:
            matches.append((snap, hours))
            break
    return matches


def _highlight_label() -> str:
    rev = (
        "反转命中"
        if REVERSAL_ENABLED
        else "(未启用反转)"
    )
    return (
        f"≤{int(HIGHLIGHT_MAX_HOURS)}h & {rev} "
        f"& 总交易量≥{int(HIGHLIGHT_MIN_TOTAL_VOLUME)}USDC "
        f"& 单边点差≤{HIGHLIGHT_MAX_ASK_DIFF:.2f} & 非黑名单"
    )


def _event_key(ms: MarketSnapshot) -> Optional[str]:
    return ms.event_id


def _outcome_price(snap: OutcomeSnapshot) -> float:
    if snap.ask is not None:
        try:
            return float(snap.ask)
        except Exception:
            pass
    if snap.bid is not None:
        try:
            return float(snap.bid)
        except Exception:
            pass
    return -1.0


def _best_outcome(hits: List[Tuple[OutcomeSnapshot, float]]) -> Tuple[OutcomeSnapshot, float]:
    def _dir_rank(name: Optional[str]) -> int:
        return 0 if (name or "").upper() == "NO" else 1

    ranked = sorted(
        hits,
        key=lambda item: (
            _dir_rank(item[0].name),
            -_outcome_price(item[0]),
        ),
    )
    return ranked[0]


def _print_topics_summary(markets: List[MarketSnapshot], stage: str = "粗筛通过") -> None:
    if not markets:
        print(f"[HEARTBEAT] {stage}：当前无候选。", flush=True)
        return

    print(
        f"[HEARTBEAT] {stage}：共 {len(markets)} 个候选，展示话题列表：",
        flush=True,
    )
    max_show = 200
    for idx, ms in enumerate(markets, start=1):
        if idx > max_show:
            print(f"  ... 其余 {len(markets) - max_show} 个已省略", flush=True)
            break
        print(f"  [{idx}] slug={ms.slug} | title={ms.title}", flush=True)
    print("[HEARTBEAT] 粗筛列表打印完毕，准备进入下一阶段。", flush=True)


# -------------------------------
# 面向自动化脚本的封装
# -------------------------------

def collect_filter_results(
    *,
    min_end_hours: float = DEFAULT_MIN_END_HOURS,
    max_end_days: int = DEFAULT_MAX_END_DAYS,
    gamma_window_days: int = DEFAULT_GAMMA_WINDOW_DAYS,
    gamma_min_window_hours: int = DEFAULT_GAMMA_MIN_WINDOW_HOURS,
    legacy_end_days: int = DEFAULT_LEGACY_END_DAYS,
    allow_illiquid: bool = False,
    skip_orderbook: bool = False,
    no_rest_backfill: bool = False,
    books_batch_size: int = 200,
    books_timeout: float = 10.0,
    only: str = "",
    blacklist_terms: Optional[Iterable[str]] = None,
    prefetched_markets: Optional[List[Dict[str, Any]]] = None,
    enable_reversal: bool = REVERSAL_ENABLED,
    reversal_p1: float = REVERSAL_P1,
    reversal_p2: float = REVERSAL_P2,
    reversal_window_hours: float = REVERSAL_WINDOW_HOURS,
    reversal_lookback_days: float = REVERSAL_LOOKBACK_DAYS,
    reversal_short_interval: str = REVERSAL_SHORT_INTERVAL,
    reversal_short_fidelity: int = REVERSAL_SHORT_FIDELITY,
    reversal_long_fidelity: int = REVERSAL_LONG_FIDELITY,
) -> FilterResult:
    """执行一次筛选流程并返回结构化结果。"""

    global REVERSAL_ENABLED, REVERSAL_P1, REVERSAL_P2, REVERSAL_WINDOW_HOURS
    global REVERSAL_LOOKBACK_DAYS, REVERSAL_SHORT_INTERVAL, REVERSAL_SHORT_FIDELITY
    global REVERSAL_LONG_FIDELITY

    REVERSAL_ENABLED = enable_reversal
    REVERSAL_P1 = reversal_p1
    REVERSAL_P2 = reversal_p2
    REVERSAL_WINDOW_HOURS = reversal_window_hours
    REVERSAL_LOOKBACK_DAYS = reversal_lookback_days
    REVERSAL_SHORT_INTERVAL = reversal_short_interval
    REVERSAL_SHORT_FIDELITY = reversal_short_fidelity
    REVERSAL_LONG_FIDELITY = reversal_long_fidelity

    if blacklist_terms is not None:
        set_blacklist_terms(blacklist_terms)

    if prefetched_markets is None:
        now = _now_utc()
        end_min = now + dt.timedelta(hours=min_end_hours)
        end_max = now + dt.timedelta(days=max_end_days)
        mkts_raw = fetch_markets_windowed(
            end_min,
            end_max,
            window_days=gamma_window_days,
            min_window_hours=gamma_min_window_hours,
        )
    else:
        mkts_raw = prefetched_markets

    only_pat = only.lower().strip()

    market_list: List[MarketSnapshot] = []
    early_rejects: List[Tuple[MarketSnapshot, str]] = []

    for raw in mkts_raw:
        title = (raw.get("question") or raw.get("title") or "")
        slug = (raw.get("slug") or "")
        if only_pat and (only_pat not in title.lower() and only_pat not in slug.lower()):
            continue
        ms = _parse_market(raw)
        ok, reason = _early_filter_reason(ms, min_end_hours, legacy_end_days)
        if ok:
            market_list.append(ms)
        else:
            early_rejects.append((ms, reason))

    print(
        f"[HEARTBEAT] 初筛完成：候选 {len(market_list)} / {len(mkts_raw)}，"
        f"被拒 {len(early_rejects)}，开始反转检测/报价回补…",
        flush=True,
    )
    _print_topics_summary(market_list, stage="粗筛通过")

    if enable_reversal and market_list:
        total_markets = len(market_list)
        for idx, ms in enumerate(market_list, start=1):
            token_id = ms.yes.token_id or ms.no.token_id
            side = "YES" if ms.yes.token_id else "NO"
            try:
                hit, detail = detect_reversal(
                    token_id,
                    p1=reversal_p1,
                    p2=reversal_p2,
                    window_hours=reversal_window_hours,
                    lookback_days=reversal_lookback_days,
                    short_interval=reversal_short_interval,
                    short_fidelity=reversal_short_fidelity,
                    long_fidelity=reversal_long_fidelity,
                )
            except Exception as exc:
                hit, detail = False, {"reason": f"反转检测异常: {exc}"}
            ms.reversal_hit = hit
            ms.reversal_side = side
            ms.reversal_detail = detail
            if idx == 1 or idx == total_markets or idx % 50 == 0:
                print(
                    f"[HEARTBEAT] 反转检测进度：{idx}/{total_markets} -> {ms.slug}",
                    flush=True,
                )
    else:
        for ms in market_list:
            ms.reversal_hit = True
            ms.reversal_side = ms.reversal_side or "YES"
            ms.reversal_detail = ms.reversal_detail or {"reason": "未启用反转检测"}

    if not skip_orderbook and market_list and (not no_rest_backfill):
        print(
            f"[HEARTBEAT] 开始 REST /books 回补：{len(market_list)} 个市场，批次大小={books_batch_size}",
            flush=True,
        )
        _rest_books_backfill(
            market_list, batch_size=books_batch_size, timeout=books_timeout
        )

    chosen: List[MarketSnapshot] = []
    rejects: List[Tuple[MarketSnapshot, str]] = early_rejects.copy()
    for ms in market_list:
        ok, reason = _final_pass_reason(ms, require_quotes=(not allow_illiquid))
        if ok:
            chosen.append(ms)
        else:
            rejects.append((ms, reason))

    event_candidates: Dict[str, List[Tuple[MarketSnapshot, OutcomeSnapshot, float, float]]] = {}
    event_reject_slugs: set[str] = set()
    missing_event_id_count = 0
    highlights: List[HighlightedOutcome] = []

    for ms in chosen:
        hits = _highlight_outcomes(ms, require_reversal=enable_reversal)
        if not hits:
            continue
        snap, hours = _best_outcome(hits)
        ek = _event_key(ms)
        if ek is None:
            missing_event_id_count += 1
            highlights.append(
                HighlightedOutcome(
                    market=ms, outcome=snap, hours_to_end=hours
                )
            )
            continue
        event_candidates.setdefault(ek, []).append(
            (ms, snap, hours, _outcome_price(snap))
        )

    for ek, cand_list in event_candidates.items():
        cand_sorted = sorted(
            cand_list,
            key=lambda item: (
                0 if (item[1].name or "").upper() == "NO" else 1,
                -item[3],
                item[0].slug,
            ),
        )

        best_ms, best_snap, best_hours, _ = cand_sorted[0]
        highlights.append(
            HighlightedOutcome(market=best_ms, outcome=best_snap, hours_to_end=best_hours)
        )

        for ms, _, _, _ in cand_sorted[1:]:
            event_reject_slugs.add(ms.slug)
            rejects.append((ms, f"同事件已选更优市场（event={ek}）"))

    if event_reject_slugs:
        chosen = [ms for ms in chosen if ms.slug not in event_reject_slugs]

    return FilterResult(
        total_markets=len(mkts_raw),
        candidates=market_list,
        chosen=chosen,
        rejected=rejects,
        highlights=highlights,
        merged_event_count=len(event_reject_slugs),
        missing_event_id_count=missing_event_id_count,
    )


def _print_highlighted(highlights: List[Tuple[MarketSnapshot, OutcomeSnapshot, float]]) -> None:
    if not highlights:
        print(f"[INFO] 当前无满足（{_highlight_label()}）条件的选项。")
        return

    print(f"[INFO] 满足（{_highlight_label()}）条件的选项：")
    for idx, (ms, snap, hours) in enumerate(highlights, start=1):
        bid = "-" if snap.bid is None else f"{snap.bid:.4f}"
        ask = "-" if snap.ask is None else f"{snap.ask:.4f}"
        end_iso = ms.end_time.isoformat() if ms.end_time else "-"
        detail = ms.reversal_detail or {}
        rev_status = "REV✔" if ms.reversal_hit else "REV✘"
        max_old = detail.get("max_old")
        max_recent = detail.get("max_recent") or detail.get("max_recent_short")
        rev_desc_parts = [rev_status]
        if max_old is not None:
            rev_desc_parts.append(f"old_max={max_old:.3f}")
        if max_recent is not None:
            rev_desc_parts.append(f"recent_max={max_recent:.3f}")
        if detail.get("reason"):
            rev_desc_parts.append(str(detail.get("reason")))
        rev_desc = "; ".join(rev_desc_parts)
        print(
            f"  [{idx}] slug={ms.slug} | 标题={ms.title} | 方向={snap.name}"
            f" | token_id={snap.token_id or '-'} | bid/ask={bid}/{ask}"
            f" | ends_in={hours}h | end_time={end_iso} | {rev_desc}"
        )


# -------------------------------
# 主流程（含流式模式）
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Polymarket 市场筛选（REST-only：/books 批量回补买一/卖一）")
    ap.add_argument("--books-batch-size", type=int, default=200, help="REST /books 批量回补的 token_id 数量上限（非流式模式）")
    ap.add_argument("--books-timeout", type=float, default=10.0, help="REST /books 回补单次请求超时时间（秒，非流式模式）")
    ap.add_argument("--no_rest_backfill", dest="no_rest_backfill", action="store_true", help="关闭 REST 回补（诊断用，默认开启）")
    ap.add_argument("--skip-orderbook", action="store_true", help="跳过任何订单簿/价格回补（仅诊断）")
    ap.add_argument("--allow-illiquid", action="store_true", help="允许无报价市场通过（仅诊断）")

    ap.add_argument("--min-end-hours", type=float, default=DEFAULT_MIN_END_HOURS, help="仅抓取结束时间晚于该阈值（小时）的市场")
    ap.add_argument("--max-end-days", type=int, default=DEFAULT_MAX_END_DAYS, help="仅抓取结束时间在未来 N 天内的市场")
    ap.add_argument("--gamma-window-days", type=int, default=DEFAULT_GAMMA_WINDOW_DAYS, help="Gamma 时间切片的窗口大小（天），命中 500 会自动递归切分")
    ap.add_argument("--gamma-min-window-hours", type=int, default=DEFAULT_GAMMA_MIN_WINDOW_HOURS, help="Gamma 时间切片命中 500 时递归拆分的最小窗口（小时）；窗口缩到该级别仍满额会按 endDate 继续分页")

    ap.add_argument("--legacy-end-days", type=int, default=DEFAULT_LEGACY_END_DAYS, help="结束早于 N 天视为旧格式/归档（默认 730 天）")

    # 高亮（严格口径）参数：不指定时使用脚本顶部的 HIGHLIGHT_* 默认值
    ap.add_argument("--hl-max-hours", type=float, default=None,
                    help="高亮条件：剩余时间 ≤ 该阈值（小时），例如 48 表示 48 小时内")
    ap.add_argument("--hl-min-total-volume", type=float, default=None,
                    help="高亮条件：总成交量下限（USDC），例如 10000 表示 ≥1 万 USDC")
    ap.add_argument("--hl-max-ask-diff", type=float, default=None,
                    help="高亮条件：单边点差 |ask-bid| 上限，例如 0.10 表示 ≤10 个点")

    # 价格反转检测参数
    ap.add_argument("--disable-reversal", action="store_true", help="关闭价格反转检测（默认开启）")
    ap.add_argument("--rev-p1", type=float, default=None, help="反转判定：旧段最高价需低于该值（默认 0.35）")
    ap.add_argument("--rev-p2", type=float, default=None, help="反转判定：近段最高价需高于该值（默认 0.80）")
    ap.add_argument("--rev-window-hours", type=float, default=None, help="反转判定：近段窗口大小（小时，默认 2）")
    ap.add_argument("--rev-lookback-days", type=float, default=None, help="反转判定：旧段回溯天数（默认 5 天）")
    ap.add_argument("--rev-short-interval", type=str, default=None, help="短窗口预筛 interval（如 6h/1d，默认 6h）")
    ap.add_argument("--rev-short-fidelity", type=int, default=None, help="短窗口 fidelity（分钟级，默认 15）")
    ap.add_argument("--rev-long-fidelity", type=int, default=None, help="长窗口 fidelity（分钟级，默认 60）")

    ap.add_argument("--diagnose", action="store_true", help="打印诊断信息（非流式模式下打印样本）")
    ap.add_argument("--diagnose-samples", type=int, default=30, help="诊断打印的样本数上限（非流式模式）")
    ap.add_argument("--only", type=str, default="", help="仅处理包含该子串的 slug/title（大小写不敏感）")

    # 流式输出选项
    ap.add_argument("--stream", action="store_true", help="启用流式逐个输出（按分片处理）")
    ap.add_argument("--stream-chunk-size", type=int, default=200, help="流式：每个分片的市场数量")
    ap.add_argument("--stream-books-batch-size", type=int, default=200, help="流式：每个分片内 REST /books 批量回补的 token_id 数量上限")
    ap.add_argument("--stream-books-timeout", type=float, default=10.0, help="流式：REST /books 回补单次请求超时时间（秒）")
    ap.add_argument("--stream-verbose", action="store_true", help="流式：逐个输出详细块（默认仅单行）")
    args = ap.parse_args()

    # 若指定了高亮参数，则覆盖全局 HIGHLIGHT_*，以便后续筛选与标签展示使用
    global HIGHLIGHT_MAX_HOURS, HIGHLIGHT_MIN_TOTAL_VOLUME, HIGHLIGHT_MAX_ASK_DIFF
    if args.hl_max_hours is not None:
        HIGHLIGHT_MAX_HOURS = args.hl_max_hours
    if args.hl_min_total_volume is not None:
        HIGHLIGHT_MIN_TOTAL_VOLUME = args.hl_min_total_volume
    if args.hl_max_ask_diff is not None:
        HIGHLIGHT_MAX_ASK_DIFF = args.hl_max_ask_diff

    global REVERSAL_ENABLED, REVERSAL_P1, REVERSAL_P2, REVERSAL_WINDOW_HOURS
    global REVERSAL_LOOKBACK_DAYS, REVERSAL_SHORT_INTERVAL, REVERSAL_SHORT_FIDELITY
    global REVERSAL_LONG_FIDELITY

    REVERSAL_ENABLED = not args.disable_reversal
    REVERSAL_P1 = REVERSAL_P1 if args.rev_p1 is None else args.rev_p1
    REVERSAL_P2 = REVERSAL_P2 if args.rev_p2 is None else args.rev_p2
    REVERSAL_WINDOW_HOURS = REVERSAL_WINDOW_HOURS if args.rev_window_hours is None else args.rev_window_hours
    REVERSAL_LOOKBACK_DAYS = REVERSAL_LOOKBACK_DAYS if args.rev_lookback_days is None else args.rev_lookback_days
    REVERSAL_SHORT_INTERVAL = REVERSAL_SHORT_INTERVAL if args.rev_short_interval is None else args.rev_short_interval
    REVERSAL_SHORT_FIDELITY = REVERSAL_SHORT_FIDELITY if args.rev_short_fidelity is None else args.rev_short_fidelity
    REVERSAL_LONG_FIDELITY = REVERSAL_LONG_FIDELITY if args.rev_long_fidelity is None else args.rev_long_fidelity

    rev_enable = REVERSAL_ENABLED
    rev_p1 = REVERSAL_P1
    rev_p2 = REVERSAL_P2
    rev_window = REVERSAL_WINDOW_HOURS
    rev_lookback = REVERSAL_LOOKBACK_DAYS
    rev_short_interval = REVERSAL_SHORT_INTERVAL
    rev_short_fidelity = REVERSAL_SHORT_FIDELITY
    rev_long_fidelity = REVERSAL_LONG_FIDELITY

    # 仅抓未来盘：时间窗口 = [now + min_end_hours, now + max_end_days]
    now = _now_utc()
    end_min = now + dt.timedelta(hours=args.min_end_hours)
    end_max = now + dt.timedelta(days=args.max_end_days)

    mkts_raw = fetch_markets_windowed(
        end_min,
        end_max,
        window_days=args.gamma_window_days,
        min_window_hours=args.gamma_min_window_hours,
    )
    print(f"[TRACE] 采用时间切片抓取完成：共获取 {len(mkts_raw)} 条（窗口={args.gamma_window_days} 天，最小窗口={args.gamma_min_window_hours} 小时）")

    only_pat = args.only.lower().strip()

    # ---------- 流式模式 ----------
    if args.stream:
        total = len(mkts_raw)
        processed = 0
        chosen_cnt = 0
        highlights: List[Tuple[MarketSnapshot, OutcomeSnapshot, float]] = []
        for s in range(0, total, args.stream_chunk_size):
            chunk_raw = mkts_raw[s:s + args.stream_chunk_size]
            chunk_idx = s // args.stream_chunk_size + 1
            # 解析 + 早筛（即时输出被拒绝的理由）
            candidates: List[MarketSnapshot] = []
            for raw in chunk_raw:
                title = (raw.get("question") or raw.get("title") or "")
                slug  = (raw.get("slug") or "")
                if only_pat and (only_pat not in title.lower() and only_pat not in slug.lower()):
                    continue
                ms = _parse_market(raw)
                ok, reason = _early_filter_reason(ms, args.min_end_hours, args.legacy_end_days)
                if ok:
                    candidates.append(ms)
                else:
                    if args.stream_verbose:
                        _print_snapshot(processed+1, total, ms)
                        print(f"[TRACE]   -> 结果：{reason}。")
                        print(f"[TRACE]   --------------------------------------------------")
                    else:
                        _print_singleline(ms, reason)
                processed += 1

            print(
                f"[HEARTBEAT] 流式分片 {chunk_idx} 初筛通过 {len(candidates)} / {len(chunk_raw)}，累计原始进度 {processed}/{total}",
                flush=True,
            )
            _print_topics_summary(candidates, stage=f"分片 {chunk_idx} 粗筛通过")

            # 分片内批量 REST 回补
            if rev_enable and candidates:
                for ms in candidates:
                    token_id = ms.yes.token_id or ms.no.token_id
                    side = "YES" if ms.yes.token_id else "NO"
                    try:
                        hit, detail = detect_reversal(
                            token_id,
                            p1=rev_p1,
                            p2=rev_p2,
                            window_hours=rev_window,
                            lookback_days=rev_lookback,
                            short_interval=rev_short_interval,
                            short_fidelity=rev_short_fidelity,
                            long_fidelity=rev_long_fidelity,
                        )
                    except Exception as exc:
                        hit, detail = False, {"reason": f"反转检测异常: {exc}"}
                    ms.reversal_hit = hit
                    ms.reversal_side = side
                    ms.reversal_detail = detail
            else:
                for ms in candidates:
                    ms.reversal_hit = True
                    ms.reversal_side = ms.reversal_side or "YES"
                    ms.reversal_detail = ms.reversal_detail or {"reason": "未启用反转检测"}

            if not args.skip_orderbook and candidates and (not args.no_rest_backfill):
                _rest_books_backfill(
                    candidates,
                    batch_size=args.stream_books_batch_size,
                    timeout=args.stream_books_timeout,
                )

            # 最终判定（即时输出）
            for ms in candidates:
                ok2, reason2 = _final_pass_reason(ms, require_quotes=(not args.allow_illiquid))
                if args.stream_verbose:
                    _print_snapshot(processed+1, total, ms)
                    print(f"[TRACE]   -> 结果：{reason2}。")
                    print(f"[TRACE]   --------------------------------------------------")
                else:
                    _print_singleline(ms, reason2)
                if ok2:
                    chosen_cnt += 1
                    for snap, hours in _highlight_outcomes(ms, require_reversal=rev_enable):
                        highlights.append((ms, snap, hours))
                processed += 1

        print("")
        _print_highlighted(highlights)
        print(f"\n[INFO] 通过筛选的市场数量：{chosen_cnt} / {len(mkts_raw)}")
        return

    # ---------- 非流式模式（批量） ----------
    result = collect_filter_results(
        min_end_hours=args.min_end_hours,
        max_end_days=args.max_end_days,
        gamma_window_days=args.gamma_window_days,
        gamma_min_window_hours=args.gamma_min_window_hours,
        legacy_end_days=args.legacy_end_days,
        allow_illiquid=args.allow_illiquid,
        skip_orderbook=args.skip_orderbook,
        no_rest_backfill=args.no_rest_backfill,
        books_batch_size=args.books_batch_size,
        books_timeout=args.books_timeout,
        only=args.only,
        prefetched_markets=mkts_raw,
        enable_reversal=rev_enable,
        reversal_p1=rev_p1,
        reversal_p2=rev_p2,
        reversal_window_hours=rev_window,
        reversal_lookback_days=rev_lookback,
        reversal_short_interval=rev_short_interval,
        reversal_short_fidelity=rev_short_fidelity,
        reversal_long_fidelity=rev_long_fidelity,
    )

    if args.diagnose:
        shown = 0
        for i, (ms, reason) in enumerate(result.rejected[:args.diagnose_samples], start=1):
            _print_snapshot(i, len(result.rejected), ms)
            print(f"[TRACE]   -> 结果：{reason}。")
            print(f"[TRACE]   --------------------------------------------------")
            shown += 1
        if result.chosen:
            print("[INFO] （通过样本，最多显示 10 个）")
            for k, ms in enumerate(result.chosen[:10], start=1):
                yb = "-" if ms.yes.bid is None else f"{ms.yes.bid:.4f}"
                ya = "-" if ms.yes.ask is None else f"{ms.yes.ask:.4f}"
                nb = "-" if ms.no.bid is None else f"{ms.no.bid:.4f}"
                na = "-" if ms.no.ask is None else f"{ms.no.ask:.4f}"
                print(f"  [{k}] {ms.slug} | YES bid/ask={yb}/{ya} | NO bid/ask={nb}/{na} | LQ={_fmt_money(ms.liquidity)} Vol={_fmt_money(ms.totalVolume)}")

    printable_highlights = [
        (ho.market, ho.outcome, ho.hours_to_end) for ho in result.highlights
    ]

    print("")
    _print_highlighted(printable_highlights)

    print("")
    print(f"[INFO] 通过筛选的市场数量：{len(result.chosen)} / {result.total_markets}")
    print(f"[INFO] 合并同类项数量：{result.merged_event_count}")
    print(f"[INFO] 未获取到事件ID的数量：{result.missing_event_id_count}")

if __name__ == "__main__":
    main()
