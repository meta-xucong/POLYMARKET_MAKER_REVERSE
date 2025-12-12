# -*- coding: utf-8 -*-
"""Maker-only execution helpers for Polymarket trading.

This module provides two high-level routines used by the volatility arbitrage
script:

``maker_buy_follow_bid``
    Place a GTC buy order at the current best bid and keep adjusting the order
    upward whenever the market bid rises. The routine polls every ``poll_sec``
    seconds, accumulates fills, and exits once the requested quantity is filled
    (or the remainder falls below the minimum notional requirement).

``maker_sell_follow_ask_with_floor_wait``
    Place a GTC sell order at ``max(best_ask, floor_X)`` and follow the ask
    downward without crossing below the provided floor price. If the ask drops
    below the floor the routine cancels the working order and waits until the
    market recovers above the floor before re-posting.

Both helpers favour websocket snapshots supplied by the caller via
``best_bid_fn`` / ``best_ask_fn``. When these callables are absent or return
``None`` the helpers fall back to best-effort REST lookups using the provided
client.

The functions return lightweight dictionaries that summarise order history and
fill statistics so that the strategy layer can update its internal state.
"""
from __future__ import annotations

import math
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

from trading.execution import ClobPolymarketAPI


BUY_PRICE_DP = 2
BUY_SIZE_DP = 4
SELL_PRICE_DP = 4
SELL_SIZE_DP = 2
_MIN_FILL_EPS = 1e-9
DEFAULT_MIN_ORDER_SIZE = 5.0


def _round_up_to_dp(value: float, dp: int) -> float:
    factor = 10 ** dp
    return math.ceil(value * factor - 1e-12) / factor


def _round_down_to_dp(value: float, dp: int) -> float:
    factor = 10 ** dp
    return math.floor(value * factor + 1e-12) / factor


def _ceil_to_dp(value: float, dp: int) -> float:
    factor = 10 ** dp
    return math.ceil(value * factor - 1e-12) / factor


def _floor_to_dp(value: float, dp: int) -> float:
    factor = 10 ** dp
    return math.floor(value * factor + 1e-12) / factor


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None
    return None


class PriceSample(NamedTuple):
    price: float
    decimals: Optional[int]


def _infer_price_decimals(value: Any, *, max_dp: int = 6) -> Optional[int]:
    candidate: Optional[Decimal] = None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            candidate = Decimal(raw)
        except (InvalidOperation, ValueError):
            return None
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            candidate = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return None
    else:
        return None

    candidate = candidate.normalize()
    if candidate.is_zero():
        return 0
    exponent = candidate.as_tuple().exponent
    if exponent >= 0:
        return 0
    return min(-int(exponent), max_dp)


def _extract_best_price(payload: Any, side: str) -> Optional[PriceSample]:
    numeric = _coerce_float(payload)
    if numeric is not None:
        decimals = _infer_price_decimals(payload)
        return PriceSample(float(numeric), decimals)

    if isinstance(payload, Mapping):
        primary_keys = {
            "bid": (
                "best_bid",
                "bestBid",
                "bid",
                "highestBid",
                "bestBidPrice",
                "bidPrice",
                "buy",
            ),
            "ask": (
                "best_ask",
                "bestAsk",
                "ask",
                "offer",
                "best_offer",
                "bestOffer",
                "lowestAsk",
                "sell",
            ),
        }[side]
        for key in primary_keys:
            if key in payload:
                extracted = _extract_best_price(payload[key], side)
                if extracted is not None:
                    return extracted

        ladder_keys = {
            "bid": ("bids", "bid_levels", "buy_orders", "buyOrders"),
            "ask": ("asks", "ask_levels", "sell_orders", "sellOrders", "offers"),
        }[side]
        for key in ladder_keys:
            if key in payload:
                ladder = payload[key]
                if isinstance(ladder, Iterable) and not isinstance(ladder, (str, bytes, bytearray)):
                    for entry in ladder:
                        if isinstance(entry, Mapping) and "price" in entry:
                            decimals = _infer_price_decimals(entry.get("price"))
                            candidate = _coerce_float(entry.get("price"))
                            if candidate is not None:
                                return PriceSample(float(candidate), decimals)
                        extracted = _extract_best_price(entry, side)
                        if extracted is not None:
                            return extracted

        for value in payload.values():
            extracted = _extract_best_price(value, side)
            if extracted is not None:
                return extracted
        return None

    if isinstance(payload, Iterable) and not isinstance(payload, (str, bytes, bytearray)):
        for item in payload:
            extracted = _extract_best_price(item, side)
            if extracted is not None:
                return extracted
        return None

    return None


def _fetch_best_price(client: Any, token_id: str, side: str) -> Optional[PriceSample]:
    method_candidates = (
        ("get_market_orderbook", {"market": token_id}),
        ("get_market_orderbook", {"token_id": token_id}),
        ("get_market_orderbook", {"market_id": token_id}),
        ("get_order_book", {"market": token_id}),
        ("get_order_book", {"token_id": token_id}),
        ("get_orderbook", {"market": token_id}),
        ("get_orderbook", {"token_id": token_id}),
        ("get_market", {"market": token_id}),
        ("get_market", {"token_id": token_id}),
        ("get_market_data", {"market": token_id}),
        ("get_market_data", {"token_id": token_id}),
        ("get_ticker", {"market": token_id}),
        ("get_ticker", {"token_id": token_id}),
    )

    for name, kwargs in method_candidates:
        fn = getattr(client, name, None)
        if not callable(fn):
            continue
        try:
            resp = fn(**kwargs)
        except TypeError:
            continue
        except Exception:
            continue

        payload = resp
        if isinstance(resp, tuple) and len(resp) == 2:
            payload = resp[1]
        if isinstance(payload, Mapping) and {"data", "status"} <= set(payload.keys()):
            payload = payload.get("data")

        best = _extract_best_price(payload, side)
        if best is not None:
            return PriceSample(float(best.price), best.decimals)
    return None


def _best_price_info(
    client: Any,
    token_id: str,
    best_fn: Optional[Callable[[], Optional[float]]],
    side: str,
) -> Optional[PriceSample]:
    if best_fn is not None:
        try:
            val = best_fn()
        except Exception:
            val = None
        if val is not None and val > 0:
            return PriceSample(float(val), _infer_price_decimals(val))
    return _fetch_best_price(client, token_id, side)


def _best_bid(
    client: Any, token_id: str, best_bid_fn: Optional[Callable[[], Optional[float]]]
) -> Optional[float]:
    info = _best_price_info(client, token_id, best_bid_fn, "bid")
    if info is None:
        return None
    return info.price


def _best_bid_info(
    client: Any, token_id: str, best_bid_fn: Optional[Callable[[], Optional[float]]]
) -> Optional[PriceSample]:
    return _best_price_info(client, token_id, best_bid_fn, "bid")


def _best_ask(
    client: Any, token_id: str, best_ask_fn: Optional[Callable[[], Optional[float]]]
) -> Optional[float]:
    info = _best_price_info(client, token_id, best_ask_fn, "ask")
    if info is None:
        return None
    return info.price


def _cancel_order(client: Any, order_id: Optional[str]) -> bool:
    if not order_id:
        return False
    method_names = (
        "cancel_order",
        "cancelOrder",
        "cancel",
        "cancel_orders",
        "cancelOrders",
        "delete_order",
        "deleteOrder",
        "cancel_limit_order",
        "cancelLimitOrder",
        "cancel_open_order",
        "cancelOpenOrder",
    )

    targets: deque[Any] = deque([client])
    visited: set[int] = set()
    while targets:
        obj = targets.popleft()
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in visited:
            continue
        visited.add(obj_id)
        for name in method_names:
            method = getattr(obj, name, None)
            if not callable(method):
                continue
            try:
                method(order_id)
                return True
            except TypeError:
                try:
                    method(id=order_id)
                    return True
                except Exception:
                    continue
            except Exception:
                continue
        for attr in ("client", "api", "private"):
            nested = getattr(obj, attr, None)
            if nested is not None:
                targets.append(nested)
    return False


def _order_tick(dp: int) -> float:
    return 10 ** (-dp)


def _update_fill_totals(
    order_id: str,
    status_payload: Dict[str, Any],
    accounted: Dict[str, float],
    notional_sum: float,
    last_known_price: float,
    *,
    status_text: Optional[str] = None,
    expected_full_size: Optional[float] = None,
) -> Tuple[float, float, float]:
    avg_price = _coerce_float(status_payload.get("avgPrice"))

    price_keys = (
        "avgPrice",
        "averagePrice",
        "avg_price",
        "filledAvgPrice",
        "filledAveragePrice",
        "executionPrice",
        "averageExecutionPrice",
        "fillPrice",
        "matchedPrice",
        "price",
        "lastPrice",
        "lastTradePrice",
        "markPrice",
    )

    size_keys = (
        "size",
        "quantity",
        "qty",
        "amount",
        "filledAmount",
        "filled",
        "filledQuantity",
        "filledSize",
        "matchedShares",
        "shares",
        "baseAmount",
        "takingAmount",
        "takerAmount",
        "taker_amount",
    )

    fills_payload = status_payload.get("fills")
    fills_sequence = fills_payload if isinstance(fills_payload, (list, tuple)) else None

    total_from_fills = 0.0
    total_notional = 0.0

    if fills_sequence is not None:
        for entry in fills_sequence:
            if not isinstance(entry, dict):
                continue
            size_val: Optional[float] = None
            for key in size_keys:
                size_val = _coerce_float(entry.get(key))
                if size_val is not None and size_val > 0:
                    break
            if size_val is None or size_val <= 0:
                continue
            total_from_fills += size_val

            price_val: Optional[float] = None
            for key in price_keys:
                price_val = _coerce_float(entry.get(key))
                if price_val is not None:
                    break
            if price_val is not None:
                total_notional += price_val * size_val

    filled_amount = _coerce_float(status_payload.get("filledAmount")) or 0.0
    filled_amount_quote = _coerce_float(status_payload.get("filledAmountQuote"))

    if filled_amount <= _MIN_FILL_EPS and total_from_fills > 0:
        filled_amount = total_from_fills

    if avg_price is None and total_from_fills > 0 and total_notional > 0:
        avg_price = total_notional / total_from_fills

    if filled_amount <= _MIN_FILL_EPS and filled_amount_quote is not None:
        price_hint = avg_price if avg_price is not None else last_known_price
        if price_hint and price_hint > 0:
            filled_amount = max(filled_amount, filled_amount_quote / max(price_hint, 1e-12))

    if avg_price is None:
        avg_price = last_known_price

    if filled_amount <= _MIN_FILL_EPS and status_text:
        status_upper = status_text.upper()
        if status_upper in {"FILLED", "MATCHED", "COMPLETED", "EXECUTED"}:
            if expected_full_size is not None and expected_full_size > 0:
                filled_amount = max(filled_amount, float(expected_full_size))

    previous = accounted.get(order_id, 0.0)
    delta = max(filled_amount - previous, 0.0)
    accounted[order_id] = filled_amount
    notional_sum += delta * avg_price
    return filled_amount, avg_price, notional_sum


def maker_buy_follow_bid(
    client: Any,
    token_id: str,
    target_size: float,
    *,
    poll_sec: float = 10.0,
    min_quote_amt: float = 1.0,
    min_order_size: float = DEFAULT_MIN_ORDER_SIZE,
    best_bid_fn: Optional[Callable[[], Optional[float]]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    progress_probe: Optional[Callable[[], None]] = None,
    progress_probe_interval: float = 60.0,
    price_dp: Optional[int] = None,
    external_fill_probe: Optional[Callable[[], Optional[float]]] = None,
) -> Dict[str, Any]:
    """Continuously maintain a maker buy order following the market bid."""

    goal_size = max(_ceil_to_dp(float(target_size), BUY_SIZE_DP), 0.0)
    api_min_qty = 0.0
    if min_order_size and min_order_size > 0:
        api_min_qty = _ceil_to_dp(float(min_order_size), BUY_SIZE_DP)
        goal_size = max(goal_size, api_min_qty)
    if goal_size <= 0:
        return {
            "status": "SKIPPED",
            "avg_price": None,
            "filled": 0.0,
            "remaining": 0.0,
            "orders": [],
        }

    adapter = ClobPolymarketAPI(client)
    orders: List[Dict[str, Any]] = []
    records: Dict[str, Dict[str, Any]] = {}
    accounted: Dict[str, float] = {}

    remaining = goal_size
    filled_total = 0.0
    notional_sum = 0.0

    active_order: Optional[str] = None
    active_price: Optional[float] = None

    final_status = "PENDING"
    base_price_dp = BUY_PRICE_DP if price_dp is None else max(int(price_dp), 0)
    price_dp_active = base_price_dp
    tick = _order_tick(price_dp_active)
    # 采用统一的两级缩减步长，先用 0.01，多次失败后升级到 0.1
    size_tick = 0.01
    shortage_retry_count = 0
    base_min_shrink_interval = 1.0
    min_shrink_interval = base_min_shrink_interval
    last_shrink_time = 0.0

    no_fill_poll_count = 0

    next_probe_at = 0.0

    def _maybe_update_price_dp(observed: Optional[int]) -> None:
        nonlocal price_dp_active, tick
        if observed is None:
            return
        desired = max(base_price_dp, int(observed))
        if desired != price_dp_active:
            price_dp_active = desired
            tick = _order_tick(price_dp_active)
            print(f"[MAKER][BUY] 检测到市场价格精度 -> decimals={price_dp_active}")

    def _is_insufficient_balance(value: object) -> bool:
        def _text_has_shortage(text: str) -> bool:
            lowered = text.lower()
            shortage_keywords = ("insufficient", "not enough")
            balance_keywords = ("balance", "fund", "allowance")
            return any(key in lowered for key in shortage_keywords) and any(
                key in lowered for key in balance_keywords
            )

        if hasattr(value, "error_message"):
            try:
                if _is_insufficient_balance(getattr(value, "error_message")):
                    return True
            except Exception:
                pass
        if hasattr(value, "response"):
            try:
                if _is_insufficient_balance(getattr(value, "response")):
                    return True
            except Exception:
                pass
        if hasattr(value, "args"):
            try:
                for arg in getattr(value, "args", ()):
                    if _is_insufficient_balance(arg):
                        return True
            except Exception:
                pass

        if isinstance(value, dict):
            for key in ("error", "message", "detail", "reason", "status"):
                if key in value and _is_insufficient_balance(value[key]):
                    return True
        try:
            return _text_has_shortage(str(value))
        except Exception:
            return False

    def _reset_shortage_recovery(note: str) -> None:
        nonlocal shortage_retry_count, min_shrink_interval, last_shrink_time

        if shortage_retry_count > 0 or min_shrink_interval != base_min_shrink_interval:
            shortage_retry_count = 0
            min_shrink_interval = base_min_shrink_interval
            last_shrink_time = time.monotonic()
            print(note)

    def _handle_balance_shortage(reason: str, min_viable: float) -> bool:
        nonlocal goal_size, remaining, active_order, active_price, final_status, shortage_retry_count, size_tick, last_shrink_time, min_shrink_interval

        print(reason)
        min_shrink_interval = max(min_shrink_interval, base_min_shrink_interval)
        if active_order:
            _cancel_order(client, active_order)
            rec = records.get(active_order)
            if rec is not None:
                rec["status"] = "CANCELLED"
        active_order = None
        active_price = None
        current_remaining = max(goal_size - filled_total, 0.0)
        if current_remaining <= _MIN_FILL_EPS:
            final_status = "FILLED" if filled_total > _MIN_FILL_EPS else final_status
            return True
        shortage_retry_count += 1
        if shortage_retry_count > 100 and size_tick < 0.1:
            size_tick = 0.1
            print("[MAKER][BUY] 余额不足重试超过 100 次，提升缩减步长至 0.1。")

        now = time.monotonic()
        elapsed = now - last_shrink_time
        if elapsed < min_shrink_interval:
            sleep_duration = min_shrink_interval - elapsed
            if sleep_duration > 0:
                sleep_fn(sleep_duration)
            now = time.monotonic()
        last_shrink_time = now

        shrink_candidate = _ceil_to_dp(max(current_remaining - size_tick, 0.0), BUY_SIZE_DP)
        min_viable = max(min_viable or 0.0, api_min_qty or 0.0)
        if shrink_candidate > _MIN_FILL_EPS and (
            not min_viable or shrink_candidate + _MIN_FILL_EPS >= min_viable
        ):
            print(
                "[MAKER][BUY] 重新调整买入目标 -> "
                f"old={current_remaining:.{BUY_SIZE_DP}f} new={shrink_candidate:.{BUY_SIZE_DP}f}"
            )
            goal_size = filled_total + shrink_candidate
            remaining = max(goal_size - filled_total, 0.0)
            return False
        print("[MAKER][BUY] 无法在满足最小下单量的前提下继续缩减，终止买入。")
        final_status = "FILLED_TRUNCATED" if filled_total > _MIN_FILL_EPS else "SKIPPED_TOO_SMALL"
        return True

    while True:
        if stop_check and stop_check():
            if active_order:
                _cancel_order(client, active_order)
                rec = records.get(active_order)
                if rec is not None:
                    rec["status"] = "CANCELLED"
            final_status = "STOPPED"
            break

        if active_order is None:
            if api_min_qty and remaining + _MIN_FILL_EPS < api_min_qty:
                final_status = "FILLED_TRUNCATED" if filled_total > _MIN_FILL_EPS else "SKIPPED_TOO_SMALL"
                break
            bid_info = _best_bid_info(client, token_id, best_bid_fn)
            if bid_info is None:
                sleep_fn(poll_sec)
                continue
            bid = bid_info.price
            if bid <= 0:
                sleep_fn(poll_sec)
                continue
            _maybe_update_price_dp(bid_info.decimals)
            px = _round_up_to_dp(bid, price_dp_active)
            if px <= 0:
                sleep_fn(poll_sec)
                continue
            min_qty = 0.0
            if min_quote_amt and min_quote_amt > 0:
                min_qty = _ceil_to_dp(min_quote_amt / max(px, 1e-9), BUY_SIZE_DP)
            eff_qty = max(remaining, min_qty)
            if api_min_qty:
                eff_qty = max(eff_qty, api_min_qty)
            eff_qty = _ceil_to_dp(eff_qty, BUY_SIZE_DP)
            if eff_qty <= 0:
                final_status = "SKIPPED"
                break
            payload = {
                "tokenId": token_id,
                "side": "BUY",
                "price": px,
                "size": eff_qty,
                "timeInForce": "GTC",
                "type": "GTC",
                "allowPartial": True,
            }
            try:
                response = adapter.create_order(payload)
            except Exception as exc:
                min_viable = max(min_qty or 0.0, api_min_qty or 0.0)
                if _is_insufficient_balance(exc):
                    should_stop = _handle_balance_shortage(
                        "[MAKER][BUY] 下单失败，疑似余额不足，尝试缩减买入目标后重试。",
                        min_viable,
                    )
                    if should_stop:
                        break
                    continue
                raise
            order_id = str(response.get("orderId"))
            record = {
                "id": order_id,
                "side": "buy",
                "price": px,
                "size": eff_qty,
                "status": "OPEN",
                "filled": 0.0,
            }
            orders.append(record)
            records[order_id] = record
            accounted[order_id] = 0.0
            active_order = order_id
            active_price = px
            _reset_shortage_recovery("[MAKER][BUY] 挂单成功，退出余额不足重试模式。")
            if progress_probe:
                interval = max(progress_probe_interval, poll_sec, 1e-6)
                try:
                    progress_probe()
                except Exception as probe_exc:
                    print(f"[MAKER][BUY] 进度探针执行异常：{probe_exc}")
                next_probe_at = time.time() + interval
            print(
                f"[MAKER][BUY] 挂单 -> price={px:.{price_dp_active}f} qty={eff_qty:.{BUY_SIZE_DP}f} remaining={remaining:.{BUY_SIZE_DP}f}"
            )
            continue

        sleep_fn(poll_sec)
        if (
            progress_probe
            and active_order
            and progress_probe_interval > 0
            and time.time() >= max(next_probe_at, 0.0)
        ):
            try:
                progress_probe()
            except Exception as probe_exc:
                print(f"[MAKER][BUY] 进度探针执行异常：{probe_exc}")
            interval = max(progress_probe_interval, poll_sec, 1e-6)
            next_probe_at = time.time() + interval
        try:
            status_payload = adapter.get_order_status(active_order)
        except Exception as exc:
            print(f"[MAKER][BUY] 查询订单状态异常：{exc}")
            status_payload = {"status": "UNKNOWN", "filledAmount": accounted.get(active_order, 0.0)}

        record = records.get(active_order)
        status_text = str(status_payload.get("status", "UNKNOWN"))
        record_size = None
        if record is not None:
            try:
                record_size = float(record.get("size", 0.0) or 0.0)
            except Exception:
                record_size = None
        last_price_hint = active_price
        if last_price_hint is None:
            last_price_hint = _coerce_float(status_payload.get("avgPrice"))
        if last_price_hint is None:
            last_price_hint = 0.0
        previous_filled_total = filled_total

        filled_amount, avg_price, notional_sum = _update_fill_totals(
            active_order,
            status_payload,
            accounted,
            notional_sum,
            float(last_price_hint),
            status_text=status_text,
            expected_full_size=record_size,
        )
        filled_total = sum(accounted.values())
        if external_fill_probe is not None:
            try:
                external_filled = external_fill_probe()
            except Exception as probe_exc:
                print(f"[MAKER][BUY] 外部持仓校对异常：{probe_exc}")
                external_filled = None
            if external_filled is not None and external_filled > filled_total + _MIN_FILL_EPS:
                filled_total = external_filled
                remaining = max(goal_size - filled_total, 0.0)
                print(
                    f"[MAKER][BUY] 校对持仓后更新累计成交 -> filled={filled_total:.{BUY_SIZE_DP}f} "
                    f"remaining={remaining:.{BUY_SIZE_DP}f}"
                )
        if filled_total > previous_filled_total + _MIN_FILL_EPS:
            no_fill_poll_count = 0
        elif shortage_retry_count > 0:
            no_fill_poll_count += 1
        else:
            no_fill_poll_count = 0
        if shortage_retry_count > 0 and no_fill_poll_count >= 30:
            print(
                "[MAKER][BUY] 挂单连续 30 次未检测到新增成交，强制校对仓位/余额后重挂。"
            )
            if external_fill_probe is not None:
                try:
                    external_filled = external_fill_probe()
                except Exception as probe_exc:
                    print(f"[MAKER][BUY] 外部持仓校对异常：{probe_exc}")
                    external_filled = None
                if external_filled is not None and external_filled > filled_total + _MIN_FILL_EPS:
                    filled_total = external_filled
                    print(
                        f"[MAKER][BUY] 二次校对后更新累计成交 -> filled={filled_total:.{BUY_SIZE_DP}f}"
                    )
            remaining = max(goal_size - filled_total, 0.0)
            _cancel_order(client, active_order)
            rec = records.get(active_order)
            if rec is not None:
                rec["status"] = "CANCELLED"
            active_order = None
            active_price = None
            no_fill_poll_count = 0
            continue
        remaining = max(goal_size - filled_total, 0.0)
        status_text_upper = status_text.upper()
        if record is not None:
            record["filled"] = filled_amount
            record["status"] = status_text_upper
            if avg_price is not None:
                record["avg_price"] = avg_price
            price_display = record.get("price", active_price)
            total_size = float(record.get("size", 0.0) or 0.0)
            remaining_slice = max(total_size - filled_amount, 0.0)
            if price_display is not None:
                print(
                    f"[MAKER][BUY] 挂单状态 -> price={float(price_display):.{price_dp_active}f} "
                    f"filled={filled_amount:.{BUY_SIZE_DP}f} remaining={remaining_slice:.{BUY_SIZE_DP}f} "
                    f"status={status_text_upper}"
                )

        current_bid_info = _best_bid_info(client, token_id, best_bid_fn)
        current_bid = current_bid_info.price if current_bid_info is not None else None
        if current_bid_info is not None:
            _maybe_update_price_dp(current_bid_info.decimals)
        min_buyable = 0.0
        if min_quote_amt and min_quote_amt > 0 and current_bid and current_bid > 0:
            min_buyable = _ceil_to_dp(min_quote_amt / max(current_bid, 1e-9), BUY_SIZE_DP)
        if api_min_qty:
            min_buyable = max(min_buyable, api_min_qty)

        if remaining <= _MIN_FILL_EPS or (min_buyable and remaining < min_buyable):
            if active_order:
                _cancel_order(client, active_order)
                rec = records.get(active_order)
                if rec is not None:
                    rec["status"] = "CANCELLED"
                active_order = None
            if remaining <= _MIN_FILL_EPS:
                final_status = "FILLED"
            else:
                final_status = "FILLED_TRUNCATED" if filled_total > _MIN_FILL_EPS else "SKIPPED_TOO_SMALL"
            break

        if current_bid is not None and active_price is not None and current_bid >= active_price + tick - 1e-12:
            print(
                f"[MAKER][BUY] 买一上行 -> 撤单重挂 | old={active_price:.{price_dp_active}f} new={current_bid:.{price_dp_active}f}"
            )
            _cancel_order(client, active_order)
            rec = records.get(active_order)
            if rec is not None:
                rec["status"] = "CANCELLED"
            active_order = None
            active_price = None
            continue

        final_states = {"FILLED", "MATCHED", "COMPLETED", "EXECUTED"}
        cancel_states = {"CANCELLED", "CANCELED", "REJECTED", "EXPIRED"}
        invalid_states = {"INVALID"}
        status_shortage = _is_insufficient_balance(status_text) or _is_insufficient_balance(status_payload)
        if shortage_retry_count > 0 and not status_shortage:
            _reset_shortage_recovery("[MAKER][BUY] 状态恢复正常，退出余额不足重试模式。")
        if status_text_upper in invalid_states or status_shortage:
            reason = "[MAKER][BUY] 订单被撮合层标记为 INVALID，尝试调整买入目标后重试。"
            if status_shortage and status_text_upper not in invalid_states:
                reason = "[MAKER][BUY] 订单状态提示余额不足，尝试调整买入目标后重试。"
            min_viable = max(min_buyable or 0.0, api_min_qty or 0.0)
            should_stop = _handle_balance_shortage(reason, min_viable)
            if should_stop:
                break
            continue
        if status_text_upper in final_states:
            active_order = None
            active_price = None
            continue
        if status_text_upper in cancel_states:
            active_order = None
            active_price = None
            continue

    avg_price = notional_sum / filled_total if filled_total > 0 else None
    remaining = max(goal_size - filled_total, 0.0)
    return {
        "status": final_status,
        "avg_price": avg_price,
        "filled": filled_total,
        "remaining": remaining,
        "orders": orders,
    }


def maker_sell_follow_ask_with_floor_wait(
    client: Any,
    token_id: str,
    position_size: float,
    floor_X: float,
    *,
    poll_sec: float = 10.0,
    min_order_size: float = DEFAULT_MIN_ORDER_SIZE,
    best_ask_fn: Optional[Callable[[], Optional[float]]] = None,
    stop_check: Optional[Callable[[], bool]] = None,
    sleep_fn: Callable[[float], None] = time.sleep,
    sell_mode: str = "conservative",
    aggressive_step: float = 0.01,
    aggressive_timeout: float = 300.0,
    progress_probe: Optional[Callable[[], None]] = None,
    progress_probe_interval: float = 60.0,
    position_fetcher: Optional[Callable[[], Optional[float]]] = None,
    position_refresh_interval: float = 30.0,
    ask_validation_interval: float = 60.0,
    price_decimals: Optional[int] = None,
) -> Dict[str, Any]:
    """Maintain a maker sell order while respecting a profit floor."""

    goal_size = max(_floor_to_dp(float(position_size), SELL_SIZE_DP), 0.0)
    api_min_qty = 0.0
    if min_order_size and min_order_size > 0:
        api_min_qty = _ceil_to_dp(float(min_order_size), SELL_SIZE_DP)
    if goal_size < 0.01:
        return {
            "status": "SKIPPED",
            "avg_price": None,
            "filled": 0.0,
            "remaining": 0.0,
            "orders": [],
        }

    def _normalize_price_dp(val: Optional[int]) -> int:
        try:
            cand = int(val) if val is not None else None
        except Exception:
            return SELL_PRICE_DP
        if cand is None or cand < 0:
            return SELL_PRICE_DP
        return min(cand, 6)

    price_dp = _normalize_price_dp(price_decimals)
    tick = _order_tick(price_dp)

    adapter = ClobPolymarketAPI(client)
    orders: List[Dict[str, Any]] = []
    records: Dict[str, Dict[str, Any]] = {}
    accounted: Dict[str, float] = {}

    remaining = goal_size
    filled_total = 0.0
    notional_sum = 0.0

    goal_cap = goal_size

    def _apply_goal_cap(candidate: float) -> float:
        capped = min(candidate, goal_cap)
        return max(capped, filled_total)

    active_order: Optional[str] = None
    active_price: Optional[float] = None

    final_status = "PENDING"

    waiting_for_floor = False
    aggressive_mode = str(sell_mode).lower() == "aggressive"
    aggressive_timer_start: Optional[float] = None
    aggressive_timer_anchor_fill: Optional[float] = None
    aggressive_floor_locked = False
    aggressive_next_price_override: Optional[float] = None
    aggressive_locked_price: Optional[float] = None
    next_price_override: Optional[float] = None
    # 连续触发仓位不足但接口仍返回可用仓位的计数
    consecutive_insufficient_with_position = 0
    missing_position_retry = 0
    shortage_retry_count = 0
    try:
        aggressive_timeout = float(aggressive_timeout)
    except (TypeError, ValueError):
        aggressive_timeout = 300.0
    try:
        aggressive_step = float(aggressive_step)
    except (TypeError, ValueError):
        aggressive_step = 0.01
    if aggressive_step <= 0:
        aggressive_mode = False
    floor_float = _round_up_to_dp(float(floor_X), price_dp)

    try:
        position_refresh_interval = float(position_refresh_interval)
    except (TypeError, ValueError):
        position_refresh_interval = 30.0
    if position_refresh_interval < 0:
        position_fetcher = None

    try:
        ask_validation_interval = float(ask_validation_interval)
    except (TypeError, ValueError):
        ask_validation_interval = 60.0
    if ask_validation_interval <= 0:
        ask_validation_interval = None

    next_probe_at = 0.0
    next_position_refresh = 0.0
    next_ask_validation = 0.0

    def _active_reserved_size() -> float:
        if not active_order:
            return 0.0
        rec = records.get(active_order)
        if rec is None:
            return 0.0
        try:
            total_size = float(rec.get("size", 0.0) or 0.0)
        except Exception:
            total_size = 0.0
        filled_so_far = accounted.get(active_order, 0.0)
        return max(total_size - filled_so_far, 0.0)

    while True:
        if stop_check and stop_check():
            if active_order:
                _cancel_order(client, active_order)
                rec = records.get(active_order)
                if rec is not None:
                    rec["status"] = "CANCELLED"
                aggressive_timer_start = None
                aggressive_timer_anchor_fill = None
            final_status = "STOPPED"
            break

        now = time.time()
        if (
            position_fetcher
            and now >= max(next_position_refresh, 0.0)
        ):
            interval = max(position_refresh_interval, poll_sec, 1e-6)
            next_position_refresh = now + interval
            try:
                live_position = position_fetcher()
            except Exception as exc:
                print(f"[MAKER][SELL] 仓位刷新失败：{exc}")
                live_position = None
            if live_position is not None:
                try:
                    live_target = max(_floor_to_dp(float(live_position), SELL_SIZE_DP), 0.0)
                except (TypeError, ValueError):
                    live_target = None
                if live_target is not None:
                    if live_target > goal_cap:
                        goal_cap = live_target
                    reserved = _active_reserved_size()
                    adjusted_target = (
                        live_target + reserved if reserved > _MIN_FILL_EPS else live_target
                    )
                    min_goal = max(filled_total, 0.0)
                    new_goal = _apply_goal_cap(max(adjusted_target, min_goal))
                    if abs(new_goal - goal_size) > _MIN_FILL_EPS:
                        change = "扩充" if new_goal > goal_size else "收缩"
                        prev_goal = goal_size
                        goal_size = new_goal
                        remaining = max(goal_size - filled_total, 0.0)
                        print(
                            "[MAKER][SELL] 仓位更新 -> "
                            f"{change}目标至 {goal_size:.{SELL_SIZE_DP}f}"
                        )
                        if remaining <= _MIN_FILL_EPS:
                            if active_order:
                                _cancel_order(client, active_order)
                                rec = records.get(active_order)
                                if rec is not None:
                                    rec["status"] = "CANCELLED"
                                active_order = None
                                active_price = None
                            final_status = "FILLED"
                            break
                        if new_goal < prev_goal - _MIN_FILL_EPS and active_order:
                            print("[MAKER][SELL] 仓位降低，撤销当前挂单以调整数量")
                            _cancel_order(client, active_order)
                            rec = records.get(active_order)
                            if rec is not None:
                                rec["status"] = "CANCELLED"
                            active_order = None
                            active_price = None
                            aggressive_timer_start = None
                            aggressive_timer_anchor_fill = None
                            aggressive_next_price_override = None
                            next_price_override = None
                            continue

        if api_min_qty and remaining + _MIN_FILL_EPS < api_min_qty:
            final_status = "FILLED_TRUNCATED" if filled_total > _MIN_FILL_EPS else "SKIPPED_TOO_SMALL"
            break

        ask_info = _best_price_info(client, token_id, best_ask_fn, "ask")
        ask = ask_info.price if ask_info is not None else None
        if ask_info and ask_info.decimals is not None:
            detected_dp = _normalize_price_dp(ask_info.decimals)
            if detected_dp != price_dp:
                price_dp = detected_dp
                tick = _order_tick(price_dp)
                floor_float = _round_up_to_dp(floor_float, price_dp)
        if ask_validation_interval and now >= max(next_ask_validation, 0.0):
            interval = max(ask_validation_interval, poll_sec, 1e-6)
            next_ask_validation = now + interval
            validated = _fetch_best_price(client, token_id, "ask")
            if validated is not None and validated.price > 0:
                validated_price = float(validated.price)
                if validated.decimals is not None:
                    detected_dp = _normalize_price_dp(validated.decimals)
                    if detected_dp != price_dp:
                        price_dp = detected_dp
                        tick = _order_tick(price_dp)
                        floor_float = _round_up_to_dp(floor_float, price_dp)
                tolerance = max(tick * 0.5, 1e-6)
                if ask is None or abs(validated_price - ask) > tolerance:
                    prev = ask
                    ask = validated_price
                    direction = "下行" if prev is not None and validated_price < prev else "上行"
                    if prev is None:
                        print(
                            f"[MAKER][SELL] 卖一校验覆盖：无本地价，采用最新卖一 {ask:.{price_dp}f}"
                        )
                    else:
                        print(
                            "[MAKER][SELL] 卖一校验覆盖（" + direction + ") -> "
                            f"old={prev:.{price_dp}f} new={ask:.{price_dp}f}"
                        )
        if not aggressive_mode:
            if ask is None or ask <= 0:
                waiting_for_floor = True
                if active_order:
                    _cancel_order(client, active_order)
                    rec = records.get(active_order)
                    if rec is not None:
                        rec["status"] = "CANCELLED"
                    active_order = None
                    active_price = None
                    aggressive_timer_start = None
                    aggressive_timer_anchor_fill = None
                    aggressive_next_price_override = None
                    next_price_override = None
                sleep_fn(poll_sec)
                continue
            if ask < floor_float - 1e-12:
                if not waiting_for_floor:
                    print(
                        f"[MAKER][SELL] 卖一跌破地板，撤单等待 | ask={ask:.{price_dp}f} floor={floor_float:.{price_dp}f}"
                    )
                waiting_for_floor = True
                if active_order:
                    _cancel_order(client, active_order)
                    rec = records.get(active_order)
                    if rec is not None:
                        rec["status"] = "CANCELLED"
                    active_order = None
                    active_price = None
                    aggressive_timer_start = None
                    aggressive_timer_anchor_fill = None
                    aggressive_next_price_override = None
                    next_price_override = None
                sleep_fn(poll_sec)
                continue
            if waiting_for_floor and ask >= floor_float:
                waiting_for_floor = False
        else:
            if ask is None or ask <= 0:
                sleep_fn(poll_sec)
                continue
            if ask <= floor_float + 1e-12:
                aggressive_floor_locked = True
                aggressive_locked_price = floor_float
            elif aggressive_floor_locked and ask > floor_float + 1e-12:
                aggressive_floor_locked = False
                aggressive_locked_price = None

        if active_order is None:
            px_candidate = max(_round_down_to_dp(ask, price_dp), floor_float)
            if next_price_override is not None:
                px_candidate = max(
                    _round_down_to_dp(next_price_override, price_dp),
                    floor_float,
                )
                next_price_override = None
            if aggressive_mode:
                if aggressive_next_price_override is not None:
                    px_candidate = max(
                        _round_down_to_dp(aggressive_next_price_override, price_dp),
                        floor_float,
                    )
                    aggressive_next_price_override = None
                elif aggressive_locked_price is not None:
                    px_candidate = max(
                        _round_down_to_dp(aggressive_locked_price, price_dp),
                        floor_float,
                    )
                if px_candidate <= floor_float + 1e-12:
                    aggressive_floor_locked = True
                    aggressive_locked_price = floor_float
                else:
                    aggressive_locked_price = None
                    aggressive_floor_locked = False
            else:
                aggressive_next_price_override = None
            px = px_candidate
            qty = _floor_to_dp(remaining, SELL_SIZE_DP)
            if qty < 0.01:
                final_status = "FILLED"
                break
            if api_min_qty and qty + _MIN_FILL_EPS < api_min_qty:
                final_status = "FILLED_TRUNCATED" if filled_total > _MIN_FILL_EPS else "SKIPPED_TOO_SMALL"
                break
            payload = {
                "tokenId": token_id,
                "side": "SELL",
                "price": px,
                "size": qty,
                "timeInForce": "GTC",
                "type": "GTC",
                "allowPartial": True,
            }
            try:
                response = adapter.create_order(payload)
            except Exception as exc:
                msg = str(exc).lower()
                insufficient = any(
                    keyword in msg for keyword in ("insufficient", "balance", "position")
                )
                if insufficient:
                    shortage_retry_count += 1
                    print("[MAKER][SELL] 下单失败，疑似仓位不足，等待60s后刷新仓位。")
                    sleep_fn(60)
                    refreshed_goal: Optional[float] = None
                    refreshed_remaining: Optional[float] = None
                    live_target: Optional[float] = None
                    if position_fetcher:
                        try:
                            live_position = position_fetcher()
                        except Exception as fetch_exc:
                            print(f"[MAKER][SELL] 仓位刷新失败：{fetch_exc}")
                            live_position = None
                        if live_position is not None:
                            try:
                                live_target = max(
                                    _floor_to_dp(float(live_position), SELL_SIZE_DP), 0.0
                                )
                            except (TypeError, ValueError):
                                live_target = None
                    if live_target is None:
                        missing_position_retry += 1
                        if missing_position_retry >= 5:
                            final_status = "FAILED"
                            print("[MAKER][SELL] 无法获取最新仓位，退出卖出流程。")
                            break
                        print(
                            "[MAKER][SELL] 无法获取最新仓位，等待60s后重试同步。 "
                            f"(attempt {missing_position_retry}/5)"
                        )
                        sleep_fn(60)
                        continue
                    missing_position_retry = 0

                    dust_cutoff = 0.01
                    if api_min_qty and api_min_qty > dust_cutoff:
                        dust_cutoff = api_min_qty
                    if live_target + _MIN_FILL_EPS < dust_cutoff:
                        final_status = (
                            "FILLED_TRUNCATED" if filled_total > _MIN_FILL_EPS else "SKIPPED_TOO_SMALL"
                        )
                        remaining = max(goal_size - filled_total, 0.0)
                        print("[MAKER][SELL] 仓位已为0或仅剩尘埃，结束卖出流程。")
                        break

                    refreshed_goal = _apply_goal_cap(max(filled_total + live_target, filled_total))
                    refreshed_remaining = max(refreshed_goal - filled_total, 0.0)
                    goal_size = refreshed_goal
                    remaining = refreshed_remaining
                    print(
                        "[MAKER][SELL] 刷新仓位后按最新可用数量重试 -> "
                        f"goal={goal_size:.{SELL_SIZE_DP}f} remain={remaining:.{SELL_SIZE_DP}f}"
                    )

                    if refreshed_remaining < 0.01 or (
                        api_min_qty and refreshed_remaining + _MIN_FILL_EPS < api_min_qty
                    ):
                        final_status = (
                            "FILLED_TRUNCATED" if filled_total > _MIN_FILL_EPS else "SKIPPED_TOO_SMALL"
                        )
                        remaining = max(goal_size - filled_total, 0.0)
                        print("[MAKER][SELL] 刷新后可卖数量不足最小挂单量，结束卖出流程。")
                        break

                    consecutive_insufficient_with_position += 1
                    if consecutive_insufficient_with_position > 10:
                        final_status = "FAILED"
                        print("[MAKER][SELL] 仓位数据接口返回数据错误，退出卖出流程。")
                        break
                    continue
                raise
            order_id = str(response.get("orderId"))
            if shortage_retry_count or consecutive_insufficient_with_position:
                shortage_retry_count = 0
                consecutive_insufficient_with_position = 0
            record = {
                "id": order_id,
                "side": "sell",
                "price": px,
                "size": qty,
                "status": "OPEN",
                "filled": 0.0,
            }
            orders.append(record)
            records[order_id] = record
            accounted[order_id] = 0.0
            active_order = order_id
            active_price = px
            if aggressive_mode:
                if px <= floor_float + 1e-12:
                    aggressive_locked_price = floor_float
                    aggressive_floor_locked = True
                    aggressive_timer_start = None
                    aggressive_timer_anchor_fill = 0.0
                else:
                    aggressive_locked_price = None
                    aggressive_floor_locked = False
                    aggressive_timer_start = time.time()
                    aggressive_timer_anchor_fill = 0.0
            print(
                f"[MAKER][SELL] 挂单 -> price={px:.{price_dp}f} qty={qty:.{SELL_SIZE_DP}f} remaining={remaining:.{SELL_SIZE_DP}f}"
            )
            if progress_probe:
                interval = max(progress_probe_interval, poll_sec, 1e-6)
                try:
                    progress_probe()
                except Exception as probe_exc:
                    print(f"[MAKER][SELL] 进度探针执行异常：{probe_exc}")
                next_probe_at = time.time() + interval
            continue

        sleep_fn(poll_sec)
        if (
            progress_probe
            and active_order
            and progress_probe_interval > 0
            and time.time() >= max(next_probe_at, 0.0)
        ):
            try:
                progress_probe()
            except Exception as probe_exc:
                print(f"[MAKER][SELL] 进度探针执行异常：{probe_exc}")
            interval = max(progress_probe_interval, poll_sec, 1e-6)
            next_probe_at = time.time() + interval
        try:
            status_payload = adapter.get_order_status(active_order)
        except Exception as exc:
            print(f"[MAKER][SELL] 查询订单状态异常：{exc}")
            status_payload = {"status": "UNKNOWN", "filledAmount": accounted.get(active_order, 0.0)}

        record = records.get(active_order)
        status_text = str(status_payload.get("status", "UNKNOWN"))
        record_size = None
        if record is not None:
            try:
                record_size = float(record.get("size", 0.0) or 0.0)
            except Exception:
                record_size = None
        last_price_hint = active_price
        if last_price_hint is None:
            last_price_hint = _coerce_float(status_payload.get("avgPrice"))
        if last_price_hint is None:
            last_price_hint = floor_float
        filled_amount, avg_price, notional_sum = _update_fill_totals(
            active_order,
            status_payload,
            accounted,
            notional_sum,
            float(last_price_hint),
            status_text=status_text,
            expected_full_size=record_size,
        )
        filled_total = sum(accounted.values())
        remaining = max(goal_size - filled_total, 0.0)
        status_text_upper = status_text.upper()
        if record is not None:
            record["filled"] = filled_amount
            record["status"] = status_text_upper
            if avg_price is not None:
                record["avg_price"] = avg_price
            price_display = record.get("price", active_price)
            total_size = float(record.get("size", 0.0) or 0.0)
            remaining_slice = max(total_size - filled_amount, 0.0)
            if price_display is not None:
                print(
                    f"[MAKER][SELL] 挂单状态 -> price={float(price_display):.{price_dp}f} "
                    f"sold={filled_amount:.{SELL_SIZE_DP}f} remaining={remaining_slice:.{SELL_SIZE_DP}f} "
                    f"status={status_text_upper}"
                )

        if api_min_qty and remaining < api_min_qty:
            if active_order:
                _cancel_order(client, active_order)
                rec = records.get(active_order)
                if rec is not None:
                    rec["status"] = "CANCELLED"
                active_order = None
                active_price = None
                aggressive_timer_start = None
                aggressive_timer_anchor_fill = None
                aggressive_next_price_override = None
                next_price_override = None
            final_status = "FILLED_TRUNCATED" if filled_total > _MIN_FILL_EPS else "SKIPPED_TOO_SMALL"
            break

        if remaining <= 0.0 or _floor_to_dp(remaining, SELL_SIZE_DP) < 0.01:
            if active_order:
                _cancel_order(client, active_order)
                rec = records.get(active_order)
                if rec is not None:
                    rec["status"] = "CANCELLED"
                active_order = None
                aggressive_timer_start = None
                aggressive_timer_anchor_fill = None
                aggressive_next_price_override = None
                next_price_override = None
            final_status = "FILLED"
            break

        ask_info = _best_price_info(client, token_id, best_ask_fn, "ask")
        ask = ask_info.price if ask_info is not None else None
        if ask_info and ask_info.decimals is not None:
            detected_dp = _normalize_price_dp(ask_info.decimals)
            if detected_dp != price_dp:
                price_dp = detected_dp
                tick = _order_tick(price_dp)
                floor_float = _round_up_to_dp(floor_float, price_dp)
        if not aggressive_mode:
            if ask is None:
                continue
            if ask < floor_float - 1e-12:
                print(
                    f"[MAKER][SELL] 卖一再次跌破地板，撤单等待 | ask={ask:.{price_dp}f} floor={floor_float:.{price_dp}f}"
                )
                _cancel_order(client, active_order)
                rec = records.get(active_order)
                if rec is not None:
                    rec["status"] = "CANCELLED"
                active_order = None
                active_price = None
                waiting_for_floor = True
                aggressive_timer_start = None
                aggressive_timer_anchor_fill = None
                aggressive_next_price_override = None
                next_price_override = None
                continue
        else:
            if ask is None:
                continue
            if ask <= floor_float + 1e-12:
                aggressive_floor_locked = True
                aggressive_locked_price = floor_float
            elif aggressive_floor_locked and ask > floor_float + 1e-12:
                aggressive_floor_locked = False
                aggressive_locked_price = None

        if aggressive_mode and active_order:
            if aggressive_timer_anchor_fill is None:
                aggressive_timer_anchor_fill = accounted.get(active_order, 0.0)
            if aggressive_timer_start is None and not aggressive_floor_locked:
                aggressive_timer_start = time.time()
                aggressive_timer_anchor_fill = accounted.get(active_order, 0.0)
            current_filled = accounted.get(active_order, 0.0)
            if current_filled > (aggressive_timer_anchor_fill or 0.0) + _MIN_FILL_EPS:
                aggressive_timer_start = time.time()
                aggressive_timer_anchor_fill = current_filled
            if not aggressive_floor_locked and aggressive_timer_start is not None:
                elapsed = time.time() - aggressive_timer_start
                if elapsed >= aggressive_timeout and active_price is not None:
                    target_price = active_price - aggressive_step
                    if target_price <= floor_float + 1e-12:
                        aggressive_floor_locked = True
                        aggressive_locked_price = floor_float
                        aggressive_timer_start = None
                        aggressive_timer_anchor_fill = current_filled
                        if active_price > floor_float + 1e-12:
                            print(
                                "[MAKER][SELL][激进] 触及地板价，保持地板挂单"
                            )
                            _cancel_order(client, active_order)
                            rec = records.get(active_order)
                            if rec is not None:
                                rec["status"] = "CANCELLED"
                            active_order = None
                            active_price = None
                            aggressive_next_price_override = floor_float
                            next_price_override = floor_float
                        continue
                    next_px = max(
                        _round_down_to_dp(target_price, price_dp),
                        floor_float,
                    )
                    if next_px < active_price - 1e-12:
                        print(
                            "[MAKER][SELL][激进] 挂单超时未成交，下调挂价 -> "
                            f"old={active_price:.{price_dp}f} new={next_px:.{price_dp}f}"
                        )
                        _cancel_order(client, active_order)
                        rec = records.get(active_order)
                        if rec is not None:
                            rec["status"] = "CANCELLED"
                        active_order = None
                        active_price = None
                        aggressive_next_price_override = next_px
                        aggressive_timer_start = None
                        aggressive_timer_anchor_fill = current_filled
                        continue

        if active_price is not None and ask <= active_price - tick - 1e-12:
            new_px = max(_round_down_to_dp(ask, price_dp), floor_float)
            if aggressive_mode:
                if active_price <= floor_float + 1e-12:
                    continue
                if new_px <= floor_float + 1e-12:
                    aggressive_floor_locked = True
                    aggressive_locked_price = floor_float
                    if active_price <= floor_float + 1e-12:
                        continue
                    print(
                        "[MAKER][SELL][激进] 卖一跌至地板价，保持地板挂单"
                    )
                    _cancel_order(client, active_order)
                    rec = records.get(active_order)
                    if rec is not None:
                        rec["status"] = "CANCELLED"
                    active_order = None
                    active_price = None
                    aggressive_timer_start = None
                    aggressive_timer_anchor_fill = None
                    aggressive_next_price_override = floor_float
                    next_price_override = floor_float
                    continue
            print(
                f"[MAKER][SELL] 卖一下行 -> 撤单重挂 | old={active_price:.{price_dp}f} new={new_px:.{price_dp}f}"
            )
            if aggressive_mode and new_px > floor_float + 1e-12:
                aggressive_floor_locked = False
                aggressive_locked_price = None
            _cancel_order(client, active_order)
            rec = records.get(active_order)
            if rec is not None:
                rec["status"] = "CANCELLED"
            active_order = None
            active_price = None
            aggressive_timer_start = None
            aggressive_timer_anchor_fill = None
            aggressive_next_price_override = new_px if aggressive_mode else None
            next_price_override = new_px
            continue

        final_states = {"FILLED", "MATCHED", "COMPLETED", "EXECUTED"}
        cancel_states = {"CANCELLED", "CANCELED", "REJECTED", "EXPIRED"}
        if status_text_upper in final_states:
            active_order = None
            active_price = None
            aggressive_timer_start = None
            aggressive_timer_anchor_fill = None
            aggressive_next_price_override = None
            next_price_override = None
            continue
        if status_text_upper in cancel_states:
            active_order = None
            active_price = None
            aggressive_timer_start = None
            aggressive_timer_anchor_fill = None
            aggressive_next_price_override = None
            next_price_override = None
            continue

    avg_price = notional_sum / filled_total if filled_total > 0 else None
    remaining = max(goal_size - filled_total, 0.0)
    return {
        "status": final_status,
        "avg_price": avg_price,
        "filled": filled_total,
        "remaining": remaining,
        "orders": orders,
    }
