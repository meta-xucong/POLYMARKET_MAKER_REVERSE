"""
多领域批量盘口扫描与价格历史拉取工具。

功能概述：
- 通过 Gamma API 获取体育/电竞等领域的 tag，按领域筛选盘口。
- 支持配置时间范围（默认近 7 天），同时获取进行中与已结束盘口。
- 可按成交量/流动性过滤盘口，提取盘口元数据。
- 调用 CLOB prices-history 接口，为每个盘口的每个 outcome 拉取完整价格曲线。
- 结果可导出为 CSV，方便后续分析。

使用示例：
    python poly_domain_scan.py --domain sports_all --days 7 --min-volume 5000 \
        --fidelity 60 --output markets_history.csv

    python3 /home/trader/polymarket_api/poly_fliter_check/poly_domain_scan.py \
        --domain sports_all \
        --sports NBA \
        --days 1 \
        --min-volume 50000 \
        --fidelity 60 \
        --output /home/trader/polymarket_api/poly_fliter_check/markets_history.csv

    # 电竞 DOTA2 示例（参数与上例一致，仅切换领域与标签线索）
    python3 /home/trader/polymarket_api/poly_fliter_check/poly_domain_scan.py \
        --domain esports \
        --esports-tag-hints DOTA2 \
        --days 1 \
        --min-volume 50000 \
        --fidelity 60 \
        --output /home/trader/polymarket_api/poly_fliter_check/markets_history.csv

参数说明：
- domain：预设领域（sports_all 表示所有体育；esports 表示电竞；custom 需配合 --tags）。
- sports：当 domain=sports_all 时可用逗号分隔指定具体体育种类（NBA,CBB,Soccer 等）。
- tags：自定义 tag id/slug/label，逗号分隔，优先级最高。
- days：未显式传入开始/结束日期时，默认抓取近 N 天。
- start-date / end-date：YYYY-MM-DD 格式，覆盖 days 设置。
- min-volume：最小成交量（USD），过滤低流动性盘口；默认 0 不限制。
- fidelity：prices-history 请求的精度（分钟），数值越小说明采样越密集。
- output：输出 CSV 路径，默认 markets_history.csv。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

GAMMA_HOST = os.environ.get("GAMMA_HOST", "https://gamma-api.polymarket.com").rstrip("/")
CLOB_HOST = os.environ.get("CLOB_HOST", "https://clob.polymarket.com").rstrip("/")


@dataclass
class MarketRecord:
    event_slug: str
    event_question: str
    market_id: str
    market_slug: str
    market_question: str
    market_type: str
    outcomes: List[str]
    token_ids: List[str]
    created_at: Optional[str]
    closed_time: Optional[str]
    game_start_time: Optional[str]
    volume: float


@dataclass
class PricePoint:
    event_slug: str
    market_id: str
    market_slug: str
    outcome: str
    timestamp: int
    price_raw: float
    price_prob: float


def _fetch_json(url: str, params: Optional[Dict] = None, *, retries: int = 3, backoff: float = 2.0) -> Dict:
    import requests

    attempt = 1
    while True:
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt >= retries:
                raise RuntimeError(f"请求失败：{url} -> {exc}") from exc

            wait = backoff ** (attempt - 1)
            print(
                f"[WARN] 请求失败（{attempt}/{retries}）：{url} -> {exc}，{wait:.1f}s 后重试…",
                flush=True,
            )
            time.sleep(wait)
            attempt += 1


def _parse_date(text: str) -> dt.datetime:
    return dt.datetime.strptime(text, "%Y-%m-%d")


def _datetime_to_ts(value: dt.datetime | None) -> int:
    if value is None:
        return 0
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return int(value.timestamp())


def _to_unix(ts: str) -> Optional[int]:
    if not ts:
        return None
    try:
        return int(dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _parse_token_ids(raw: str | Sequence[str] | None) -> List[str]:
    """解析 clobTokenIds 字段，兼容列表/JSON 字符串/分隔字符串，并尽量去除包裹符号。"""

    if raw is None:
        return []

    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []

        # 优先 JSON 解析；若为单引号包裹也做一次转换
        for candidate in (text, text.replace("'", '"')):
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list):
                    return [str(x).strip() for x in parsed if str(x).strip()]
            except json.JSONDecodeError:
                pass

        # 去掉外层括号/中括号后按常见分隔符拆分
        trimmed = re.sub(r"^[\[(]+|[\])]+$", "", text)
        if "|" in trimmed or "," in trimmed:
            tokens = [seg.strip() for seg in re.split(r"[|,]", trimmed) if seg.strip()]
        else:
            tokens = [trimmed] if trimmed else []

        # 如果仍然只有一个字符串，尝试从其中提取类似 token 的片段
        if len(tokens) == 1 and (" " in tokens[0] or "\"" in tokens[0]):
            candidates = re.findall(r"[A-Za-z0-9]{20,}", tokens[0])
            tokens = candidates or tokens

        return tokens

    return [str(x).strip() for x in raw if str(x).strip()]


def _extract_token_ids_from_tokens(raw_tokens: Sequence[Dict] | None) -> List[str]:
    """从 tokens 字段提取 tokenId/uid，作为 clobTokenIds 的兜底。"""

    if not raw_tokens:
        return []

    token_ids: List[str] = []
    for item in raw_tokens:
        if isinstance(item, dict):
            token_id = item.get("tokenId") or item.get("id") or item.get("uid")
            if token_id:
                token_ids.append(str(token_id).strip())
    return token_ids


def _normalize_outcomes(raw_outcomes: Sequence[str] | str | None) -> List[str]:
    """清洗 outcome 字段，兼容字符串/列表输入并移除噪声符号。"""

    if raw_outcomes is None:
        return []

    # 先将输入标准化为列表
    if isinstance(raw_outcomes, str):
        text = raw_outcomes.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                candidates = parsed
            else:
                candidates = [text]
        except json.JSONDecodeError:
            # 去掉方括号后再尝试按 | 或 , 分割
            no_brackets = re.sub(r"[\[\]]", "", text)
            if "|" in no_brackets or "," in no_brackets:
                candidates = re.split(r"[|,]", no_brackets)
            else:
                candidates = [no_brackets]
    else:
        candidates = list(raw_outcomes)

    cleaned: List[str] = []
    for o in candidates:
        text = str(o or "")
        # 去除管道、引号与多余空白
        text = re.sub(r"[\"'|]", "", text).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _extract_slug_from_url(url: str | None) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    return parts[-1] if parts else ""


def _safe_float(value: str | float | int | None) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _normalize_tag_key(text: str) -> str:
    """统一标签键的格式，去除非字母数字字符并转为小写，便于模糊匹配。"""

    return re.sub(r"[^a-z0-9]", "", text.lower())


class DomainResolver:
    GENERIC_SPORT_TAGS = {1, 100639}

    def __init__(self, esports_tag_hints: Optional[List[str]] = None) -> None:
        self.esports_tag_hints = esports_tag_hints or []
        self._sports_tags: Dict[str, List[int]] = {}
        self._tag_lookup: Dict[str, int] = {}

    def load_metadata(self) -> None:
        sports = _fetch_json(f"{GAMMA_HOST}/sports") or []
        for item in sports:
            sport_name = str(item.get("sport") or "").lower()
            tag_ids = []
            tags_raw = item.get("tags")
            if isinstance(tags_raw, str):
                tag_ids = [int(t) for t in tags_raw.split(",") if t]
            elif isinstance(tags_raw, list):
                tag_ids = [int(t) for t in tags_raw if t is not None]
            if sport_name:
                self._sports_tags[sport_name] = tag_ids

        tags_resp = _fetch_json(f"{GAMMA_HOST}/tags") or []
        tags_list: Iterable[Dict] = tags_resp if isinstance(tags_resp, list) else tags_resp.get("tags", [])
        for tag in tags_list:
            tag_id = tag.get("id")
            label = str(tag.get("label") or tag.get("slug") or tag.get("name") or "").lower()
            if tag_id is not None and label:
                tag_id_int = int(tag_id)
                norm_label = _normalize_tag_key(label)
                self._tag_lookup[label] = tag_id_int
                if norm_label:
                    self._tag_lookup.setdefault(norm_label, tag_id_int)

    def resolve(self, domain: str, sports: Optional[List[str]] = None, custom_tags: Optional[List[str]] = None) -> List[int]:
        domain = domain.lower()
        sports = [s.lower() for s in sports or []]
        custom_tags = [t.lower() for t in custom_tags or []]

        tag_ids: List[int] = []
        if domain == "custom" and custom_tags:
            for tag in custom_tags:
                normalized = _normalize_tag_key(tag)
                if tag.isdigit():
                    tag_ids.append(int(tag))
                elif normalized in self._tag_lookup:
                    tag_ids.append(self._tag_lookup[normalized])
        elif domain == "esports":
            for hint in self.esports_tag_hints:
                normalized = _normalize_tag_key(hint)
                if hint.isdigit():
                    tag_ids.append(int(hint))
                elif normalized in self._tag_lookup:
                    tag_ids.append(self._tag_lookup[normalized])
        else:
            # sports_all 或者指定体育种类
            if sports:
                for sp in sports:
                    ids = self._sports_tags.get(sp, [])
                    filtered = [t for t in ids if t not in self.GENERIC_SPORT_TAGS]
                    tag_ids.extend(filtered)
            else:
                for ids in self._sports_tags.values():
                    tag_ids.extend(ids)
        # 去重
        return sorted(set(tag_ids))


def fetch_markets(
    tag_ids: List[int],
    start_date: dt.datetime,
    end_date: dt.datetime,
    sports_market_types: Optional[List[str]] = None,
) -> List[Dict]:
    if not tag_ids:
        return []
    markets: List[Dict] = []
    params_base = {
        "start_date_min": start_date.strftime("%Y-%m-%d"),
        "start_date_max": end_date.strftime("%Y-%m-%d"),
    }
    if sports_market_types:
        params_base["sports_market_types"] = ",".join(sports_market_types)

    for tag in tag_ids:
        offset = 0
        while True:
            params = dict(params_base)
            params.update({"tag_id": tag, "limit": 100, "offset": offset})
            data = _fetch_json(f"{GAMMA_HOST}/markets", params=params)
            batch: Iterable[Dict] = data if isinstance(data, list) else data.get("markets", [])
            batch_list = list(batch)
            markets.extend(batch_list)
            if len(batch_list) < 100:
                break
            offset += 100
    return markets


def build_market_records(markets: List[Dict], min_volume: float) -> List[MarketRecord]:
    records: List[MarketRecord] = []
    for mk in markets:
        if not mk.get("enableOrderBook", True):
            continue
        volume = float(mk.get("volume") or mk.get("volume24hr") or 0)
        if volume < min_volume:
            continue

        market_slug = str(mk.get("slug") or mk.get("marketSlug") or mk.get("market_slug") or "").strip()
        market_id = str(mk.get("id") or mk.get("_id") or "").strip()
        if not market_slug:
            # 用 id 兜底，避免空 slug 导致下游无法关联
            market_slug = market_id

        outcomes_raw = mk.get("outcomes") or mk.get("shortOutcomes") or []
        outcomes = _normalize_outcomes(outcomes_raw)
        token_ids = _parse_token_ids(mk.get("clobTokenIds") or mk.get("clob_token_ids") or mk.get("clobTokens"))

        # 尝试从 tokens 字段兜底补齐缺失的 token_ids
        if len(token_ids) < len(outcomes):
            token_fallback = _extract_token_ids_from_tokens(mk.get("tokens"))
            if token_fallback:
                # 仅在 fallback 能提供更多 token 时合并
                token_ids = token_ids + [tid for tid in token_fallback if tid not in token_ids]

        if not market_slug or not outcomes or not token_ids:
            print(
                f"[WARN] 跳过盘口（缺少关键信息）：slug={market_slug!r}, outcomes={outcomes}, tokens={token_ids}"
            )
            continue

        # 对齐 outcomes 与 token_ids，长度不一致时尝试补齐，仍失败则跳过以避免缺失 outcome 价格
        if len(outcomes) != len(token_ids):
            print(
                f"[WARN] outcomes 与 token_ids 长度不一致，将跳过：slug={market_slug!r}, outcomes={len(outcomes)}, tokens={len(token_ids)}"
            )
            continue

        event_slug = str(
            mk.get("eventSlug")
            or mk.get("event_slug")
            or _extract_slug_from_url(mk.get("eventUrl") or mk.get("event_url"))
            or ""
        )
        event_question = str(mk.get("eventQuestion") or mk.get("eventTitle") or mk.get("event") or "")
        records.append(
            MarketRecord(
                event_slug=event_slug,
                event_question=event_question,
                market_id=market_id,
                market_slug=market_slug,
                market_question=str(mk.get("question") or mk.get("title") or ""),
                market_type=str(mk.get("sportsMarketType") or mk.get("marketType") or mk.get("type") or ""),
                outcomes=outcomes,
                token_ids=token_ids,
                created_at=mk.get("createdAt"),
                closed_time=mk.get("closedTime") or mk.get("endDate"),
                game_start_time=mk.get("gameStartTime"),
                volume=volume,
            )
        )
    return records


def fetch_price_history(token_id: str, start_ts: int, end_ts: int, fidelity: int) -> List[Tuple[int, float]]:
    slice_seconds = 7 * 86400
    window_start = start_ts
    slice_index = 1
    result: List[Tuple[int, float]] = []

    while window_start < end_ts:
        window_end = min(window_start + slice_seconds, end_ts)
        print(
            f"[INFO] 价格切片 {slice_index} 开始：token={token_id}, "
            f"{dt.datetime.utcfromtimestamp(window_start)} -> {dt.datetime.utcfromtimestamp(window_end)}",
            flush=True,
        )
        params = {"market": token_id, "startTs": window_start, "endTs": window_end, "fidelity": fidelity}
        data = _fetch_json(f"{CLOB_HOST}/prices-history", params=params)
        history = data.get("history") or []

        for point in history:
            t = point.get("t")
            p = point.get("p")
            if t is None or p is None:
                continue
            try:
                result.append((int(t), float(p)))
            except (TypeError, ValueError):
                continue

        print(
            f"[INFO] 价格切片 {slice_index} 完成：token={token_id}, 数据点={len(history)}",
            flush=True,
        )
        slice_index += 1
        window_start = window_end
        if window_start < end_ts:
            time.sleep(1)

    return result


def _determine_price_window(record: MarketRecord, default_start_ts: int, default_end_ts: int) -> Tuple[int, int]:
    """为价格抓取确定统一的起止时间，优先使用命令行指定的时间范围。"""

    now_ts = int(time.time())
    start_ts = default_start_ts or 0
    end_ts = default_end_ts or now_ts

    # 若未显式指定时间范围，再退回到盘口元数据的创建/关闭时间
    if not start_ts:
        start_ts = min(_to_unix(record.created_at) or now_ts - 7 * 86400, now_ts)
    if not end_ts:
        end_ts = min(_to_unix(record.closed_time) or now_ts, now_ts)
    return start_ts, end_ts


def collect_price_points(
    record: MarketRecord, fidelity: int, start_ts: int, end_ts: int
) -> List[PricePoint]:
    start_ts, end_ts = _determine_price_window(record, start_ts, end_ts)
    if end_ts <= start_ts:
        print(
            f"[WARN] 跳过价格拉取（时间范围异常）：slug={record.market_slug!r}, start={start_ts}, end={end_ts}"
        )
        return []
    if not record.market_slug:
        print(f"[WARN] 跳过价格拉取（缺少 market_slug）")
        return []

    points: List[PricePoint] = []
    missing_tokens: List[str] = []
    for idx, (outcome, token_id) in enumerate(zip(record.outcomes, record.token_ids), start=1):
        if not token_id:
            missing_tokens.append(outcome)
            continue

        print(
            f"[INFO] 拉取盘口价格：slug={record.market_slug!r}, outcome={outcome!r}, "
            f"第 {idx}/{len(record.token_ids)} 个 outcome",
            flush=True,
        )
        history = fetch_price_history(token_id, start_ts, end_ts, fidelity)
        if not history:
            print(f"[WARN] 未获取到价格历史：slug={record.market_slug!r}, token={token_id}")
            continue

        for t, p in history:
            price_prob = p / 10000 if p > 1 else p / 100
            points.append(
                PricePoint(
                    event_slug=record.event_slug,
                    market_id=record.market_id,
                    market_slug=record.market_slug,
                    outcome=outcome,
                    timestamp=t,
                    price_raw=p,
                    price_prob=price_prob,
                )
            )
        time.sleep(0.1)

    if missing_tokens:
        print(
            f"[WARN] 盘口缺少 token_id，未能抓取所有 outcome：slug={record.market_slug!r}, 缺失={missing_tokens}"
        )
    expected = len(record.outcomes)
    actual = len({pt.outcome for pt in points})
    if expected and actual and actual < expected:
        print(
            f"[WARN] 盘口价格历史不完整：slug={record.market_slug!r}, outcomes={expected}, fetched={actual}"
        )
    return points


def load_market_records_from_csv(path: str) -> List[MarketRecord]:
    records: List[MarketRecord] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            outcomes = _normalize_outcomes(row.get("outcomes"))
            token_ids = _parse_token_ids(row.get("token_ids"))
            records.append(
                MarketRecord(
                    event_slug=row.get("event_slug", ""),
                    event_question=row.get("event_question", ""),
                    market_id=row.get("market_id", ""),
                    market_slug=row.get("market_slug", ""),
                    market_question=row.get("market_question", ""),
                    market_type=row.get("market_type", ""),
                    outcomes=outcomes,
                    token_ids=token_ids,
                    created_at=row.get("created_at"),
                    closed_time=row.get("closed_time"),
                    game_start_time=row.get("game_start_time"),
                    volume=_safe_float(row.get("volume")),
                )
            )
    return records


def load_price_points_from_csv(path: str) -> List[PricePoint]:
    points: List[PricePoint] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                timestamp = int(row.get("timestamp", "0") or 0)
            except ValueError:
                timestamp = 0
            points.append(
                PricePoint(
                    event_slug=row.get("event_slug", ""),
                    market_id=row.get("market_id", ""),
                    market_slug=row.get("market_slug", ""),
                    outcome=row.get("outcome", ""),
                    timestamp=timestamp,
                    price_raw=_safe_float(row.get("price_raw")),
                    price_prob=_safe_float(row.get("price_prob")),
                )
            )
    return points


def export_wide_prices(
    records: List[MarketRecord], prices: List[PricePoint], output_path: str
) -> str:
    """将价格历史按盘口与时间戳合并为宽表，每个 outcome 拆为单独列。"""

    if not records or not prices:
        raise SystemExit("记录或价格列表为空，无法生成宽表。")

    record_map: Dict[str, MarketRecord] = {rec.market_slug: rec for rec in records}
    max_outcomes = max((len(rec.outcomes) for rec in records), default=0)
    header = ["event_slug", "market_slug", "market_question", "market_type", "timestamp"]
    for idx in range(max_outcomes):
        header.extend([f"outcome_{idx + 1}_name", f"outcome_{idx + 1}_prob"])

    grouped: Dict[Tuple[str, int], Dict[str, float]] = {}
    for pt in prices:
        key = (pt.market_slug, pt.timestamp)
        bucket = grouped.setdefault(key, {})
        bucket[pt.outcome] = pt.price_prob

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for (market_slug, ts), outcome_map in sorted(grouped.items()):
            rec = record_map.get(market_slug)
            if not rec:
                continue
            row = [rec.event_slug, market_slug, rec.market_question, rec.market_type, ts]
            for name in rec.outcomes:
                row.extend([name, outcome_map.get(name)])
            # 填满剩余 outcome 列（有的盘口只有两个 outcome）
            remaining = max_outcomes - len(rec.outcomes)
            for _ in range(remaining):
                row.extend(["", ""])
            writer.writerow(row)

    return output_path


def export_csv(records: List[MarketRecord], prices: List[PricePoint], path: str) -> Tuple[str, str]:
    root, ext = os.path.splitext(path)
    ext = ext or ".csv"
    markets_path = path
    prices_path = f"{root}_prices{ext}"

    with open(markets_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "event_slug",
                "event_question",
                "market_id",
                "market_slug",
                "market_question",
                "market_type",
                "outcomes",
                "token_ids",
                "created_at",
                "closed_time",
                "game_start_time",
                "volume",
            ]
        )
        for rec in records:
            writer.writerow(
                [
                    rec.event_slug,
                    rec.event_question,
                    rec.market_id,
                    rec.market_slug,
                    rec.market_question,
                    rec.market_type,
                    "|".join(rec.outcomes),
                    "|".join(rec.token_ids),
                    rec.created_at or "",
                    rec.closed_time or "",
                    rec.game_start_time or "",
                    rec.volume,
                ]
            )

    with open(prices_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["event_slug", "market_id", "market_slug", "outcome", "timestamp", "price_raw", "price_prob"]
        )
        for pt in prices:
            writer.writerow(
                [
                    pt.event_slug,
                    pt.market_id,
                    pt.market_slug,
                    pt.outcome,
                    pt.timestamp,
                    pt.price_raw,
                    pt.price_prob,
                ]
            )

    return markets_path, prices_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按领域批量抓取 Polymarket 盘口和价格历史")
    parser.add_argument("--domain", default="sports_all", help="sports_all / esports / custom")
    parser.add_argument("--sports", help="限定体育种类，逗号分隔，例如 NBA,Soccer")
    parser.add_argument("--tags", help="自定义 tag id/slug/label，逗号分隔，仅 domain=custom 生效")
    parser.add_argument("--days", type=int, default=7, help="未指定日期范围时，默认抓取近 N 天")
    parser.add_argument("--start-date", dest="start_date", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", dest="end_date", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--min-volume", type=float, default=0.0, help="最小成交量 USD，过滤盘口")
    parser.add_argument("--fidelity", type=int, default=60, help="prices-history 的采样精度（分钟）")
    parser.add_argument("--output", default="markets_history.csv", help="输出 CSV 路径")
    parser.add_argument(
        "--sports-market-types",
        dest="sports_market_types",
        help="限定体育盘口类型，逗号分隔，例如 moneyline,spread,totals",
    )
    parser.add_argument(
        "--esports-tag-hints",
        dest="esports_tag_hints",
        help="电竞标签线索（id/slug/label），逗号分隔，用于 domain=esports",
    )
    parser.add_argument(
        "--markets-csv",
        dest="markets_csv",
        help="使用本地 markets_history.csv，跳过 API 抓取",
    )
    parser.add_argument(
        "--prices-csv",
        dest="prices_csv",
        help="使用本地 markets_history_prices.csv，跳过价格抓取",
    )
    parser.add_argument(
        "--wide-output",
        dest="wide_output",
        help="输出宽表路径（按盘口合并 outcome 列），例如 merged_prices.csv",
    )
    return parser.parse_args()


def determine_date_range(args: argparse.Namespace) -> Tuple[dt.datetime, dt.datetime]:
    if args.start_date and args.end_date:
        start = _parse_date(args.start_date)
        end = _parse_date(args.end_date)
    else:
        end = dt.datetime.now(dt.UTC)
        start = end - dt.timedelta(days=args.days)
    return start, end


def main() -> None:
    args = parse_args()
    # 支持直接使用已有 CSV，方便快速合并为宽表
    if args.markets_csv and args.prices_csv:
        records = load_market_records_from_csv(args.markets_csv)
        price_points = load_price_points_from_csv(args.prices_csv)
        markets_path = args.markets_csv
        prices_path = args.prices_csv
        print(f"已加载本地数据：{markets_path} 与 {prices_path}")
    else:
        start_date, end_date = determine_date_range(args)
        start_ts = _datetime_to_ts(start_date)
        end_ts = _datetime_to_ts(end_date)

        resolver = DomainResolver(
            esports_tag_hints=args.esports_tag_hints.split(",") if args.esports_tag_hints else None
        )
        resolver.load_metadata()
        sports_list = args.sports.split(",") if args.sports else None
        custom_tags = args.tags.split(",") if args.tags else None
        tag_ids = resolver.resolve(args.domain, sports=sports_list, custom_tags=custom_tags)
        if not tag_ids:
            raise SystemExit("未能解析到有效的 tag id，请检查 domain/sports/tags 设置。")

        sports_market_types = (
            [t.strip() for t in args.sports_market_types.split(",") if t.strip()]
            if args.sports_market_types
            else None
        )
        markets_raw = fetch_markets(tag_ids, start_date, end_date, sports_market_types=sports_market_types)
        records = build_market_records(markets_raw, min_volume=args.min_volume)
        print(f"共获取盘口 {len(records)} 个，准备拉取价格历史…")

        price_points: List[PricePoint] = []
        total = len(records)
        for rec_idx, rec in enumerate(records, start=1):
            print(
                f"[INFO] 处理盘口 {rec_idx}/{total}：slug={rec.market_slug!r} ({rec.market_question})",
                flush=True,
            )
            price_points.extend(
                collect_price_points(
                    rec,
                    fidelity=args.fidelity,
                    start_ts=start_ts,
                    end_ts=end_ts,
                )
            )

        markets_path, prices_path = export_csv(records, price_points, args.output)
        print(f"完成，盘口信息写入 {markets_path}，价格历史写入 {prices_path}")

    if args.wide_output:
        wide_path = export_wide_prices(records, price_points, args.wide_output)
        print(f"宽表已写入 {wide_path}")


if __name__ == "__main__":
    main()
