"""
poly_maker_autorun
-------------------

基础骨架：配置加载、主循环、命令/交互入口。
后续步骤将补充筛选对接、历史去重、子进程调度等能力。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import queue
import select
import signal
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import Customize_fliter_blacklist as filter_script

# =====================
# 配置与常量
# =====================
PROJECT_ROOT = Path(__file__).resolve().parent
MAKER_ROOT = PROJECT_ROOT / "POLYMARKET_MAKER"

DEFAULT_GLOBAL_CONFIG = {
    "topics_poll_sec": 10.0,
    "command_poll_sec": 1.0,
    "max_concurrent_tasks": 2,
    "log_dir": str(MAKER_ROOT / "logs" / "autorun"),
    "data_dir": str(MAKER_ROOT / "data"),
    "handled_topics_path": str(MAKER_ROOT / "data" / "handled_topics.json"),
    "filter_output_path": str(MAKER_ROOT / "data" / "topics_filtered.json"),
    "filter_params_path": str(MAKER_ROOT / "config" / "filter_params.json"),
    "filter_timeout_sec": None,
    "filter_max_retries": 1,
    "filter_retry_delay_sec": 3.0,
    "process_start_retries": 1,
    "process_retry_delay_sec": 2.0,
    "process_graceful_timeout_sec": 5.0,
    "process_stagger_max_sec": 0.5,
    "runtime_status_path": str(MAKER_ROOT / "data" / "autorun_status.json"),
}

FILTER_CONFIG_RELOAD_INTERVAL_SEC = 3600
ORDER_SIZE_DECIMALS = 4  # Polymarket 下单数量精度（按买单精度取整）


def _topic_id_from_entry(entry: Any) -> str:
    """从筛选结果条目中提取 topic_id/slug，兼容字符串或 dict。"""

    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict):
        return str(entry.get("slug") or entry.get("topic_id") or "").strip()
    return str(entry).strip()


def _safe_topic_filename(topic_id: str) -> str:
    return topic_id.replace("/", "_").replace("\\", "_")


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):
            raw = value.replace(",", "").strip()
            if not raw:
                return None
            return float(raw)
    except Exception:
        return None
    return None


def _ceil_to_precision(value: float, decimals: int) -> float:
    factor = 10 ** decimals
    return math.ceil(value * factor - 1e-12) / factor


def _scale_order_size_by_volume(
    base_size: float,
    total_volume: float,
    *,
    base_volume: Optional[float] = None,
    growth_factor: float = 0.5,
    decimals: int = ORDER_SIZE_DECIMALS,
) -> float:
    """根据市场成交量对基础下单份数进行递增（边际递减）。"""

    if base_size <= 0 or total_volume <= 0:
        return base_size

    effective_base_volume = _coerce_float(base_volume) or total_volume
    if effective_base_volume <= 0:
        return base_size

    effective_growth = max(growth_factor, 0.0)
    vol_ratio = max(total_volume / effective_base_volume, 1.0)
    # 使用对数增长控制放大：
    #   - base_volume 附近仅有轻微提升；
    #   - 成交量每提升 10 倍仅线性增加 growth_factor，边际效用递减。
    weight = 1.0 + effective_growth * math.log10(vol_ratio)
    weighted_size = base_size * weight
    return _ceil_to_precision(weighted_size, decimals)


def _load_json_file(path: Path) -> Dict[str, Any]:
    """读取 JSON 配置，不存在则返回空 dict。"""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError as exc:  # pragma: no cover - 粗略校验
            raise RuntimeError(f"无法解析 JSON 配置: {path}: {exc}") from exc


def _dump_json_file(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_handled_topics(path: Path) -> set[str]:
    """读取历史已处理话题集合，空文件或字段缺失则返回空集合。"""

    data = _load_json_file(path)
    topics = data.get("topics") or data.get("handled_topics")
    if topics is None:
        return set()
    if not isinstance(topics, list):  # pragma: no cover - 容错
        print(f"[WARN] handled_topics 文件格式异常，已忽略: {path}")
        return set()
    return {str(t) for t in topics}


def write_handled_topics(path: Path, topics: set[str]) -> None:
    """写入最新的已处理话题集合。"""

    payload = {
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total": len(topics),
        "topics": sorted(topics),
    }
    _dump_json_file(path, payload)


def compute_new_topics(latest: List[Any], handled: set[str]) -> List[str]:
    """从最新筛选结果中筛出尚未处理的话题列表。"""

    result: List[str] = []
    for entry in latest:
        topic_id = _topic_id_from_entry(entry)
        if topic_id and topic_id not in handled:
            result.append(topic_id)
    return result


@dataclass
class GlobalConfig:
    topics_poll_sec: float = DEFAULT_GLOBAL_CONFIG["topics_poll_sec"]
    command_poll_sec: float = DEFAULT_GLOBAL_CONFIG["command_poll_sec"]
    max_concurrent_tasks: int = DEFAULT_GLOBAL_CONFIG["max_concurrent_tasks"]
    log_dir: Path = field(default_factory=lambda: Path(DEFAULT_GLOBAL_CONFIG["log_dir"]))
    data_dir: Path = field(default_factory=lambda: Path(DEFAULT_GLOBAL_CONFIG["data_dir"]))
    handled_topics_path: Path = field(
        default_factory=lambda: Path(DEFAULT_GLOBAL_CONFIG["handled_topics_path"])
    )
    filter_output_path: Path = field(
        default_factory=lambda: Path(DEFAULT_GLOBAL_CONFIG["filter_output_path"])
    )
    filter_params_path: Path = field(
        default_factory=lambda: Path(DEFAULT_GLOBAL_CONFIG["filter_params_path"])
    )
    filter_timeout_sec: Optional[float] = DEFAULT_GLOBAL_CONFIG["filter_timeout_sec"]
    filter_max_retries: int = DEFAULT_GLOBAL_CONFIG["filter_max_retries"]
    filter_retry_delay_sec: float = DEFAULT_GLOBAL_CONFIG["filter_retry_delay_sec"]
    process_start_retries: int = DEFAULT_GLOBAL_CONFIG["process_start_retries"]
    process_retry_delay_sec: float = DEFAULT_GLOBAL_CONFIG["process_retry_delay_sec"]
    process_graceful_timeout_sec: float = DEFAULT_GLOBAL_CONFIG[
        "process_graceful_timeout_sec"
    ]
    process_stagger_max_sec: float = DEFAULT_GLOBAL_CONFIG["process_stagger_max_sec"]
    runtime_status_path: Path = field(
        default_factory=lambda: Path(DEFAULT_GLOBAL_CONFIG["runtime_status_path"])
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GlobalConfig":
        data = data or {}
        scheduler = data.get("scheduler") or {}
        paths = data.get("paths") or {}
        flat_overrides = {k: v for k, v in data.items() if k not in {"scheduler", "paths"}}
        merged = {**DEFAULT_GLOBAL_CONFIG, **flat_overrides}

        log_dir = Path(
            paths.get("log_directory")
            or merged.get("log_dir", DEFAULT_GLOBAL_CONFIG["log_dir"])
        )
        data_dir = Path(
            paths.get("data_directory")
            or merged.get("data_dir", DEFAULT_GLOBAL_CONFIG["data_dir"])
        )

        handled_topics_path = Path(
            merged.get("handled_topics_path")
            or paths.get("handled_topics_file")
            or data_dir / "handled_topics.json"
        )
        filter_output_path = Path(
            merged.get("filter_output_path")
            or paths.get("filter_output_file")
            or data_dir / "topics_filtered.json"
        )
        filter_params_path = Path(
            merged.get("filter_params_path")
            or paths.get("filter_params_file")
            or MAKER_ROOT / "config" / "filter_params.json"
        )
        runtime_status_path = Path(
            merged.get("runtime_status_path")
            or paths.get("run_state_file")
            or data_dir / "autorun_status.json"
        )

        return cls(
            topics_poll_sec=float(
                scheduler.get("poll_interval_seconds")
                or merged.get("topics_poll_sec", DEFAULT_GLOBAL_CONFIG["topics_poll_sec"])
            ),
            command_poll_sec=float(
                merged.get("command_poll_sec", DEFAULT_GLOBAL_CONFIG["command_poll_sec"])
            ),
            max_concurrent_tasks=int(
                scheduler.get("max_concurrent_jobs")
                or merged.get(
                    "max_concurrent_tasks", DEFAULT_GLOBAL_CONFIG["max_concurrent_tasks"]
                )
            ),
            log_dir=log_dir,
            data_dir=data_dir,
            handled_topics_path=handled_topics_path,
            filter_output_path=filter_output_path,
            filter_params_path=filter_params_path,
            filter_timeout_sec=cls._parse_timeout(
                merged.get("filter_timeout_sec", cls.filter_timeout_sec)
            ),
            filter_max_retries=int(merged.get("filter_max_retries", cls.filter_max_retries)),
            filter_retry_delay_sec=float(
                merged.get("filter_retry_delay_sec", cls.filter_retry_delay_sec)
            ),
            process_start_retries=int(
                merged.get("process_start_retries", cls.process_start_retries)
            ),
            process_retry_delay_sec=float(
                merged.get("process_retry_delay_sec", cls.process_retry_delay_sec)
            ),
            process_graceful_timeout_sec=float(
                merged.get(
                    "process_graceful_timeout_sec", cls.process_graceful_timeout_sec
                )
            ),
            process_stagger_max_sec=float(
                merged.get("process_stagger_max_sec", cls.process_stagger_max_sec)
            ),
            runtime_status_path=runtime_status_path,
        )

    @staticmethod
    def _parse_timeout(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except Exception:
            return None

    def ensure_dirs(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class HighlightConfig:
    max_hours: Optional[float] = 72.0
    min_total_volume: Optional[float] = 20000.0
    max_ask_diff: Optional[float] = 0.2

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "HighlightConfig":
        data = data or {}
        return cls(
            max_hours=data.get("max_hours", cls.max_hours),
            min_total_volume=data.get("min_total_volume", cls.min_total_volume),
            max_ask_diff=data.get("max_ask_diff", cls.max_ask_diff),
        )

    def apply_to_filter(self) -> None:
        if self.max_hours is not None:
            filter_script.HIGHLIGHT_MAX_HOURS = float(self.max_hours)
        if self.min_total_volume is not None:
            filter_script.HIGHLIGHT_MIN_TOTAL_VOLUME = float(self.min_total_volume)
        if self.max_ask_diff is not None:
            filter_script.HIGHLIGHT_MAX_ASK_DIFF = float(self.max_ask_diff)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_hours": self.max_hours,
            "min_total_volume": self.min_total_volume,
            "max_ask_diff": self.max_ask_diff,
        }


@dataclass
class ReversalConfig:
    enabled: bool = True
    p1: float = filter_script.REVERSAL_P1
    p2: float = filter_script.REVERSAL_P2
    window_hours: float = filter_script.REVERSAL_WINDOW_HOURS
    lookback_days: float = filter_script.REVERSAL_LOOKBACK_DAYS
    short_interval: str = filter_script.REVERSAL_SHORT_INTERVAL
    short_fidelity: int = filter_script.REVERSAL_SHORT_FIDELITY
    long_fidelity: int = filter_script.REVERSAL_LONG_FIDELITY

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ReversalConfig":
        data = data or {}
        return cls(
            enabled=bool(data.get("enabled", cls.enabled)),
            p1=float(data.get("p1", cls.p1)),
            p2=float(data.get("p2", cls.p2)),
            window_hours=float(data.get("window_hours", cls.window_hours)),
            lookback_days=float(data.get("lookback_days", cls.lookback_days)),
            short_interval=str(data.get("short_interval", cls.short_interval)),
            short_fidelity=int(data.get("short_fidelity", cls.short_fidelity)),
            long_fidelity=int(data.get("long_fidelity", cls.long_fidelity)),
        )

    def apply_to_filter(self) -> None:
        filter_script.REVERSAL_ENABLED = bool(self.enabled)
        filter_script.REVERSAL_P1 = float(self.p1)
        filter_script.REVERSAL_P2 = float(self.p2)
        filter_script.REVERSAL_WINDOW_HOURS = float(self.window_hours)
        filter_script.REVERSAL_LOOKBACK_DAYS = float(self.lookback_days)
        filter_script.REVERSAL_SHORT_INTERVAL = str(self.short_interval)
        filter_script.REVERSAL_SHORT_FIDELITY = int(self.short_fidelity)
        filter_script.REVERSAL_LONG_FIDELITY = int(self.long_fidelity)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "p1": self.p1,
            "p2": self.p2,
            "window_hours": self.window_hours,
            "lookback_days": self.lookback_days,
            "short_interval": self.short_interval,
            "short_fidelity": self.short_fidelity,
            "long_fidelity": self.long_fidelity,
        }


@dataclass
class FilterConfig:
    min_end_hours: float = filter_script.DEFAULT_MIN_END_HOURS
    max_end_days: int = 5
    gamma_window_days: int = 2
    gamma_min_window_hours: int = 1
    legacy_end_days: int = filter_script.DEFAULT_LEGACY_END_DAYS
    allow_illiquid: bool = False
    skip_orderbook: bool = False
    no_rest_backfill: bool = False
    books_batch_size: int = 200
    books_timeout_sec: float = 5.0
    only: str = ""
    blacklist_terms: List[str] = field(
        default_factory=lambda: list(filter_script.DEFAULT_BLACKLIST_TERMS)
    )
    highlight: HighlightConfig = field(default_factory=HighlightConfig)
    reversal: ReversalConfig = field(default_factory=ReversalConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FilterConfig":
        data = data or {}
        highlight_conf = HighlightConfig.from_dict(data.get("highlight"))
        reversal_conf = ReversalConfig.from_dict(data.get("reversal"))
        return cls(
            min_end_hours=float(data.get("min_end_hours", cls.min_end_hours)),
            max_end_days=int(data.get("max_end_days", cls.max_end_days)),
            gamma_window_days=int(data.get("gamma_window_days", cls.gamma_window_days)),
            gamma_min_window_hours=int(data.get("gamma_min_window_hours", cls.gamma_min_window_hours)),
            legacy_end_days=int(data.get("legacy_end_days", cls.legacy_end_days)),
            allow_illiquid=bool(data.get("allow_illiquid", cls.allow_illiquid)),
            skip_orderbook=bool(data.get("skip_orderbook", cls.skip_orderbook)),
            no_rest_backfill=bool(data.get("no_rest_backfill", cls.no_rest_backfill)),
            books_batch_size=int(data.get("books_batch_size", cls.books_batch_size)),
            books_timeout_sec=float(
                data.get("books_timeout_sec", cls.books_timeout_sec)
            ),
            only=str(data.get("only", cls.only)),
            blacklist_terms=[
                str(t).strip()
                for t in data.get("blacklist_terms", filter_script.DEFAULT_BLACKLIST_TERMS)
                if str(t).strip()
            ],
            highlight=highlight_conf,
            reversal=reversal_conf,
        )

    def to_filter_kwargs(self) -> Dict[str, Any]:
        return {
            "min_end_hours": self.min_end_hours,
            "max_end_days": self.max_end_days,
            "gamma_window_days": self.gamma_window_days,
            "gamma_min_window_hours": self.gamma_min_window_hours,
            "legacy_end_days": self.legacy_end_days,
            "allow_illiquid": self.allow_illiquid,
            "skip_orderbook": self.skip_orderbook,
            "no_rest_backfill": self.no_rest_backfill,
            "books_batch_size": self.books_batch_size,
            "books_timeout": self.books_timeout_sec,
            "only": self.only,
            "blacklist_terms": self.blacklist_terms,
            "enable_reversal": self.reversal.enabled,
            "reversal_p1": self.reversal.p1,
            "reversal_p2": self.reversal.p2,
            "reversal_window_hours": self.reversal.window_hours,
            "reversal_lookback_days": self.reversal.lookback_days,
            "reversal_short_interval": self.reversal.short_interval,
            "reversal_short_fidelity": self.reversal.short_fidelity,
            "reversal_long_fidelity": self.reversal.long_fidelity,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.to_filter_kwargs(),
            "highlight": self.highlight.to_dict(),
            "reversal": self.reversal.to_dict(),
        }

    def apply_highlight(self) -> None:
        self.highlight.apply_to_filter()

    def apply_blacklist(self) -> None:
        filter_script.set_blacklist_terms(self.blacklist_terms)

    def apply_reversal(self) -> None:
        self.reversal.apply_to_filter()


@dataclass
class TopicTask:
    topic_id: str
    status: str = "pending"
    start_time: float = field(default_factory=time.time)
    last_heartbeat: Optional[float] = None
    notes: List[str] = field(default_factory=list)
    process: Optional[subprocess.Popen] = None
    log_path: Optional[Path] = None
    config_path: Optional[Path] = None
    log_excerpt: str = ""
    restart_attempts: int = 0
    no_restart: bool = False
    end_reason: Optional[str] = None

    def heartbeat(self, message: str) -> None:
        self.last_heartbeat = time.time()
        self.notes.append(message)

    def is_running(self) -> bool:
        return bool(self.process) and (self.process.poll() is None)


class AutoRunManager:
    def __init__(
        self,
        global_config: GlobalConfig,
        strategy_defaults: Dict[str, Any],
        filter_config: FilterConfig,
        run_params_template: Dict[str, Any],
    ):
        self.config = global_config
        self.strategy_defaults = strategy_defaults
        self.filter_config = filter_config
        self.run_params_template = run_params_template or {}
        self.stop_event = threading.Event()
        self.command_queue: "queue.Queue[str]" = queue.Queue()
        self.tasks: Dict[str, TopicTask] = {}
        self.latest_topics: List[Dict[str, Any]] = []
        self.topic_details: Dict[str, Dict[str, Any]] = {}
        self.handled_topics: set[str] = set()
        self.pending_topics: List[str] = []
        self._next_topics_refresh: float = 0.0
        self._next_status_dump: float = 0.0
        self._next_filter_reload: float = 0.0
        self._filter_conf_mtime: Optional[float] = None
        self.status_path = self.config.runtime_status_path

    # ========== 核心循环 ==========
    def run_loop(self) -> None:
        self.config.ensure_dirs()
        self._load_handled_topics()
        self._restore_runtime_status()
        print(f"[INIT] autorun start | poll={self.config.topics_poll_sec}s")
        try:
            while not self.stop_event.is_set():
                try:
                    now = time.time()
                    self._process_commands()
                    self._poll_tasks()
                    self._schedule_pending_topics()
                    self._maybe_reload_filter_config(now)
                    self._purge_inactive_tasks()
                    if now >= self._next_topics_refresh:
                        self._refresh_topics()
                        self._next_topics_refresh = now + self.config.topics_poll_sec
                    if now >= self._next_status_dump:
                        self._print_status()
                        self._dump_runtime_status()
                        self._next_status_dump = now + max(
                            5.0, self.config.command_poll_sec
                        )
                    time.sleep(self.config.command_poll_sec)
                except Exception as exc:  # pragma: no cover - 防御性保护
                    print(f"[ERROR] 主循环异常已捕获，将继续运行: {exc}")
                    traceback.print_exc()
                    time.sleep(max(1.0, self.config.command_poll_sec))
        finally:
            self._cleanup_all_tasks()
            self._dump_runtime_status()
            print("[DONE] autorun stopped")

    def _poll_tasks(self) -> None:
        for task in list(self.tasks.values()):
            proc = task.process
            if not proc:
                continue
            rc = proc.poll()
            if rc is None:
                task.status = "running"
                task.last_heartbeat = time.time()
                self._update_log_excerpt(task)
                if self._log_indicates_market_end(task):
                    task.status = "ended"
                    task.no_restart = True
                    task.end_reason = "market closed"
                    task.heartbeat("market end detected from log")
                    print(
                        f"[AUTO] topic={task.topic_id} 日志显示市场已结束，自动结束该话题。"
                    )
                    self._terminate_task(task, reason="market closed (auto)")
                elif self._log_indicates_missing_side(task):
                    task.status = "ended"
                    task.no_restart = True
                    task.end_reason = "missing side"
                    task.heartbeat("missing side detected from log")
                    print(
                        f"[AUTO] topic={task.topic_id} 检测到无法确定下单方向，视为话题结束，释放执行名额。"
                    )
                    self._terminate_task(task, reason="missing side (auto)")
                continue
            self._handle_process_exit(task, rc)

        self._purge_inactive_tasks()

    def _handle_process_exit(self, task: TopicTask, rc: int) -> None:
        task.process = None
        if task.status not in {"stopped", "exited", "error", "ended"}:
            task.status = "exited" if rc == 0 else "error"
        task.heartbeat(f"process finished rc={rc}")
        self._update_log_excerpt(task)

        if self._log_indicates_missing_side(task):
            task.no_restart = True
            task.status = "ended"
            task.end_reason = "missing side"
            task.heartbeat("missing side detected from log on exit")
            return

        if task.no_restart:
            return

        if rc != 0:
            max_retries = max(0, int(self.config.process_start_retries))
            if task.restart_attempts < max_retries:
                task.restart_attempts += 1
                task.status = "restarting"
                task.heartbeat(
                    f"restart attempt {task.restart_attempts}/{max_retries} after rc={rc}"
                )
                time.sleep(self.config.process_retry_delay_sec)
                if self._start_topic_process(task.topic_id):
                    return
                if task.restart_attempts < max_retries and task.topic_id not in self.pending_topics:
                    self.pending_topics.append(task.topic_id)
            task.status = "error"

    def _update_log_excerpt(self, task: TopicTask, max_bytes: int = 2000) -> None:
        if not task.log_path or not task.log_path.exists():
            task.log_excerpt = ""
            return
        try:
            with task.log_path.open("rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - max_bytes))
                data = f.read().decode("utf-8", errors="ignore")
            lines = data.strip().splitlines()
            task.log_excerpt = "\n".join(lines[-5:])
        except OSError as exc:  # pragma: no cover - 文件访问异常
            task.log_excerpt = f"<log read error: {exc}>"

    def _log_indicates_market_end(self, task: TopicTask) -> bool:
        excerpt = (task.log_excerpt or "").lower()
        if not excerpt:
            return False
        patterns = (
            "[market] 已确认市场结束",
            "[market] 市场结束",
            "[market] 达到市场截止时间",
            "[market] 收到市场关闭事件",
            "[exit] 最终状态",
        )
        return any(p.lower() in excerpt for p in patterns)

    def _log_indicates_missing_side(self, task: TopicTask) -> bool:
        excerpt = (task.log_excerpt or "").lower()
        if not excerpt:
            return False
        patterns = (
            "未提供下单方向 side，且未能从 preferred_side/highlight_sides 推断",
        )
        return any(p.lower() in excerpt for p in patterns)

    def _schedule_pending_topics(self) -> None:
        running = sum(1 for t in self.tasks.values() if t.is_running())
        while (
            self.pending_topics
            and running < max(1, int(self.config.max_concurrent_tasks))
        ):
            topic_id = self.pending_topics.pop(0)
            if topic_id in self.tasks and self.tasks[topic_id].is_running():
                continue
            try:
                started = self._start_topic_process(topic_id)
            except Exception as exc:  # pragma: no cover - 防御性保护
                print(f"[ERROR] 调度话题 {topic_id} 时异常: {exc}")
                traceback.print_exc()
                started = False
            if not started and topic_id not in self.pending_topics:
                # 启动失败时重新入队，避免话题被遗忘
                self.pending_topics.append(topic_id)
            running = sum(1 for t in self.tasks.values() if t.is_running())

    def _get_order_base_volume(self) -> Optional[float]:
        highlight_conf = getattr(self.filter_config, "highlight", None)
        base_volume = getattr(highlight_conf, "min_total_volume", None)
        base_volume = _coerce_float(base_volume)
        if base_volume is None or base_volume <= 0:
            return None
        return base_volume

    def _build_run_config(self, topic_id: str) -> Dict[str, Any]:
        base_template_raw = json.loads(json.dumps(self.run_params_template or {}))
        base_template = {k: v for k, v in base_template_raw.items() if v is not None}

        base_raw = self.strategy_defaults.get("default", {}) or {}
        base = {k: v for k, v in base_raw.items() if v is not None}

        topic_overrides_raw = (self.strategy_defaults.get("topics") or {}).get(
            topic_id, {}
        )
        topic_overrides = {
            k: v for k, v in topic_overrides_raw.items() if v is not None
        }

        merged = {**base_template, **base, **topic_overrides}

        topic_info = self.topic_details.get(topic_id, {})
        slug = topic_info.get("slug") or topic_id
        merged["market_url"] = f"https://polymarket.com/market/{slug}"
        merged["topic_id"] = topic_id

        if topic_info.get("title"):
            merged["topic_name"] = topic_info.get("title")
        if topic_info.get("yes_token"):
            merged["yes_token"] = topic_info.get("yes_token")
        if topic_info.get("no_token"):
            merged["no_token"] = topic_info.get("no_token")
        if topic_info.get("end_time"):
            merged["end_time"] = topic_info.get("end_time")

        highlight_sides = topic_info.get("highlight_sides") or []
        preferred_side = topic_info.get("preferred_side") or None
        if preferred_side is None and highlight_sides:
            preferred_side = highlight_sides[0]
        if preferred_side:
            merged["side"] = preferred_side
        if highlight_sides:
            merged["highlight_sides"] = highlight_sides

        base_order_size = _coerce_float(merged.get("order_size"))
        total_volume = _coerce_float(topic_info.get("total_volume"))
        volume_growth_factor = _coerce_float(merged.get("volume_growth_factor"))
        if base_order_size is not None and total_volume is not None:
            scaled_size = _scale_order_size_by_volume(
                base_order_size,
                total_volume,
                base_volume=self._get_order_base_volume(),
                growth_factor=volume_growth_factor
                if volume_growth_factor is not None and volume_growth_factor > 0
                else 0.5,
            )
            merged["order_size"] = scaled_size
        return merged

    def _start_topic_process(self, topic_id: str) -> bool:
        config_data = self._build_run_config(topic_id)
        cfg_path = self.config.data_dir / f"run_params_{_safe_topic_filename(topic_id)}.json"
        _dump_json_file(cfg_path, config_data)

        log_path = self.config.log_dir / f"autorun_{_safe_topic_filename(topic_id)}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            log_file = log_path.open("a", encoding="utf-8")
        except OSError as exc:  # pragma: no cover - 文件系统异常
            print(f"[ERROR] 无法创建日志文件 {log_path}: {exc}")
            return False

        max_stagger = max(0.0, float(self.config.process_stagger_max_sec))
        if max_stagger > 0:
            delay = random.uniform(0, max_stagger)
            if delay > 0:
                print(
                    f"[SCHEDULE] topic={topic_id} 启动前随机延迟 {delay:.2f}s 以错峰运行"
                )
                time.sleep(delay)

        cmd = [
            sys.executable,
            str(MAKER_ROOT / "Volatility_arbitrage_run.py"),
            str(cfg_path),
        ]
        proc: Optional[subprocess.Popen] = None
        attempts = max(1, int(self.config.process_start_retries))
        for attempt in range(1, attempts + 1):
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                log_file.close()
                break
            except Exception as exc:  # pragma: no cover - 子进程异常
                print(
                    f"[ERROR] 启动 topic={topic_id} 失败（尝试 {attempt}/{attempts}）: {exc}"
                )
                if attempt >= attempts:
                    log_file.close()
                    return False
                time.sleep(self.config.process_retry_delay_sec)

        if not proc or proc.poll() is not None:
            rc_text = proc.poll() if proc else "?"
            print(
                f"[ERROR] topic={topic_id} 启动后立即退出 rc={rc_text}，将重试"
            )
            return False

        task = self.tasks.get(topic_id) or TopicTask(topic_id=topic_id)
        task.process = proc
        task.config_path = cfg_path
        task.log_path = log_path
        task.status = "running"
        task.heartbeat("started")
        self.tasks[topic_id] = task
        self._update_handled_topics([topic_id])
        print(f"[START] topic={topic_id} pid={proc.pid} log={log_path}")
        return True

    # ========== 历史记录 ==========
    def _load_handled_topics(self) -> None:
        self.handled_topics = read_handled_topics(self.config.handled_topics_path)
        if self.handled_topics:
            preview = ", ".join(sorted(self.handled_topics)[:5])
            print(
                f"[INIT] 已加载历史话题 {len(self.handled_topics)} 个 preview={preview}"
            )
        else:
            print("[INIT] 尚无历史处理话题记录")

    def _update_handled_topics(self, new_topics: List[str]) -> None:
        if not new_topics:
            return
        self.handled_topics.update(new_topics)
        write_handled_topics(self.config.handled_topics_path, self.handled_topics)

    # ========== 命令处理 ==========
    def enqueue_command(self, command: str) -> None:
        self.command_queue.put(command)

    def _process_commands(self) -> None:
        while True:
            try:
                cmd = self.command_queue.get_nowait()
            except queue.Empty:
                break
            print(f"[CMD] processing: {cmd}")
            self._handle_command(cmd.strip())

    def _handle_command(self, cmd: str) -> None:
        if not cmd:
            print("[CMD] 忽略空命令（可能未正确捕获输入或输入仅为空白）")
            return
        if cmd in {"quit", "exit"}:
            print("[CHOICE] exit requested")
            self.stop_event.set()
            return
        if cmd == "list":
            self._print_status()
            return
        if cmd.startswith("stop "):
            _, topic_id = cmd.split(" ", 1)
            self._stop_topic(topic_id.strip())
            return
        if cmd == "refresh":
            self._refresh_topics()
            return
        print(f"[WARN] 未识别命令: {cmd}")

    def _print_status(self) -> None:
        if not self.tasks:
            print("[RUN] 当前无运行中的话题")
            return
        running_tasks = self._ordered_running_tasks()
        if not running_tasks:
            print("[RUN] 当前无运行中的话题")
            return

        for idx, task in enumerate(running_tasks, 1):
            self._print_single_task(task, idx)

    def _print_single_task(self, task: TopicTask, index: Optional[int] = None) -> None:
        hb = task.last_heartbeat
        hb_text = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(hb)) if hb else "-"
        pid_text = str(task.process.pid) if task.process else "-"
        log_name = task.log_path.name if task.log_path else "-"
        log_hint = (task.log_excerpt.splitlines() or ["-"])[-1].strip()

        prefix = f"[RUN {index}]" if index is not None else "[RUN]"
        print(
            f"{prefix} topic={task.topic_id} status={task.status} "
            f"start={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(task.start_time))} "
            f"pid={pid_text} hb={hb_text} notes={len(task.notes)} "
            f"log={log_name} last_line={log_hint or '-'}"
        )

    def _ordered_running_tasks(self) -> List[TopicTask]:
        return sorted(
            [task for task in self.tasks.values() if task.is_running()],
            key=lambda t: (t.start_time, t.topic_id),
        )

    def _stop_topic(self, topic_or_index: str) -> None:
        topic_id = self._resolve_topic_identifier(topic_or_index)
        if not topic_id:
            return
        task = self.tasks.get(topic_id)
        if not task:
            print(f"[WARN] topic {topic_id} 不在运行列表中")
            return
        task.no_restart = True
        task.end_reason = "stopped by user"
        # 标记为已处理，避免后续 refresh 把同一话题再次入队
        if topic_id not in self.handled_topics:
            self.handled_topics.add(topic_id)
            write_handled_topics(self.config.handled_topics_path, self.handled_topics)
        if topic_id in self.pending_topics:
            try:
                self.pending_topics.remove(topic_id)
            except ValueError:
                pass
        self._terminate_task(task, reason="stopped by user")
        self._purge_inactive_tasks()
        print(f"[CHOICE] stop topic={topic_id}")

    def _resolve_topic_identifier(self, text: str) -> Optional[str]:
        text = text.strip()
        if not text:
            print("[WARN] stop 命令缺少参数")
            return None
        if text.isdigit():
            index = int(text)
            running_tasks = self._ordered_running_tasks()
            if 1 <= index <= len(running_tasks):
                return running_tasks[index - 1].topic_id
            print(
                f"[WARN] 无效的序号 {index}，当前运行中的任务数为 {len(running_tasks)}"
            )
            return None
        return text

    def _terminate_task(self, task: TopicTask, reason: str) -> None:
        proc = task.process
        if proc and proc.poll() is None:
            try:
                proc.terminate()
            except Exception as exc:  # pragma: no cover - 终止异常
                print(f"[WARN] 无法终止 topic {task.topic_id}: {exc}")
            try:
                proc.wait(timeout=self.config.process_graceful_timeout_sec)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=1.0)
                except Exception as exc:  # pragma: no cover - kill 失败
                    print(f"[WARN] 无法强杀 topic {task.topic_id}: {exc}")
        if task.status not in {"error", "ended"}:
            task.status = "stopped"
        task.heartbeat(reason)

    def _purge_inactive_tasks(self) -> None:
        """移除已停止/结束且不再需要展示的任务。"""

        removable: List[str] = []
        for topic_id, task in list(self.tasks.items()):
            if task.is_running():
                continue
            if task.status in {"stopped", "ended", "exited", "error"} or task.no_restart:
                removable.append(topic_id)

        if not removable:
            return

        for topic_id in removable:
            self.tasks.pop(topic_id, None)
            if topic_id in self.pending_topics:
                try:
                    self.pending_topics.remove(topic_id)
                except ValueError:
                    pass

    def _refresh_topics(self) -> None:
        try:
            self.latest_topics = run_filter_once(
                self.filter_config,
                self.config.filter_output_path,
                timeout_sec=self.config.filter_timeout_sec,
                max_retries=self.config.filter_max_retries,
                retry_delay_sec=self.config.filter_retry_delay_sec,
            )
            self.topic_details = {
                _topic_id_from_entry(item): item
                for item in self.latest_topics
                if _topic_id_from_entry(item)
            }
            new_topics = compute_new_topics(self.latest_topics, self.handled_topics)
            if new_topics:
                preview = ", ".join(new_topics[:5])
                print(
                    f"[INCR] 新话题 {len(new_topics)} 个，将更新历史记录 preview={preview}"
                )
                for topic_id in new_topics:
                    if topic_id in self.pending_topics:
                        continue
                    if topic_id in self.tasks and self.tasks[topic_id].is_running():
                        continue
                    self.pending_topics.append(topic_id)
            else:
                print("[INCR] 无新增话题")
        except Exception as exc:  # pragma: no cover - 网络/外部依赖
            print(f"[ERROR] 筛选流程失败：{exc}")
            self.latest_topics = []

    def _cleanup_all_tasks(self) -> None:
        for task in list(self.tasks.values()):
            if task.is_running():
                print(f"[CLEAN] 停止 topic={task.topic_id} ...")
                self._terminate_task(task, reason="cleanup")
        # 写回 handled_topics，确保最新状态落盘
        write_handled_topics(self.config.handled_topics_path, self.handled_topics)

    def _maybe_reload_filter_config(self, now: Optional[float] = None) -> None:
        if now is None:
            now = time.time()
        if now < self._next_filter_reload:
            return

        self._next_filter_reload = now + FILTER_CONFIG_RELOAD_INTERVAL_SEC

        try:
            current_mtime = self.config.filter_params_path.stat().st_mtime
        except OSError:
            print(
                f"[WARN] 无法访问筛选配置文件：{self.config.filter_params_path}，保留现有配置。"
            )
            return

        if self._filter_conf_mtime is not None and current_mtime <= self._filter_conf_mtime:
            return

        try:
            filter_conf_raw = _load_json_file(self.config.filter_params_path)
            self.filter_config = FilterConfig.from_dict(filter_conf_raw)
            self._filter_conf_mtime = current_mtime
            print(
                f"[CONFIG] 已重新加载筛选配置（每 {FILTER_CONFIG_RELOAD_INTERVAL_SEC // 60:.0f} 分钟轮询一次）。"
            )
        except Exception as exc:  # pragma: no cover - 文件读取/解析异常
            print(f"[WARN] 重载筛选配置失败，将继续使用旧配置：{exc}")

    def _restore_runtime_status(self) -> None:
        """尝试从上次运行的状态文件恢复待处理队列等信息。"""

        if not self.status_path.exists():
            return
        try:
            payload = _load_json_file(self.status_path)
            handled_topics = payload.get("handled_topics") or []
            pending_topics = payload.get("pending_topics") or []
            tasks_snapshot = payload.get("tasks") or {}
        except Exception as exc:  # pragma: no cover - 容错
            print(f"[WARN] 无法读取运行状态文件，已忽略: {exc}")
            return

        if handled_topics:
            self.handled_topics.update(str(t) for t in handled_topics)

        restored_topics: List[str] = []
        for topic_id in pending_topics:
            topic_id = str(topic_id)
            if topic_id in self.pending_topics or topic_id in self.handled_topics:
                continue
            restored_topics.append(topic_id)
            self.pending_topics.append(topic_id)

        for topic_id, info in tasks_snapshot.items():
            topic_id = str(topic_id)
            if topic_id in self.handled_topics:
                continue
            if topic_id not in self.pending_topics:
                restored_topics.append(topic_id)
                self.pending_topics.append(topic_id)

            task = TopicTask(topic_id=topic_id)
            task.status = "pending"
            task.notes.append("restored from runtime_status")
            config_path = info.get("config_path")
            log_path = info.get("log_path")
            if config_path:
                task.config_path = Path(config_path)
            if log_path:
                task.log_path = Path(log_path)
            self.tasks[topic_id] = task

        if restored_topics:
            preview = ", ".join(restored_topics[:5])
            print(f"[RESTORE] 已从运行状态恢复 {len(restored_topics)} 个话题：{preview}")

    def _dump_runtime_status(self) -> None:
        payload = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "handled_topics_total": len(self.handled_topics),
            "handled_topics": sorted(self.handled_topics),
            "pending_topics": list(self.pending_topics),
            "tasks": {},
        }
        for topic_id, task in self.tasks.items():
            payload["tasks"][topic_id] = {
                "status": task.status,
                "pid": task.process.pid if task.process else None,
                "last_heartbeat": task.last_heartbeat,
                "notes": task.notes,
                "log_path": str(task.log_path) if task.log_path else None,
                "config_path": str(task.config_path) if task.config_path else None,
            }
        _dump_json_file(self.status_path, payload)
        print(f"[STATE] 已写入运行状态到 {self.status_path}")

    # ========== 入口方法 ==========
    def command_loop(self) -> None:
        try:
            prompt_shown = False
            while not self.stop_event.is_set():
                try:
                    if not prompt_shown:
                        # 主动刷新提示符，避免被后台日志刷屏覆盖
                        print("poly> ", end="", flush=True)
                        prompt_shown = True

                    ready, _, _ = select.select(
                        [sys.stdin], [], [], self.config.command_poll_sec
                    )
                    if not ready:
                        continue

                    line = sys.stdin.readline()
                    if line == "":
                        cmd = "exit"
                    else:
                        cmd = line.rstrip("\n")
                    prompt_shown = False
                except EOFError:
                    cmd = "exit"
                except Exception as exc:  # pragma: no cover - 保护交互循环不被意外异常终止
                    print(f"[ERROR] command loop input failed: {exc}")
                    traceback.print_exc()
                    time.sleep(self.config.command_poll_sec)
                    continue
                # 立刻反馈收到的命令，避免在日志刷屏时用户误以为命令未被捕获
                if cmd:
                    print(f"[CMD] received: {cmd}")
                else:
                    # 空行依旧入队，后续会在 _handle_command 里被忽略
                    print("[CMD] received: <empty>")
                self.enqueue_command(cmd)
                # 轻微休眠，防止输入为空或重复换行时产生过多提示刷屏
                time.sleep(self.config.command_poll_sec)
        except KeyboardInterrupt:
            print("\n[WARN] Ctrl+C detected, stopping...")
            self.stop_event.set()
        except Exception as exc:  # pragma: no cover - 防御性保护
            print(f"[ERROR] command loop crashed: {exc}")
            traceback.print_exc()


# =====================
# CLI 入口
# =====================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket maker autorun")
    parser.add_argument(
        "--global-config",
        type=Path,
        default=MAKER_ROOT / "config" / "global_config.json",
        help="全局调度配置 JSON 路径",
    )
    parser.add_argument(
        "--strategy-config",
        type=Path,
        default=MAKER_ROOT / "config" / "strategy_defaults.json",
        help="策略参数模板 JSON 路径",
    )
    parser.add_argument(
        "--filter-config",
        type=Path,
        default=MAKER_ROOT / "config" / "filter_params.json",
        help="筛选参数配置 JSON 路径",
    )
    parser.add_argument(
        "--run-config-template",
        type=Path,
        default=MAKER_ROOT / "config" / "run_params.json",
        help="运行参数模板 JSON 路径（传递给 Volatility_arbitrage_run.py）",
    )
    parser.add_argument(
        "--no-repl",
        action="store_true",
        help="禁用交互式命令循环，仅按配置运行",
    )
    parser.add_argument(
        "--command",
        action="append",
        help="启动后自动执行的命令（可多次提供），例如 list 或 stop <topic_id>",
    )
    return parser.parse_args(argv)


def load_configs(
    args: argparse.Namespace,
) -> tuple[GlobalConfig, Dict[str, Any], FilterConfig, Dict[str, Any]]:
    global_conf_raw = _load_json_file(args.global_config)
    strategy_conf_raw = _load_json_file(args.strategy_config)
    filter_conf_raw = _load_json_file(args.filter_config)
    run_params_template = _load_json_file(args.run_config_template)
    return (
        GlobalConfig.from_dict(global_conf_raw),
        strategy_conf_raw,
        FilterConfig.from_dict(filter_conf_raw),
        run_params_template,
    )


def run_filter_once(
    filter_conf: FilterConfig,
    output_path: Path,
    *,
    timeout_sec: Optional[float] = None,
    max_retries: int = 0,
    retry_delay_sec: float = 3.0,
) -> List[Dict[str, Any]]:
    """调用筛选脚本，落盘 JSON，并返回话题列表，带超时与可选重试。"""

    filter_conf.apply_blacklist()
    filter_conf.apply_highlight()
    filter_conf.apply_reversal()

    attempts = max(1, int(max_retries) + 1)
    for attempt in range(1, attempts + 1):
        timeout_label = f"{timeout_sec}s" if timeout_sec is not None else "no-timeout"
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    filter_script.collect_filter_results, **filter_conf.to_filter_kwargs()
                )
                result = (
                    future.result(timeout=timeout_sec)
                    if timeout_sec is not None
                    else future.result()
                )
            break
        except concurrent.futures.TimeoutError as exc:  # pragma: no cover - 线程超时
            print(
                "[WARN] 筛选调用超时（{} 内未返回，通常为 Gamma/clob 接口无响应或窗口过大）".format(
                    timeout_label
                )
            )
            if attempt >= attempts:
                raise
            time.sleep(retry_delay_sec)
        except Exception as exc:  # pragma: no cover - 网络/线程异常
            print(
                f"[WARN] 筛选调用失败（尝试 {attempt}/{attempts}，timeout={timeout_label}）: {exc}"
            )
            if attempt >= attempts:
                raise
            time.sleep(retry_delay_sec)

    highlight_map: Dict[str, List[str]] = {}
    for ho in result.highlights:
        slug = ho.market.slug
        side = (ho.outcome.name or "").upper()
        if not side:
            continue
        highlight_map.setdefault(slug, []).append(side)

    topics: List[Dict[str, Any]] = []
    for ms in result.chosen:
        highlight_sides = highlight_map.get(ms.slug, [])
        if not highlight_sides:
            # 仅保留命中高亮条件的市场，避免不满足高亮口径的条目进入 topics 列表
            continue
        preferred_side = highlight_sides[0]
        topics.append(
            {
                "slug": ms.slug,
                "title": ms.title,
                "yes_token": ms.yes.token_id,
                "no_token": ms.no.token_id,
                "end_time": ms.end_time.isoformat() if ms.end_time else None,
                "liquidity": ms.liquidity,
                "total_volume": ms.totalVolume,
                "preferred_side": preferred_side,
                "highlight_sides": highlight_sides,
            }
        )

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": filter_conf.to_dict(),
        "total_markets": result.total_markets,
        "candidates": len(result.candidates),
        "chosen": len(result.chosen),
        "rejected": len(result.rejected),
        "highlights": len(result.highlights),
        "topics": topics,
    }
    _dump_json_file(output_path, payload)
    print(f"[FILTER] 已写入筛选结果到 {output_path}，共 {len(topics)} 个话题")
    return topics


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    global_conf, strategy_conf, filter_conf, run_params_template = load_configs(args)

    manager = AutoRunManager(global_conf, strategy_conf, filter_conf, run_params_template)

    def _handle_sigterm(signum: int, frame: Any) -> None:  # pragma: no cover - 信号处理不可测
        print(f"\n[WARN] signal {signum} received, exiting...")
        manager.stop_event.set()

    signal.signal(signal.SIGTERM, _handle_sigterm)

    worker = threading.Thread(target=manager.run_loop, daemon=True)
    worker.start()

    if args.command:
        for cmd in args.command:
            manager.enqueue_command(cmd)

    if args.no_repl or args.command:
        try:
            while worker.is_alive():
                time.sleep(global_conf.command_poll_sec)
        except KeyboardInterrupt:
            print("\n[WARN] Ctrl+C detected, stopping...")
            manager.stop_event.set()
    else:
        manager.command_loop()

    worker.join()


if __name__ == "__main__":
    main()
