# JSON 配置字段说明

本文件汇总 `POLYMARKET_MAKER/config/` 目录下各个 JSON 配置的字段含义、推荐取值与格式要求，方便在自动化脚本或单市场策略运行前快速校验参数。

## run_params.json —— 单市场运行参数
用于 `Volatility_arbitrage_run.py` 等单市场脚本。所有百分比字段均支持写成 0~1 之间的小数；若填写大于 1 的数值会被视作百分比并自动除以 100（如填写 5 代表 5%）。【F:POLYMARKET_MAKER/Volatility_arbitrage_run.py†L138-L147】

| 字段 | 作用 | 类型/格式 | 推荐设置 |
| --- | --- | --- | --- |
| `market_url` | 目标市场或子问题的 URL/slug；为空则无法启动。 | 字符串，支持完整链接或 slug | 必填，保持无空格。 |
| `timezone` | 人工指定市场时区，缺失时默认 `America/New_York`。 | IANA 时区名称 | 建议与市场实际时区一致。 |
| `deadline_override_ts` | 人工指定的截止时间戳；设置后覆盖自动识别的截止时间。 | UNIX 秒级或毫秒级时间戳 | 仅当自动解析不可靠时填写。 |
| `disable_deadline_checks` | 是否完全跳过截止时间校验/提示。 | 布尔 | 仅在明确无需截止时间时启用。 |
| `deadline_policy.override_choice` | 选择常用截止时间模板（1=12:00 PM ET，2=23:59 ET，3=00:00 UTC，4=不设定）。 | 整数 1~4 或 `null` | 不填则沿用自动解析；需要人工覆盖时填写编号。 |
| `deadline_policy.disable_deadline` | 是否强制清空截止时间。 | 布尔 | 特殊诊断场景下使用。 |
| `deadline_policy.timezone` | 应用默认截止时间时的时区。 | IANA 时区名称 | 与 `default_deadline` 保持一致。 |
| `deadline_policy.default_deadline.time` | 当需要回退到默认截止时间时使用的“时:分”字符串，可写 `HH:MM` 或 `HH.MM`。 | 字符串（24 小时制） | 例如 `12:59`。 |
| `deadline_policy.default_deadline.timezone` | 默认截止时间对应的时区。 | IANA 时区名称 | 例如 `America/New_York`。 |
| `side` | 下单方向，`YES` 或 `NO`；缺失时会尝试使用 `preferred_side` 或 `highlight_sides[0]`，仍无法确定或值非法时直接报错退出。【F:POLYMARKET_MAKER/Volatility_arbitrage_run.py†L153-L172】【F:POLYMARKET_MAKER/Volatility_arbitrage_run.py†L1941-L1947】 | 字符串（不区分大小写） | 必填；建议与目标 token 一致。 |
| `order_size` | 手动指定份额，配合 `order_size_is_target` 判断含义。 | 正数 | 留空则按 $1 等额推算。 |
| `order_size_is_target` | 为真时将 `order_size` 视为目标总持仓，否则视为单笔下单量。【F:POLYMARKET_MAKER/Volatility_arbitrage_run.py†L1940-L1950】 | 布尔 | 需要限制总敞口时设为 `true`。 |
| `sell_mode` | 卖出挂单策略：`aggressive` 更靠近盘口，`conservative` 稍远。 | 枚举字符串 | `aggressive`/`conservative`。 |
| `buy_price_threshold` | 仅当买入价格低于该阈值（0~1，小数形式）时才下单。 | 浮点数 | 常用 0.1~0.9；留空则按策略默认。 |
| `drop_window_minutes` | 计算下跌幅度的滑动窗口长度（分钟）。 | 浮点数 | 推荐 10~120 之间。 |
| `drop_pct` | 触发买入的相对高点跌幅阈值；>1 会按百分比自动换算。【F:POLYMARKET_MAKER/Volatility_arbitrage_run.py†L1960-L1970】 | 浮点比例 | 例如 `0.05`（5%）。 |
| `profit_pct` | 止盈阈值（收益率）。>1 同样视为百分比输入。【F:POLYMARKET_MAKER/Volatility_arbitrage_run.py†L1960-L1970】 | 浮点比例 | 例如 `0.01`~`0.1`。 |
| `enable_incremental_drop_pct` | 是否在卖出后逐步提高下一次买入的跌幅阈值。 | 布尔 | 与 `incremental_drop_pct_step` 搭配使用。 |
| `incremental_drop_pct_step` | 每次卖出后提升的跌幅阈值步长；0~1 或百分比形式。【F:POLYMARKET_MAKER/Volatility_arbitrage_run.py†L1965-L1979】 | 浮点比例 | 例如 `0.002`（0.2%）。 |
| `countdown.minutes_before_end` | 距离市场结束 N 分钟时切换为“仅卖出”模式。 | 浮点数 | 常用 60~360；为空则尝试用绝对时间。 |
| `countdown.absolute_time` | 直接指定倒计时开始的绝对时间，支持时间戳、ISO 字符串（`YYYY-MM-DDTHH:MM:SSZ`）或简单日期/日期时间文本。【F:POLYMARKET_MAKER/Volatility_arbitrage_run.py†L436-L476】【F:POLYMARKET_MAKER/Volatility_arbitrage_run.py†L1981-L2019】 | 数字或字符串 | 优先写 UTC/带时区的 ISO 格式。 |
| `countdown.timezone` | 当 `absolute_time` 只写日期/无时区时，用于推断时区。 | IANA 时区名称 | 与市场时区一致即可。 |

## global_config.json —— 自动化调度与路径
供 `poly_maker_autorun.py` 使用，控制筛选/调度循环、文件路径与重试策略。【F:poly_maker_autorun.py†L32-L48】【F:poly_maker_autorun.py†L117-L219】

| 字段 | 作用 | 类型/格式 | 推荐设置 |
| --- | --- | --- | --- |
| `scheduler.max_concurrent_jobs` | 同时运行的子进程/任务上限。 | 整数 | 根据机器核数调整，1~4 为宜。 |
| `scheduler.poll_interval_seconds` | 轮询筛选结果的间隔秒数。 | 浮点 | 1~10 秒。 |
| `scheduler.task_timeout_seconds` | 单个任务的超时时长。 | 整数（秒） | 180 为默认，可按策略时长调整。 |
| `paths.log_directory` | 日志目录（可相对或绝对路径）。 | 字符串 | 统一写入 `POLYMARKET_MAKER/logs/`。 |
| `paths.data_directory` | 数据目录（去重状态、筛选结果等）。 | 字符串 | 默认与日志共用 `POLYMARKET_MAKER/logs/`，便于一处收集。 |
| `paths.order_history_file` | 历史订单记录文件路径。 | 字符串 | 默认写入 `POLYMARKET_MAKER/logs/order_history.jsonl`。 |
| `paths.run_state_file` | 运行状态快照文件。 | 字符串 | 默认写入 `POLYMARKET_MAKER/logs/run_state.json`。 |
| `retry_strategy.max_attempts` | 筛选或子任务的最大重试次数。 | 整数 | 3~5。 |
| `retry_strategy.initial_backoff_seconds` | 首次重试等待秒数。 | 浮点 | 1.0 起步。 |
| `retry_strategy.backoff_multiplier` | 指数退避倍率。 | 浮点 | 2.0 表示每次翻倍。 |
| `retry_strategy.max_backoff_seconds` | 退避的最大等待上限。 | 浮点 | 30.0 或更高。 |
| `retry_strategy.jitter_fraction` | 退避抖动比例，避免雪崩。 | 0~1 小数 | 0.1~0.3。 |
| `monitoring.metrics_flush_interval_seconds` | 指标刷写周期。 | 整数/浮点（秒） | 15 秒默认。 |
| `monitoring.healthcheck_interval_seconds` | 健康检查间隔。 | 整数/浮点（秒） | 30 秒默认。 |

## strategy_defaults.json —— 策略模板
为不同话题/主题提供默认下单参数与覆盖示例。【F:POLYMARKET_MAKER/config/strategy_defaults.json†L1-L24】

| 字段 | 作用 | 类型/格式 | 推荐设置 |
| --- | --- | --- | --- |
| `default.min_edge` | 最小优势阈值（策略内部估值与盘口差异）。 | 0~1 小数 | 0.01~0.05。 |
| `default.max_position_per_market` | 单市场最大持仓（份额或名义金额）。 | 浮点 | 按风险偏好调整。 |
| `default.order_size` | 默认下单量。 | 浮点 | 20~100 之间常用。 |
| `default.spread_target` | 期望挂单点差目标。 | 0~1 小数 | 0.005~0.02。 |
| `default.refresh_interval_seconds` | 策略刷新/再报价周期。 | 整数（秒） | 5~15。 |
| `default.max_open_orders` | 同时挂单数量上限。 | 整数 | 10~30。 |
| `topics.*` | 针对特定话题 ID/slug 的覆盖：可单独调整 `topic_name`、`min_edge`、`max_position_per_market`、`order_size`、`spread_target`、`refresh_interval_seconds`、`max_open_orders`。 | 按字段类型填写 | 仅覆盖需要调整的字段，其余沿用 `default`。 |

## filter_params.json —— 市场筛选参数
驱动 `Customize_fliter_reverse.py` 的 REST 筛选器，字段与命令行参数一致，可配置高亮与价格反转检测等阈值。【F:POLYMARKET_MAKER/config/filter_params.json†L1-L71】【F:Customize_fliter_reverse.py†L1214-L1280】

| 字段 | 作用 | 类型/格式 | 推荐设置 |
| --- | --- | --- | --- |
| `min_end_hours` | 抓取结束时间至少距离当前多少小时的市场。 | 浮点（小时） | ≥1。 |
| `max_end_days` | 抓取未来 N 天内结束的市场。 | 整数（天） | 1~7。 |
| `gamma_window_days` | Gamma 时间切片窗口长度；命中 500 条时会递归切分。 | 整数（天） | 1~3。 |
| `gamma_min_window_hours` | 递归切分的最小窗口大小。 | 整数（小时） | 1~6。 |
| `legacy_end_days` | 结束早于该天数的市场视为旧格式/归档。 | 整数（天） | 默认 730。 |
| `allow_illiquid` | 是否允许无报价市场通过。 | 布尔 | 诊断用，生产环境建议 `false`。 |
| `skip_orderbook` | 是否跳过订单簿/价格回补。 | 布尔 | 仅诊断或节省请求时使用。 |
| `no_rest_backfill` | 关闭 REST 回补（默认开启）。 | 布尔 | 一般保持 `false`。 |
| `books_batch_size` | `/books` 回补的 token_id 批量大小。 | 整数 | 100~500。 |
| `books_timeout_sec` | `/books` 回补单次请求超时。 | 浮点（秒） | 3~10。 |
| `only` | 仅处理包含该子串的 slug/title（大小写不敏感）。 | 字符串 | 留空表示不过滤。 |
| `blacklist_terms` | 标题/slug 命中任意黑名单词条即跳过。 | 字符串数组 | 可按需增删。 |
| `highlight.max_hours` | 高亮条件：距离结束 ≤ 该小时数。 | 浮点（小时） | 24~72。 |
| `highlight.min_total_volume` | 高亮条件：总成交量下限（USDC）。 | 浮点 | 1_000~50_000。 |
| `highlight.max_ask_diff` | 高亮条件：单边点差上限（|ask-bid|），YES/NO 任一侧满足即可。 | 0~1 小数 | 0.05~0.2。 |
| `reversal.enabled` | 是否开启价格反转检测，高亮时会优先筛选命中反转的市场。 | 布尔 | 建议保持 `true`。 |
| `reversal.p1` | 反转判定：旧段最高价需低于该值。 | 0~1 小数 | 0.3~0.4。 |
| `reversal.p2` | 反转判定：近段最高价需高于该值。 | 0~1 小数 | 0.75~0.85。 |
| `reversal.window_hours` | 近段回溯窗口大小。 | 浮点（小时） | 1~6。 |
| `reversal.lookback_days` | 旧段回溯天数。 | 浮点（天） | 3~7。 |
| `reversal.short_interval` | 预筛时短窗口的 interval 表达式。 | 字符串（如 `6h`） | 与策略保持一致即可。 |
| `reversal.short_fidelity` | 短窗口 fidelity（分钟级采样）。 | 整数 | 10~30。 |
| `reversal.long_fidelity` | 长窗口 fidelity（分钟级采样）。 | 整数 | 30~90。 |

> 提示：`filter_params.json` 直接对应命令行参数，修改后无需转换即可被 `Customize_fliter_reverse.py` 读取并复用，且高亮/反转参数会覆盖脚本顶部的默认值。【F:Customize_fliter_reverse.py†L1214-L1280】【F:Customize_fliter_reverse.py†L1249-L1280】
