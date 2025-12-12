# POLYMARKET_MAKER_REVERSE 使用说明

## 项目概览
本仓库提供多话题挂单策略的自动化运行流程：`poly_maker_autorun.py` 会调用 `Customize_fliter_blacklist.py` 进行市场筛选，再按预设策略启动 `POLYMARKET_MAKER/Volatility_arbitrage_run.py` 等交易脚本，实现自动去重、并发控制与运行状态监控。

## 目录与核心文件
- `poly_maker_autorun.py`：自动化主控脚本，负责筛选、增量识别、任务调度与命令控制。
- `Customize_fliter_blacklist.py`：REST-only 市场筛选脚本，支持高亮参数与流式输出。
- `POLYMARKET_MAKER/config/`：调度、筛选与策略配置示例（`global_config.json`、`filter_params.json`、`strategy_defaults.json`、`run_params.json`、`trading.yaml`）。
- `POLYMARKET_MAKER/logs/`：运行时数据与日志目录（筛选结果、去重状态、运行快照、子任务日志等，示例配置均指向此处）。

## 环境准备
1. 安装 Python 3.10+。
2. 建议在虚拟环境中安装依赖；筛选脚本至少需要 `requests`，交易脚本依赖 Polymarket 相关库。根据实际部署补充安装命令，例如：
   ```bash
   pip install requests
   ```

## 快速运行：poly_maker_autorun
1. 准备配置文件（可直接使用默认路径）：
   - 全局调度：`POLYMARKET_MAKER/config/global_config.json`
   - 策略模板：`POLYMARKET_MAKER/config/strategy_defaults.json`
   - 筛选参数：`POLYMARKET_MAKER/config/filter_params.json`
2. 启动自动化流程（默认进入交互命令行）：
   ```bash
   python poly_maker_autorun.py \
     --global-config POLYMARKET_MAKER/config/global_config.json \
     --strategy-config POLYMARKET_MAKER/config/strategy_defaults.json \
     --filter-config POLYMARKET_MAKER/config/filter_params.json
   ```
3. 非交互模式/自动执行命令：
   ```bash
   python poly_maker_autorun.py --no-repl --command "list"
   ```
   `--command` 可重复提供（如 `--command "list" --command "stop <topic_id>"`）。

### 运行时命令
- `list`：打印当前运行任务、进程号、最近心跳与日志摘要。
- `stop <topic_id>`：终止指定话题的子进程。
- `refresh`：立即重新拉取筛选结果并刷新待启动的话题。
- `exit` / `quit`：停止所有任务并退出。

## 配置与参数示例
- **全局调度（global_config.json）**：
  - 调度与超时：`scheduler.max_concurrent_jobs`、`scheduler.poll_interval_seconds`、`scheduler.task_timeout_seconds`。
  - 路径：`paths.log_directory`、`paths.data_directory`、`paths.order_history_file`、`paths.run_state_file`。
  - 重试与监控：`retry_strategy.*` 与 `monitoring.*` 字段提供指数退避与健康检查周期的示例。
- **筛选参数（filter_params.json）**：结束时间窗口、是否跳过订单簿、允许非流动性市场、可配置黑名单词条，以及高亮条件（`highlight.max_hours`、`highlight.min_total_volume`、`highlight.max_ask_diff` 等）。
- **价格反转检测（reversal）**：替代原先的价格区间高亮，默认启用“两段式”检测：短窗口（如 `6h` interval）触发冲高阈值 `p2=0.8`，命中后再用 5 天长窗口确认旧段最高价低于 `p1=0.35` 且近段最高价继续高于 `p2`。相关参数位于 `filter_params.json` 的 `reversal` 区段，可通过 CLI/配置开关调整。
- **策略模板（strategy_defaults.json）**：`default` 段定义最小优势、下单量、点差目标、刷新周期等，`topics` 段可按话题 ID/名称覆盖。
- **单市场运行参数（run_params.json）**：`market_url`、`side`、下单大小策略、跌幅/盈利阈值、倒计时配置等。
- **交易执行参数（trading.yaml）**：下单切片区间、重试次数、价格让步步长、订单轮询频率与最小报价金额。

## 日志与数据产物
`poly_maker_autorun.py` 会按照全局配置落盘（示例配置将所有文件集中在 `POLYMARKET_MAKER/logs/` 下）：
- 筛选结果：`POLYMARKET_MAKER/logs/topics_filtered.json`（最新筛选详情与话题列表）。
- 历史去重：`POLYMARKET_MAKER/logs/handled_topics.json`（已处理话题 ID 集合）。
- 运行快照：`POLYMARKET_MAKER/logs/run_state.json`（任务状态、PID、日志路径）。
- 子任务日志：`POLYMARKET_MAKER/logs/autorun/<topic_id>.log`（每个话题独立日志，文件名中的斜杠会被替换为下划线）。

## 独立运行市场筛选脚本
若仅需筛选，可直接运行：
```bash
python Customize_fliter_blacklist.py --help
```
常用示例：
```bash
python Customize_fliter_blacklist.py \
  --min-end-hours 1 --max-end-days 5 \
  --hl-max-hours 48 --hl-min-total-volume 20000 --hl-max-ask-diff 0.2 \
  --rev-p1 0.35 --rev-p2 0.8 --rev-window-hours 2 --rev-short-interval 6h
```
可追加 `--stream` 查看分片流式输出或调整高亮阈值以适配不同市场环境。
