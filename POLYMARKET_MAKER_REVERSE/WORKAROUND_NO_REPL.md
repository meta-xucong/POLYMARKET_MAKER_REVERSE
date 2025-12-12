# 无 REPL 后台运行 + 独立终端批量指令操作指南

当后台挂单任务与 REPL 共用同一终端时，子进程会占用标准输入，导致在 REPL 输入指令后阻塞、无法执行。以下步骤利用 `--no-repl` 关闭交互 REPL，并在另一终端通过 `--command` 参数批量发送指令，避免输入争用。

## 前置要求
- Python 3.10+ 环境。
- 确认已在仓库根目录（包含 `poly_maker_autorun.py`）。
- 按需准备配置文件（如 `POLYMARKET_MAKER/config/global_config.json`、`POLYMARKET_MAKER/config/strategy_defaults.json`、`POLYMARKET_MAKER/config/filter_params.json`）。

## 步骤概览
1. **终端 A：以非交互模式启动主控脚本**（不占用 REPL）。
2. **终端 B：按需发送批量命令**（列出任务、刷新、停止等）。

## 详细步骤
### 1. 终端 A：后台启动（无 REPL）
在第一个终端运行下列命令，将主控脚本以无 REPL 模式启动，并可使用 `nohup` 让其在关闭终端后继续运行：
```bash
cd /workspace/POLYMARKET_MAKER_AUTO
nohup python poly_maker_autorun.py \
  --no-repl \
  --global-config POLYMARKET_MAKER/config/global_config.json \
  --strategy-config POLYMARKET_MAKER/config/strategy_defaults.json \
  --filter-config POLYMARKET_MAKER/config/filter_params.json \
  > logs_autorun_no_repl.out 2>&1 &
```
要点：
- `--no-repl` 关闭交互循环，防止子进程抢占 stdin。
- `nohup ... &` 可选，用于保持后台运行；输出重定向到 `logs_autorun_no_repl.out` 便于查看。

### 2. 终端 B：批量发送命令
在另一个终端中，每次需要查看或控制任务时，使用 `--command` 参数调用同一脚本即可。示例：
- 查看当前运行的任务：
  ```bash
  cd /workspace/POLYMARKET_MAKER_AUTO
  python poly_maker_autorun.py --command "list"
  ```
- 立即刷新筛选并查看新增话题：
  ```bash
  python poly_maker_autorun.py --command "refresh" --command "list"
  ```
- 停止指定话题（用实际话题 ID 替换 `<topic_id>`）：
  ```bash
  python poly_maker_autorun.py --command "stop <topic_id>" --command "list"
  ```
说明：
- `--command` 可多次提供，将按顺序执行。执行完毕即退出，不会进入 REPL。
- 这些命令与后台主控脚本共享同一运行状态文件，因此能即时生效。

### 3. 查看后台输出与状态
- 主控脚本运行日志：查看 `logs_autorun_no_repl.out`（或自定义路径）。
- 子任务日志：参见 `POLYMARKET_MAKER/logs/` 下的各话题日志（文件名通常以 `autorun_<topic>.log` 命名）。
- 运行状态快照：`POLYMARKET_MAKER/logs/run_state.json`（或配置中指定的路径）。

## 取消后台进程
如需停止后台主控脚本，可在终端 B 执行：
```bash
python poly_maker_autorun.py --command "exit"
```
若使用了 `nohup ... &`，也可通过 `ps -ef | grep poly_maker_autorun.py` 查 PID 后手动结束进程。

## 常见问题
- **命令无响应或提示空命令**：确认主控脚本确实以 `--no-repl` 运行，并在发送指令时使用了新的终端会话。
- **日志未更新**：检查 `nohup` 输出文件路径是否正确，或查看 `POLYMARKET_MAKER/logs/` 目录下的各话题日志。

通过上述分离式操作，REPL 不再与子进程争抢输入，确保指令可以即时执行。
