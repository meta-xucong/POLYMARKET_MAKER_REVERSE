import poly_maker_reverse as module


def _build_manager(tmp_path):
    base = {
        "log_dir": tmp_path / "logs",
        "data_dir": tmp_path / "data",
        "handled_topics_path": tmp_path / "handled.json",
        "filter_output_path": tmp_path / "topics.json",
        "filter_params_path": tmp_path / "filter_params_reverse.json",
        "runtime_status_path": tmp_path / "status.json",
    }
    global_conf = module.GlobalConfig(
        **base,
        topics_poll_sec=1.0,
        command_poll_sec=0.1,
    )
    return module.AutoRunManager(global_conf, {}, module.FilterConfig(), {})


def test_log_indicates_missing_side_detection(tmp_path):
    mgr = _build_manager(tmp_path)
    task = module.TopicTask(topic_id="t1")
    task.log_excerpt = "[ERR] 未提供下单方向 side，且未能从 preferred_side/highlight_sides 推断。"

    assert mgr._log_indicates_missing_side(task) is True


def test_missing_side_log_stops_restart(tmp_path):
    mgr = _build_manager(tmp_path)
    log_path = tmp_path / "t1.log"
    log_path.write_text(
        "[ERR] 未提供下单方向 side，且未能从 preferred_side/highlight_sides 推断。\n",
        encoding="utf-8",
    )

    task = module.TopicTask(topic_id="t1", log_path=log_path)
    mgr._handle_process_exit(task, rc=1)

    assert task.no_restart is True
    assert task.status == "ended"
    assert task.end_reason == "missing side"
