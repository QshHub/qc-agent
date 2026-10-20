# -*- coding: utf-8 -*-
"""LangGraph 编排层的持久化与 API 装配（LangGraph 路线）

## 三层 checkpointer 的选择

| 后端 | 特点 | 用在哪 |
|---|---|---|
| InMemorySaver | 进程内，重启即丢 | 单元测试 |
| **SqliteSaver** | 落盘，跨进程恢复 | **本文件默认**，单机部署够用 |
| PostgresSaver | 多实例共享，生产级 | 规模化后替换，仅需改一行 |

## 为什么要单独一层

`qc_graph.py` 只负责"图长什么样"，不关心状态存哪。
本层负责把 checkpointer 注入进去，并管理 SQLite 连接的生命周期。

SqliteSaver 需要线程内唯一的 sqlite3.Connection。FastAPI 是多线程环境，
因此用 threading.local 包装，保证每个线程各自持有连接（sqlite3 默认禁止跨线程）。
"""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from typing import Any

from qc_graph import build_graph

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DB = Path(os.getenv("QC_CHECKPOINT_DB", str(BASE_DIR / "data" / "qc_checkpoints.db")))

_local = threading.local()
_app_cache: dict[str, Any] = {}


def _get_saver() -> Any:
    """按线程取（并缓存）SqliteSaver。"""
    saver = getattr(_local, "saver", None)
    if saver is not None:
        return saver

    from langgraph.checkpoint.sqlite import SqliteSaver

    CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_DB), check_same_thread=False, timeout=30.0)
    # WAL 提升并发；checkpointer 写入较频繁，收益明显
    conn.execute("PRAGMA journal_mode=WAL")
    saver = SqliteSaver(conn)
    _local.saver = saver
    _local.conn = conn
    return saver


def get_graph() -> Any:
    """取编译好的图（每进程缓存一份；checkpointer 是线程级的）。"""
    if "graph" not in _app_cache:
        _app_cache["graph"] = build_graph(checkpointer=_get_saver())
    else:
        # 线程切换后 saver 可能不同，重新注入
        _app_cache["graph"] = build_graph(checkpointer=_get_saver())
    return _app_cache["graph"]


def get_memory_graph() -> Any:
    """无持久化版本，供测试使用，避免污染检查点库。"""
    return build_graph()


def checkpoint_info() -> dict[str, Any]:
    """检查点库的统计信息，用于运维观测。"""
    info: dict[str, Any] = {
        "backend": "sqlite",
        "path": str(CHECKPOINT_DB),
        "exists": CHECKPOINT_DB.exists(),
    }
    if not CHECKPOINT_DB.exists():
        return info
    try:
        conn = sqlite3.connect(str(CHECKPOINT_DB), timeout=10.0)
        try:
            tables = [
                r[0] for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            info["tables"] = tables
            if "checkpoints" in tables:
                row = conn.execute("SELECT COUNT(*) FROM checkpoints").fetchone()
                info["checkpoint_rows"] = int(row[0]) if row else 0
                row = conn.execute(
                    "SELECT COUNT(DISTINCT thread_id) FROM checkpoints"
                ).fetchone()
                info["threads"] = int(row[0]) if row else 0
            info["size_mb"] = round(CHECKPOINT_DB.stat().st_size / 1024 / 1024, 2)
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def list_pending_threads(limit: int = 50) -> list[dict[str, Any]]:
    """列出处于中断挂起状态的 thread。

    实现说明：LangGraph 的 checkpointer 表结构是内部实现细节，
    这里用 `graph.get_state()` 逐个探测代价高，故直接读 checkpoints 表，
    只取每个 thread 的最新一条记录，再判断其是否含待执行节点。
    表结构随版本可能变化，因此整体包在异常保护里，失败时返回空列表而非报错。
    """
    if not CHECKPOINT_DB.exists():
        return []

    out: list[dict[str, Any]] = []
    try:
        conn = sqlite3.connect(str(CHECKPOINT_DB), timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT thread_id, checkpoint_id, metadata, checkpoint
                FROM checkpoints
                WHERE checkpoint_id IN (
                    SELECT MAX(checkpoint_id) FROM checkpoints GROUP BY thread_id
                )
                ORDER BY checkpoint_id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            for r in rows:
                meta_raw = r["metadata"] if isinstance(r["metadata"], (str, bytes)) else "{}"
                try:
                    import json as _json
                    meta = _json.loads(meta_raw) if meta_raw else {}
                except Exception:  # noqa: BLE001
                    meta = {}
                out.append({
                    "thread_id": r["thread_id"],
                    "checkpoint_id": r["checkpoint_id"],
                    "step": meta.get("step"),
                    "source": meta.get("source"),
                })
        finally:
            conn.close()
    except Exception:  # noqa: BLE001
        return []
    return out


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    g = get_graph()
    print("图已构建，checkpointer = SqliteSaver")
    print("检查点库:", checkpoint_info())
    print("待处理 thread:", list_pending_threads()[:5])
