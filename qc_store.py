# -*- coding: utf-8 -*-
"""持久化层：质检单据与人工复核（第五阶段）

为什么用 SQLite 而不是 ORM（面试可讲）：
  * 质检单据是**强追溯要求**的数据，SQL 直白可审计，出问题能手写 SQL 查；
  * 单机部署场景下 SQLite 的并发能力足够，且**零外部依赖**（标准库自带），
    不会因为多一个数据库服务而增加部署复杂度；
  * 表结构简单（两张表），用 SQLAlchemy 属于过度设计。
  规模上来后换 PostgreSQL，只需替换本模块的连接层与少量方言差异。

闭环设计（这块是简历第 2 条的核心）：
  报告生成 → 若 requires_review 则自动进入复核队列（pending）
           → 复核员认领（in_review）→ 判定（approved / rejected）
           → 判定结果与原因分类落库
  关键点：**复核结论会覆盖模型结论**，成为该单据的权威记录（final_conclusion）。
  这样"人工复核"不是走形式，而是真正修正系统输出的环节。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent / "data" / "qc_agent.db"

# 复核驳回原因分类：这些分类是后续做 Bad Case 归因与提示词优化的输入
REJECT_CATEGORIES: dict[str, str] = {
    "misclassify": "缺陷类别判错",
    "grade_wrong": "严重等级判错",
    "miss_defect": "漏检缺陷",
    "false_positive": "误检（实际无缺陷）",
    "clause_wrong": "判据条款引用错误",
    "evidence_weak": "证据不足不应下结论",
    "other": "其他",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    report_id           TEXT PRIMARY KEY,
    created_at          TEXT NOT NULL,
    batch_no            TEXT DEFAULT '',
    image_path          TEXT NOT NULL,
    gsd_mm_per_px       REAL DEFAULT 0.05,
    session_id          TEXT DEFAULT '',
    conclusion          TEXT NOT NULL,
    grade               TEXT NOT NULL,
    defect_count        INTEGER DEFAULT 0,
    requires_review     INTEGER DEFAULT 0,
    confidence          REAL DEFAULT 0.0,
    report_json         TEXT NOT NULL,
    usage_json          TEXT DEFAULT '{}',
    -- 人工复核后的权威结论；未复核时与模型结论一致
    final_conclusion    TEXT DEFAULT NULL,
    final_grade         TEXT DEFAULT NULL,
    reviewed            INTEGER DEFAULT 0,
    reviewed_at         TEXT DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id      TEXT PRIMARY KEY,
    report_id      TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'pending',   -- pending | in_review | decided
    created_at     TEXT NOT NULL,
    claimed_by     TEXT DEFAULT '',
    claimed_at     TEXT DEFAULT NULL,
    reviewer       TEXT DEFAULT '',
    verdict        TEXT DEFAULT NULL,                 -- approved | rejected
    category       TEXT DEFAULT NULL,                 -- 驳回原因分类
    comment        TEXT DEFAULT '',
    corrected_conclusion TEXT DEFAULT NULL,
    corrected_grade      TEXT DEFAULT NULL,
    decided_at     TEXT DEFAULT NULL,
    FOREIGN KEY (report_id) REFERENCES reports(report_id)
);

CREATE INDEX IF NOT EXISTS idx_reviews_status  ON reviews(status);
CREATE INDEX IF NOT EXISTS idx_reviews_report  ON reviews(report_id);
CREATE INDEX IF NOT EXISTS idx_reports_batch   ON reports(batch_no);
CREATE INDEX IF NOT EXISTS idx_reports_created ON reports(created_at);
"""

# SQLite 默认禁止跨线程使用同一连接。FastAPI 的同步/异步混合调用下，
# 用「每线程独立连接」最省心；check_same_thread=False + 锁是另一种选择。
_local = threading.local()
_write_lock = threading.Lock()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def get_conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH, timeout=10.0)
        conn.row_factory = sqlite3.Row
        # WAL 模式：读写并发更好，且崩溃后不易损坏
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return conn


def init_db() -> None:
    with _write_lock:
        conn = get_conn()
        conn.executescript(_SCHEMA)
        conn.commit()


# --------------------------------------------------------------------------
# 报告
# --------------------------------------------------------------------------


def save_report(
    *,
    report_id: str,
    image_path: str,
    report: dict[str, Any],
    usage: dict[str, Any] | None = None,
    batch_no: str = "",
    session_id: str = "",
    gsd_mm_per_px: float = 0.05,
) -> dict[str, Any]:
    """写入报告；若需要复核则同时创建复核任务。返回落库后的记录摘要。"""
    created_at = _now()
    requires_review = bool(report.get("requires_review"))

    with _write_lock:
        conn = get_conn()
        conn.execute(
            """
            INSERT OR REPLACE INTO reports (
                report_id, created_at, batch_no, image_path, gsd_mm_per_px, session_id,
                conclusion, grade, defect_count, requires_review, confidence,
                report_json, usage_json, final_conclusion, final_grade, reviewed, reviewed_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                report_id, created_at, batch_no, image_path, gsd_mm_per_px, session_id,
                report.get("conclusion", ""), report.get("grade", ""),
                int(report.get("defect_count", 0)), int(requires_review),
                float(report.get("confidence", 0.0)),
                json.dumps(report, ensure_ascii=False),
                json.dumps(usage or {}, ensure_ascii=False),
                # 未复核时，权威结论 = 模型结论
                None, None, 0, None,
            ),
        )

        review_id = None
        if requires_review:
            review_id = f"V{uuid.uuid4().hex[:10].upper()}"
            conn.execute(
                """
                INSERT INTO reviews (review_id, report_id, status, created_at)
                VALUES (?,?,?,?)
                """,
                (review_id, report_id, "pending", created_at),
            )
        conn.commit()

    return {
        "report_id": report_id,
        "created_at": created_at,
        "requires_review": requires_review,
        "review_id": review_id,
    }


def get_report(report_id: str) -> dict[str, Any] | None:
    conn = get_conn()
    row = conn.execute("SELECT * FROM reports WHERE report_id = ?", (report_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["report"] = json.loads(d.pop("report_json"))
    d["usage"] = json.loads(d.pop("usage_json") or "{}")
    d["requires_review"] = bool(d["requires_review"])
    d["reviewed"] = bool(d["reviewed"])
    d["review"] = get_review_by_report(report_id)
    return d


def list_reports(limit: int = 20, batch_no: str | None = None) -> list[dict[str, Any]]:
    conn = get_conn()
    if batch_no:
        rows = conn.execute(
            "SELECT * FROM reports WHERE batch_no = ? ORDER BY created_at DESC LIMIT ?",
            (batch_no, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM reports ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()

    out = []
    for row in rows:
        d = dict(row)
        d["report"] = json.loads(d.pop("report_json"))
        d.pop("usage_json", None)
        d["requires_review"] = bool(d["requires_review"])
        d["reviewed"] = bool(d["reviewed"])
        out.append(d)
    return out


# --------------------------------------------------------------------------
# 复核
# --------------------------------------------------------------------------


def get_review_by_report(report_id: str) -> dict[str, Any] | None:
    row = get_conn().execute(
        "SELECT * FROM reviews WHERE report_id = ? ORDER BY created_at DESC LIMIT 1",
        (report_id,),
    ).fetchone()
    return dict(row) if row else None


def list_reviews(status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """复核队列。默认返回待处理（pending + in_review）。"""
    conn = get_conn()
    if status:
        rows = conn.execute(
            "SELECT * FROM reviews WHERE status = ? ORDER BY created_at ASC LIMIT ?",
            (status, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM reviews WHERE status IN ('pending','in_review') "
            "ORDER BY created_at ASC LIMIT ?",
            (limit,),
        ).fetchall()

    out = []
    for row in rows:
        v = dict(row)
        rep = conn.execute(
            "SELECT image_path, conclusion, grade, confidence, batch_no, report_json "
            "FROM reports WHERE report_id = ?",
            (v["report_id"],),
        ).fetchone()
        if rep:
            v["image_path"] = rep["image_path"]
            v["conclusion"] = rep["conclusion"]
            v["grade"] = rep["grade"]
            v["confidence"] = rep["confidence"]
            v["batch_no"] = rep["batch_no"]
            try:
                v["review_reason"] = json.loads(rep["report_json"]).get("review_reason", "")
            except Exception:  # noqa: BLE001
                v["review_reason"] = ""
        out.append(v)
    return out


def review_stats() -> dict[str, Any]:
    """复核统计：队列积压、通过率、驳回原因分布。

    驳回原因分布是**优化提示词与工具的输入** —— 哪类错误最多，
    下一轮就往哪里投入。这构成了"使用 → 归因 → 改进"的闭环。
    """
    conn = get_conn()

    def scalar(sql: str, args: tuple = ()) -> int:
        r = conn.execute(sql, args).fetchone()
        return int(r[0]) if r and r[0] is not None else 0

    total_reports = scalar("SELECT COUNT(*) FROM reports")
    pending = scalar("SELECT COUNT(*) FROM reviews WHERE status = 'pending'")
    in_review = scalar("SELECT COUNT(*) FROM reviews WHERE status = 'in_review'")
    approved = scalar("SELECT COUNT(*) FROM reviews WHERE verdict = 'approved'")
    rejected = scalar("SELECT COUNT(*) FROM reviews WHERE verdict = 'rejected'")

    cat_rows = conn.execute(
        "SELECT category, COUNT(*) AS n FROM reviews "
        "WHERE verdict = 'rejected' AND category IS NOT NULL "
        "GROUP BY category ORDER BY n DESC"
    ).fetchall()
    categories = [
        {"category": r["category"], "label": REJECT_CATEGORIES.get(r["category"], r["category"]), "count": r["n"]}
        for r in cat_rows
    ]

    decided = approved + rejected
    decided_total = decided
    return {
        "total_reports": total_reports,
        "queue": {"pending": pending, "in_review": in_review, "total_open": pending + in_review},
        "decided": {"approved": approved, "rejected": rejected, "total": decided_total},
        "approval_rate": round(approved / decided, 4) if decided_total else None,
        "reject_categories": categories,
    }


def claim_review(review_id: str, reviewer: str) -> dict[str, Any]:
    """认领复核任务（pending -> in_review），避免多人重复处理同一单。"""
    with _write_lock:
        conn = get_conn()
        row = conn.execute("SELECT * FROM reviews WHERE review_id = ?", (review_id,)).fetchone()
        if row is None:
            raise KeyError(f"复核任务不存在: {review_id}")
        if row["status"] == "decided":
            raise ValueError(f"该任务已完成判定，无法认领: {review_id}")
        if row["status"] == "in_review" and row["claimed_by"] != reviewer:
            raise ValueError(f"该任务已被 {row['claimed_by']} 认领")

        conn.execute(
            "UPDATE reviews SET status='in_review', claimed_by=?, claimed_at=? WHERE review_id=?",
            (reviewer, _now(), review_id),
        )
        conn.commit()
        return dict(conn.execute("SELECT * FROM reviews WHERE review_id=?", (review_id,)).fetchone())


def decide_review(
    *,
    review_id: str,
    reviewer: str,
    verdict: str,
    category: str = "",
    comment: str = "",
    corrected_conclusion: str = "",
    corrected_grade: str = "",
) -> dict[str, Any]:
    """提交复核判定，并把人工结论写入报告表作为权威记录。"""
    if verdict not in ("approved", "rejected"):
        raise ValueError(f"verdict 必须是 approved 或 rejected，收到: {verdict}")
    if verdict == "rejected" and category and category not in REJECT_CATEGORIES:
        raise ValueError(f"未知的驳回原因分类: {category}（可选：{list(REJECT_CATEGORIES)}）")

    now = _now()
    with _write_lock:
        conn = get_conn()
        row = conn.execute("SELECT * FROM reviews WHERE review_id = ?", (review_id,)).fetchone()
        if row is None:
            raise KeyError(f"复核任务不存在: {review_id}")
        if row["status"] == "decided":
            raise ValueError(f"该任务已完成判定: {review_id}")

        report_id = row["report_id"]
        rep = conn.execute(
            "SELECT conclusion, grade FROM reports WHERE report_id = ?", (report_id,)
        ).fetchone()

        # 人工结论优先：未提供修正值时沿用模型结论
        final_conclusion = corrected_conclusion or (rep["conclusion"] if rep else "")
        final_grade = corrected_grade or (rep["grade"] if rep else "")

        conn.execute(
            """
            UPDATE reviews SET status='decided', reviewer=?, verdict=?, category=?,
                   comment=?, corrected_conclusion=?, corrected_grade=?, decided_at=?
            WHERE review_id=?
            """,
            (
                reviewer, verdict, category or None, comment,
                corrected_conclusion or None, corrected_grade or None, now, review_id,
            ),
        )
        conn.execute(
            """
            UPDATE reports SET reviewed=1, reviewed_at=?, final_conclusion=?, final_grade=?
            WHERE report_id=?
            """,
            (now, final_conclusion, final_grade, report_id),
        )
        conn.commit()

    return {
        "review_id": review_id,
        "report_id": report_id,
        "verdict": verdict,
        "category": category,
        "final_conclusion": final_conclusion,
        "final_grade": final_grade,
        "decided_at": now,
        "overridden": bool(corrected_conclusion and rep and corrected_conclusion != rep["conclusion"]),
    }


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    init_db()
    print(f"数据库已初始化: {DB_PATH}")

    # 自测：写入一份需复核的报告 -> 认领 -> 驳回 -> 看统计
    demo = {
        "conclusion": "需复核", "grade": "major", "defect_count": 1,
        "confidence": 0.35, "requires_review": True,
        "review_reason": "等级为 major 且置信度低于阈值",
    }
    r = save_report(report_id="RTEST001", image_path="demo.png", report=demo,
                    usage={"input_tokens": 100}, batch_no="BTEST")
    print("写入报告:", r)

    queue = list_reviews()
    print(f"队列长度: {len(queue)}")
    rid = queue[0]["review_id"]

    print("认领:", claim_review(rid, "reviewer_a")["status"])
    out = decide_review(review_id=rid, reviewer="reviewer_a", verdict="rejected",
                        category="clause_wrong", comment="条款引用错误，应为 6.1.2",
                        corrected_conclusion="不合格", corrected_grade="critical")
    print("判定:", out)

    print("\n统计:")
    print(json.dumps(review_stats(), ensure_ascii=False, indent=2))
