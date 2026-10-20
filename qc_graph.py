# -*- coding: utf-8 -*-
"""质检业务编排层（LangGraph 路线）

## 为什么要有这一层

当前实现的 AgentScope `Agent` 负责"单轮内的推理与工具调用"，它擅长的是
**局部决策**。但质检是一条有明确阶段、需要断点续检、且必须有人工闸门的
**业务流程**——这正是 LangGraph 的定位：

| 能力 | AgentScope 单独 | 加上 LangGraph |
|---|---|---|
| 阶段划分 | Prompt 里约定，模型可能跳步 | **图结构强制**，节点不可跳过 |
| 断点续检 | 无（进程重启即丢） | **Checkpointer** 逐阶段落库，可按 thread_id 恢复 |
| 人工介入 | 事后复核（生成完再改判） | **interrupt 真正挂起**，状态冻结等待人工，可恢复继续 |
| 条件路由 | 模型自行判断 | **条件边**显式声明，可测试、可审计 |

## 五阶段图

    START → validate → perceive → lookup → judge → gate ─┬─→ archive → END
                                        ↑                 └─（需复核时）interrupt 挂起
                                        └──── resume（人工判定后继续）

* validate  图像存在性与可读性检查（确定性，不调模型）
* perceive  调用分割后端，产出缺陷实测数据（确定性）
* lookup    按缺陷类别查判据条款（确定性）
* judge     把实测数据与判据交给 Agent，产出结构化报告（唯一调用 LLM 的节点）
* gate      依 requires_review 决定是否 interrupt
* archive   落库并标记复核结论

设计要点：**只有 judge 一个节点依赖 LLM**。其余节点确定性，这带来两个好处：
  1. 多数失败原因可以被精确定位（是读图失败、判据缺失，还是模型判断问题）；
  2. 断点恢复时不必重跑 LLM 调用，省钱且更快。
"""

from __future__ import annotations

import uuid
from typing import Any, Literal, TypedDict

from qc_knowledge import DEFECT_TAXONOMY, lookup_standard
from qc_tools import annotate_defects

try:
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover —— 未安装时给出明确提示而非 ImportError 崩溃
    LANGGRAPH_AVAILABLE = False
    InMemorySaver = None  # type: ignore[assignment]
    StateGraph = None  # type: ignore[assignment]
    END = START = None  # type: ignore[assignment]
    Command = None  # type: ignore[assignment]
    interrupt = None  # type: ignore[assignment]


# --------------------------------------------------------------------------
# 状态定义
# --------------------------------------------------------------------------


class QCState(TypedDict, total=False):
    """图状态。

    LangGraph 会把每个节点返回的 dict 合并进该状态，并**逐阶段写入 checkpointer**。
    因此状态里只放可 JSON 序列化的数据，不放模型对象或数据库连接
    —— 放了会导致 checkpoint 无法序列化。
    """

    # 输入
    image_path: str
    gsd_mm_per_px: float
    batch_no: str

    # 各阶段产出
    image_size_px: list[int]
    measurement: dict[str, Any]      # segment 结果（确定性）
    standard: dict[str, Any]         # 判据（确定性）
    report: dict[str, Any]           # 结构化报告（LLM 产出）

    # 流程控制与审计
    stage: str
    error: str
    needs_review: bool
    review_id: str
    report_id: str
    trace: list[dict[str, Any]]      # 每阶段一条，用于失败归因


def _trace_update(state: QCState, stage: str, status: str, detail: str = "") -> list[dict[str, Any]]:
    """追加一条阶段轨迹。轨迹随 checkpoint 持久化，可事后回放。"""
    trace = list(state.get("trace") or [])
    trace.append({"stage": stage, "status": status, "detail": detail})
    return trace


# --------------------------------------------------------------------------
# 节点实现
# --------------------------------------------------------------------------


def node_validate(state: QCState) -> dict[str, Any]:
    """图像校验：只做确定性检查，不调用模型。

    这一步先把"读不到图"挡在外面。否则下游 Agent 会把"没读到图"
    误当成"没有缺陷"而放行 —— 这是质检系统最危险的静默失效。
    """
    from pathlib import Path

    from PIL import Image

    path = state["image_path"]
    try:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"图像不存在: {path}")
        with Image.open(p) as im:
            size = [int(im.size[0]), int(im.size[1])]
            im.verify()
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        return {
            "stage": "validate",
            "error": msg,
            "trace": _trace_update(state, "validate", "failed", msg),
        }

    return {
        "stage": "validate",
        "image_size_px": size,
        "error": "",
        "trace": _trace_update(state, "validate", "ok", f"{size[0]}x{size[1]}"),
    }


def node_perceive(state: QCState) -> dict[str, Any]:
    """缺陷感知：调用分割后端。确定性节点。"""
    if state.get("error"):
        return {"stage": "perceive", "trace": _trace_update(state, "perceive", "skipped", "前置阶段失败")}

    try:
        result = annotate_defects(state["image_path"], state.get("gsd_mm_per_px", 0.05))
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        return {
            "stage": "perceive",
            "error": msg,
            "trace": _trace_update(state, "perceive", "failed", msg),
        }

    return {
        "stage": "perceive",
        "measurement": result,
        "trace": _trace_update(
            state, "perceive", "ok",
            f"{result['defect_count']} 处缺陷，最高等级 {result['max_severity']}",
        ),
    }


def node_lookup(state: QCState) -> dict[str, Any]:
    """判据检索：按缺陷特征匹配判据条款。确定性节点。

    当前用内置规则库；未来换成向量检索时，只替换本函数内部实现，
    状态结构与下游节点完全不受影响。
    """
    if state.get("error") or not state.get("measurement"):
        return {"stage": "lookup", "trace": _trace_update(state, "lookup", "skipped", "无测量数据")}

    measurement = state["measurement"]
    if not measurement.get("defects"):
        return {
            "stage": "lookup",
            "standard": {"matched": False, "reason": "未检出缺陷，无需引用判据"},
            "trace": _trace_update(state, "lookup", "ok", "无缺陷，跳过判据"),
        }

    # 取最严重的那条缺陷去匹配判据（决定整体定级）
    order = {"critical": 3, "major": 2, "minor": 1}
    top = max(measurement["defects"], key=lambda d: order.get(d.get("severity", ""), 0))

    # 缺陷类别由 Agent 在 judge 阶段判定；此处先按"最大缺陷"给出候选判据集
    candidates = {k: v for k, v in DEFECT_TAXONOMY.items()
                  if v["category"] == "缝隙分离类"}  # 参考实现默认嫌疑类别

    standard = {
        "matched": True,
        "suspect_category": "缝隙分离类",
        "candidates": [f"{v['name']}（{v['standard']} {v['clause']}）" for v in candidates.values()],
        "top_defect": top,
        "rule_text": lookup_standard("轴向裂缝"),  # 参考实现：默认按裂缝判据
    }
    return {
        "stage": "lookup",
        "standard": standard,
        "trace": _trace_update(state, "lookup", "ok", f"候选判据 {len(candidates)} 条"),
    }


def node_judge(state: QCState) -> dict[str, Any]:
    """结论判定：**唯一调用 LLM 的节点**。

    把确定性的实测数据与判据作为输入交给 Agent，要求它输出结构化报告。
    这样模型只负责"解释与归纳"，不负责产数值 —— 数值由上游节点保证。
    """
    if state.get("error") or not state.get("measurement"):
        return {
            "stage": "judge",
            "needs_review": True,
            "trace": _trace_update(state, "judge", "skipped", "前置阶段失败，转人工"),
        }

    from qc_judge import run_judge

    try:
        report = run_judge(
            image_path=state["image_path"],
            measurement=state["measurement"],
            standard=state.get("standard") or {},
            batch_no=state.get("batch_no", ""),
        )
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        return {
            "stage": "judge",
            "error": msg,
            "needs_review": True,
            "trace": _trace_update(state, "judge", "failed", msg),
        }

    return {
        "stage": "judge",
        "report": report,
        "needs_review": bool(report.get("requires_review")),
        "trace": _trace_update(
            state, "judge", "ok",
            f"结论={report.get('conclusion')} 等级={report.get('grade')}",
        ),
    }


def node_gate(state: QCState) -> dict[str, Any]:
    """人工闸门：需要复核时**真正挂起**，状态冻结在 checkpointer 中。

    `interrupt()` 的语义（LangGraph 1.x）：
      * 首次执行到这里会中断，把传给它的 payload 暴露给调用方；
      * 调用方用 `Command(resume=<决策>)` 再次 invoke 同一个 thread_id，
        interrupt() 的返回值就是这里的 resume 值 —— 于是后续代码继续执行。
    这比"生成完再改判"更严格：**未经人工确认，归档节点根本不会执行**。
    """
    if state.get("error") or state.get("needs_review"):
        report = state.get("report") or {}
        decision = interrupt({
            "type": "qc_review_required",
            "image_path": state["image_path"],
            "batch_no": state.get("batch_no", ""),
            "error": state.get("error", ""),
            "conclusion": report.get("conclusion", "无法判定"),
            "grade": report.get("grade", "none"),
            "review_reason": report.get("review_reason") or state.get("error", "前置阶段失败"),
            "defects": (state.get("measurement") or {}).get("defects", []),
            "trace": state.get("trace", []),
        })

        # 人工恢复后进入这里
        verdict = (decision or {}).get("verdict", "approved")
        corrected_conclusion = (decision or {}).get("corrected_conclusion") or report.get("conclusion")
        corrected_grade = (decision or {}).get("corrected_grade") or report.get("grade")

        report = dict(report)
        report["conclusion"] = corrected_conclusion
        report["grade"] = corrected_grade
        report["requires_review"] = False  # 已复核完毕

        return {
            "stage": "gate",
            "report": report,
            "needs_review": False,
            "gate_decision": dict(decision or {}, verdict=verdict),
            "trace": _trace_update(
                state, "gate", "resumed",
                f"人工判定 {verdict}，结论={corrected_conclusion} 等级={corrected_grade}",
            ),
        }

    return {
        "stage": "gate",
        "trace": _trace_update(state, "gate", "auto_passed", "无需复核"),
    }


def node_archive(state: QCState) -> dict[str, Any]:
    """归档：落库并创建复核任务（若尚未被人工处理）。"""
    from qc_store import save_report

    report = state.get("report") or {}

    # 前置阶段失败且人工只做了放行：仍归档，但标明数据不完整
    if not report:
        report = {
            "conclusion": "无法判定",
            "grade": "none",
            "defect_count": 0,
            "defects": [],
            "decision_reason": state.get("error", "流程中断"),
            "requires_review": False,
        }

    report_id = f"R{uuid.uuid4().hex[:12].upper()}"
    review_id = None
    try:
        saved = save_report(
            report_id=report_id,
            image_path=state["image_path"],
            report=report,
            usage={},
            batch_no=state.get("batch_no", ""),
            session_id="",
            gsd_mm_per_px=state.get("gsd_mm_per_px", 0.05),
        )
        review_id = saved.get("review_id")
    except Exception as exc:  # noqa: BLE001
        return {
            "stage": "archive",
            "error": f"归档失败: {exc}",
            "trace": _trace_update(state, "archive", "failed", str(exc)),
        }

    return {
        "stage": "archive",
        "report_id": report_id,
        "report": report,
        "trace": _trace_update(state, "archive", "ok", f"报告 {report_id}"),
    }


# --------------------------------------------------------------------------
# 条件边
# --------------------------------------------------------------------------


def route_after_validate(state: QCState) -> Literal["perceive", "gate"]:
    """校验失败直接进人工闸门，不消耗模型调用。"""
    return "gate" if state.get("error") else "perceive"


def route_after_judge(state: QCState) -> Literal["gate", "archive"]:
    """判定失败或需复核 -> 闸门；否则直接归档。"""
    if state.get("error") or state.get("needs_review"):
        return "gate"
    return "archive"


# --------------------------------------------------------------------------
# 图构建
# --------------------------------------------------------------------------


def build_graph(checkpointer: Any = None):
    """构建并编译质检流程图。

    checkpointer 决定持久化后端：
      * None            -> InMemorySaver（进程内，重启即丢，仅用于测试）
      * SqliteSaver     -> 落盘，支持跨进程断点续检
    """
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError(
            "未安装 langgraph。请执行：pip install langgraph langgraph-checkpoint-sqlite"
        )

    g = StateGraph(QCState)
    g.add_node("validate", node_validate)
    g.add_node("perceive", node_perceive)
    g.add_node("lookup", node_lookup)
    g.add_node("judge", node_judge)
    g.add_node("gate", node_gate)
    g.add_node("archive", node_archive)

    g.add_edge(START, "validate")
    g.add_conditional_edges("validate", route_after_validate, {"perceive": "perceive", "gate": "gate"})
    g.add_edge("perceive", "lookup")
    g.add_edge("lookup", "judge")
    g.add_conditional_edges("judge", route_after_judge, {"gate": "gate", "archive": "archive"})
    g.add_edge("gate", "archive")
    g.add_edge("archive", END)

    return g.compile(checkpointer=checkpointer or InMemorySaver())


def make_thread_id() -> str:
    return f"qc-{uuid.uuid4().hex[:12]}"


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    print("LangGraph 可用:", LANGGRAPH_AVAILABLE)
    if LANGGRAPH_AVAILABLE:
        app = build_graph()
        print("图已编译，节点:", sorted(app.get_graph().nodes.keys()))
        print("\n拓扑：")
        for e in app.get_graph().edges:
            print(f"  {e.source} -> {e.target}" + ("  [条件边]" if e.conditional else ""))
