# -*- coding: utf-8 -*-
"""judge 节点实现：把确定性实测数据交给 Agent，产出结构化报告

## 为什么单独成文件

LangGraph 的节点必须是**同步函数**（图执行在独立执行器里跑），
而 AgentScope 的 `Agent.reply` 是异步的。两者衔接需要在独立线程里
创建事件循环，且不能破坏调用方已有的事件循环（FastAPI 里已有）。

这里暴露一个同步入口 `run_judge()`，内部处理：
  * 检测当前线程是否已有运行中的事件循环；
  * 若有，则把协程丢到新线程的独立事件循环里执行（避免 "event loop is already running"）；
  * 若无，直接 asyncio.run。

## 与 qc_api.py 的关系

共用同一套 system prompt 与 `InspectionReport` schema（从 qc_api 导入），
保证两条路线（AgentScope 直接调用 / LangGraph 编排）产出的报告结构一致，
评测指标才有可比性。若日后拆分部署，可把这两个定义抽到 qc_schema.py。
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from typing import Any

from qcagent.core.knowledge import GRADE_RULES
from qcagent.core.tools import segment_defects

# 共享定义：prompt、报告 schema、工具集工厂
# （从 qc_api 抽出来，避免为了一个 Pydantic 模型把 FastAPI 全家桶拖进 LangGraph 路线）
from qcagent.graph.shared import AGENT_NAME, MODEL_NAME, InspectionReport, SYSTEM_PROMPT, build_toolkit  # noqa: F401


def _build_evidence_block(
    image_path: str,
    measurement: dict[str, Any],
    standard: dict[str, Any],
) -> str:
    """把确定性数据整理成 Agent 的输入。

    关键：实测数值直接写进 prompt，Agent 不需要再算，也不需要重复调工具
    （judge 节点在图里是"解释与归纳"环节）。这既省 token 又避免重复推理。
    """
    lines = [
        "# 质检任务",
        f"图像路径：{image_path}",
        "",
        "## 一、缺陷实测数据（由分割工具产出，确定可信）",
    ]

    if not measurement.get("defects"):
        lines.append("未检出任何缺陷（缺陷数 = 0）。")
    else:
        lines.append(
            f"共 {measurement['defect_count']} 处缺陷，"
            f"总面积 {measurement['total_area_mm2']} mm²，"
            f"最高等级 {measurement['max_severity']}，"
            f"平均置信度 {measurement['mean_confidence']}，"
            f"相机分辨率 {measurement['gsd_mm_per_px']} mm/px。"
        )
        for d in measurement["defects"]:
            lines.append(
                f"  - {d['instance_id']}: 位置 {tuple(d['bbox_px'])}，"
                f"面积 {d['area_mm2']} mm²，最大尺寸 {d['max_length_mm']} mm，"
                f"等级 {d['severity']}，置信度 {d['confidence']}"
            )

    lines += ["", "## 二、判据信息（来自内置判据库）"]
    if standard.get("matched") or standard.get("candidates"):
        if standard.get("candidates"):
            lines.append("候选判据类别：")
            for c in standard["candidates"]:
                lines.append(f"  - {c}")
        if standard.get("rule_text"):
            lines.append("")
            lines.append("默认判据原文：")
            lines.append(standard["rule_text"])
    else:
        lines.append(standard.get("reason", "无匹配判据"))

    lines += ["", "## 三、等级规则"]
    for g in GRADE_RULES:
        lines.append(f"  - {g['grade']}（{g['label']}）：{g['condition']} → {g['action']}")

    lines += [
        "",
        "## 四、报告填写要求",
        "1. 缺陷面积与尺寸**必须原样引用上面的实测值**，不得四舍五入或推算；",
        "2. 缺陷类别需与判据库的类别名称一致；",
        f"3. 若实测数据缺失或执行失败，conclusion 填「无法判定」；",
        "4. 需要人工复核时在 review_reason 中写明具体触发条件。",
    ]
    return "\n".join(lines)


async def _run_async(prompt: str) -> dict[str, Any]:
    """在独立事件循环里跑一次 Agent，返回结构化报告 dict。"""
    from agentscope.agent import Agent
    from agentscope.credential import DeepSeekCredential
    from agentscope.message import Msg, TextBlock
    from agentscope.model import DeepSeekChatModel

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY")

    agent = Agent(
        name=AGENT_NAME,
        system_prompt=SYSTEM_PROMPT,
        model=DeepSeekChatModel(
            credential=DeepSeekCredential(api_key=api_key),
            model=MODEL_NAME,
            stream=False,
        ),
        toolkit=build_toolkit(),
    )

    reply = await agent.reply(
        Msg(name="user", role="user", content=[TextBlock(type="text", text=prompt)]),
        structured_schema=InspectionReport,
    )

    raw = getattr(reply, "structured_output", None)
    if raw is None:
        raw = getattr(
            getattr(getattr(agent, "state", None), "reply_context", None),
            "structured_output",
            None,
        )
    if raw is None:
        raise RuntimeError(
            "未能取得结构化报告。回复文本：" + str(reply.get_text_content())[:300]
        )
    return InspectionReport.model_validate(raw).model_dump()


def _in_running_loop() -> bool:
    try:
        asyncio.get_running_loop()
        return True
    except RuntimeError:
        return False


def run_judge(
    *,
    image_path: str,
    measurement: dict[str, Any],
    standard: dict[str, Any],
    batch_no: str = "",
) -> dict[str, Any]:
    """同步入口：LangGraph 节点直接调用。"""
    prompt = _build_evidence_block(image_path, measurement, standard)
    if batch_no:
        prompt += f"\n\n批次号：{batch_no}"

    if not _in_running_loop():
        return asyncio.run(_run_async(prompt))

    # 已有运行中的事件循环（比如被 FastAPI 的 async 端点间接调用）：
    # 在独立线程里新建事件循环执行，避免嵌套冲突。
    result: dict[str, Any] = {}
    error: list[BaseException] = []

    def _worker() -> None:
        try:
            result.update(asyncio.run(_run_async(prompt)))
        except BaseException as exc:  # noqa: BLE001
            error.append(exc)

    t = threading.Thread(target=_worker, name="qc-judge", daemon=True)
    t.start()
    t.join(timeout=300)
    if t.is_alive():
        raise TimeoutError("judge 节点执行超时（300s）")
    if error:
        raise error[0]
    return result


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    from qcagent.core.knowledge import lookup_standard
    from qcagent.core.tools import annotate_defects

    sample = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "samples", "sample_crack.png")
    m = annotate_defects(sample, 0.05)
    s = {"matched": True, "rule_text": lookup_standard("轴向裂缝"),
         "candidates": ["轴向裂缝（Q/QC-JS-002 6.1.1）"]}
    print("构造的 evidence prompt：")
    print(_build_evidence_block(sample, m, s))
    print("\n" + "=" * 60)
    report = run_judge(image_path=sample, measurement=m, standard=s, batch_no="BTEST")
    print("报告：")
    print(json.dumps(report, ensure_ascii=False, indent=2)[:1200])
