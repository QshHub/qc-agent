# -*- coding: utf-8 -*-
"""共享定义：Agent 提示词、报告 Schema、工具集工厂

## 为什么独立成文件

原先 `InspectionReport` / `SYSTEM_PROMPT` / `build_toolkit` 都定义在 `qc_api.py` 里。
LangGraph 路线的 judge 节点也要用它们，结果变成：

    qc_graph -> qc_judge -> qc_api -> fastapi / uvicorn / dotenv

**只为了拿一个 Pydantic 模型，却把整个 Web 服务栈拖了进来**，导致
图编排层无法脱离 HTTP 服务独立测试，也让部署时的依赖关系变脏。

抽到本模块后依赖关系变成：

    qc_api    ─┐
               ├─→ qc_agent_shared  （只依赖 agentscope + pydantic + qc_tools）
    qc_judge  ─┘

于是 LangGraph 路线只需 langgraph + agentscope + pydantic + pillow + numpy，
不需要 fastapi。这是"按依赖方向分层"的具体实践。
"""

from __future__ import annotations

import os
from typing import Any

from pydantic import BaseModel, Field

from qc_knowledge import list_taxonomy, lookup_standard, match_taxonomy_by_severity
from qc_tools import get_image_info, segment_defects

# --------------------------------------------------------------------------
# Agent 基础配置
# --------------------------------------------------------------------------

AGENT_NAME = "质检助手"

MODEL_NAME = os.getenv("QC_MODEL", "deepseek-flash")

SYSTEM_PROMPT = """你是一名工业质检助手，服务于蜂窝催化剂产线。

## 工作流程（必须遵守）
1. 用户提供图像路径时，**先调用 segment_defects** 获取真实检测数据；
2. 拿到缺陷数据后，**调用 lookup_standard** 查询对应缺陷类别的判据条款；
3. 依据"实测数据 + 判据条款"给出判定，严禁凭猜测描述图像内容；
4. 若工具返回"未发现缺陷"，判定为合格，不要臆造问题；
5. **若工具返回以「检测失败」开头的内容，说明图像未被成功读取、没有任何测量数据。
   此时结论必须填「无法判定」，并在 decision_reason 中说明读取失败的原因，
   绝不允许给出"合格"结论。** 把"没检到缺陷"和"没能读取图像"混为一谈是严重错误；
6. 若信息不足或工具失败，如实说明，不得编造结论与条款号。

## 缺陷分类体系（3 大类 12 小类）
- 材料缺失类：孔道缺失、端面缺损、涂层脱落、局部塌陷
- 缝隙分离类：轴向裂缝、环向裂缝、界面剥离、边缘开裂
- 结构形变类：整体变形、孔道扭曲、端面倾斜、尺寸超差

## 严重等级（以工具实测的数值为准）
- critical：面积 > 50 mm² 或最大尺寸 > 20 mm
- major：面积 > 20 mm² 或最大尺寸 > 8 mm
- minor：其余

## 复核触发条件（满足任一即需人工复核）
- 缺陷等级为 critical 或 major
- 工具返回的平均置信度低于 0.7
- 判据条款中对当前缺陷的适用性存在歧义

## 回答要求
先给结论，再列依据（引用工具返回的具体数字与条款号），最后给处置建议。
简洁专业，使用质检术语。
"""


# --------------------------------------------------------------------------
# 结构化报告 Schema
# --------------------------------------------------------------------------


class ReportDefect(BaseModel):
    """报告中的单条缺陷记录。"""

    defect_type: str = Field(..., description="缺陷类别键，如 gap_separation.axial_crack")
    defect_name: str = Field(..., description="缺陷中文名，如 轴向裂缝")
    severity: str = Field(..., description="严重等级：critical 或 major 或 minor")
    area_mm2: float = Field(..., description="缺陷面积（mm²），取自工具实测值")
    max_length_mm: float = Field(..., description="缺陷最大尺寸（mm），取自工具实测值")
    location: str = Field(..., description="缺陷位置描述，如 端面中偏左")
    standard_ref: str = Field(..., description="引用判据条款号，如 Q/QC-JS-002 6.1.1")


class InspectionReport(BaseModel):
    """质检报告：本模型同时充当 LLM 的结构化输出 schema。

    字段描述会作为 schema 的一部分传给模型，因此写得越明确，模型填得越准。
    这是"用类型系统约束模型输出"的实践。
    """

    conclusion: str = Field(
        ...,
        description=(
            "总体结论，只能取以下四个值之一："
            "合格 / 不合格 / 需复核 / 无法判定。"
            "当图像不可读或检测工具执行失败、没有任何可用测量数据时，"
            "必须填「无法判定」，不得用「合格」代替，也不要用「需复核」"
            "（「需复核」仅用于确实检出缺陷但需人工确认的情形）。"
        ),
    )
    grade: str = Field(..., description="综合严重等级：critical / major / minor / none")
    defect_count: int = Field(..., description="检出缺陷数量")
    defects: list[ReportDefect] = Field(default_factory=list, description="缺陷明细列表")

    measurement_basis: str = Field(..., description="测量依据，说明数据来自哪个工具的哪次调用")
    standard_basis: str = Field(..., description="判定依据，引用的标准条款号及原文要点")
    decision_reason: str = Field(..., description="判定理由，串联实测数据与判据条款的推理过程")

    disposal: str = Field(..., description="处置建议，明确放行/判废/返修/降级")
    process_hint: str = Field(default="", description="工艺追溯方向，如涉及某工序参数")

    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="结论置信度 0~1。证据充分取 0.9 以上，证据薄弱或条款适用存疑时取 0.5 以下",
    )
    requires_review: bool = Field(
        ..., description="是否需要人工复核：等级为 major/critical、或置信度低于 0.7 时为 true"
    )
    review_reason: str = Field(default="", description="需复核时说明原因，不需要则留空")


# --------------------------------------------------------------------------
# 工具集工厂
# --------------------------------------------------------------------------


def _allow_all() -> Any:
    """质检工具是只读的，直接放行，避免 API 调用阻塞在人工确认上。

    （FunctionTool 的 permission 不传时默认 ASK，在无人值守的服务里会挂起请求）
    """
    from agentscope.permission import PermissionBehavior, PermissionDecision

    return PermissionDecision(
        behavior=PermissionBehavior.ALLOW,
        message="质检工具为只读操作，自动放行。",
    )


def build_toolkit() -> Any:
    """注册质检工具集：视觉工具 + 判据库工具。"""
    from agentscope.tool import FunctionTool, Toolkit

    return Toolkit(
        tools=[
            # 视觉工具
            FunctionTool(segment_defects, is_read_only=True, permission=_allow_all()),
            FunctionTool(get_image_info, is_read_only=True, permission=_allow_all()),
            # 判据库工具
            FunctionTool(list_taxonomy, is_read_only=True, permission=_allow_all()),
            FunctionTool(lookup_standard, is_read_only=True, permission=_allow_all()),
            FunctionTool(match_taxonomy_by_severity, is_read_only=True, permission=_allow_all()),
        ]
    )


__all__ = [
    "AGENT_NAME",
    "MODEL_NAME",
    "SYSTEM_PROMPT",
    "InspectionReport",
    "ReportDefect",
    "build_toolkit",
]
