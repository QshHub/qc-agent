# -*- coding: utf-8 -*-
"""QC-Agent 后端服务（第三阶段：结构化质检报告）

相对第二阶段的三处改动：
  1. 接入知识库工具（list_taxonomy / lookup_standard / match_taxonomy_by_severity）
  2. 新增 `InspectionReport` 结构化 schema + `/qc/report` 接口
  3. 报告落盘归档（第四阶段会替换为数据库）

关于结构化输出（读源码确认的机制）：
  AgentScope 通过内置工具 `GenerateStructuredOutput` 实现，schema 支持
  Pydantic 模型类或 JSON Schema dict。模型填的字段会先经校验，
  **校验失败时错误信息会回传给模型让它自我修正**，这是内置的可靠性机制。
  生成结果同时出现在 `Msg.structured_output` 与 agent state 中。

运行：
    uvicorn qc_api:app --reload --port 8000
"""

from __future__ import annotations

from qcagent import settings  # noqa: E402  统一路径常量

import asyncio
import json
import logging
import os
import sys
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------
# 环境变量加载（必须在读取配置之前）
# --------------------------------------------------------------------------

_ENV_FILE = settings.ENV_FILE
load_dotenv(_ENV_FILE, override=False)

# Windows 控制台默认 GBK，输出含 mm² 等字符会抛 UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001 —— 非终端环境下无 stream 可配置
        pass

from agentscope.agent import Agent  # noqa: E402
from agentscope.credential import DeepSeekCredential  # noqa: E402
from agentscope.message import Msg, TextBlock  # noqa: E402
from agentscope.model import DeepSeekChatModel  # noqa: E402
from agentscope.permission import PermissionBehavior, PermissionDecision  # noqa: E402
from agentscope.tool import FunctionTool, Toolkit  # noqa: E402

from qcagent.core.events import EventTranslator, sse_pack, translate_stream  # noqa: E402
from qcagent.core.knowledge import (  # noqa: E402
    criteria_status,
    list_taxonomy,
    lookup_standard,
    match_taxonomy_by_severity,
)
from qcagent.core.store import (  # noqa: E402
    REJECT_CATEGORIES,
    claim_review,
    decide_review,
    get_report,
    init_db,
    list_reports,
    list_reviews,
    review_stats,
    save_report,
)
from qcagent.core.tools import (  # noqa: E402
    annotate_defects,
    get_image_info,
    make_sample_image,
    segment_defects,
)

# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

MODEL_NAME = os.getenv("QC_MODEL", "deepseek-flash")
MAX_SESSIONS = int(os.getenv("QC_MAX_SESSIONS", "50"))
AGENT_NAME = "质检助手"

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

## 缺陷分类体系（以工具返回的 class_name / defect_type 为准）
工具会直接给出缺陷的类别名与类别键，**不要自行改判类别**。
类别键前缀表示大类：
- `material_loss.*` —— 材料缺失类（孔塌陷、孔洞、端面缺口、孔壁缺失、表皮缺口、缺角、损坏）
- `gap_separation.*` —— 缝隙分离类（端面裂纹、侧壁撕裂、表皮裂纹、横向裂纹）
- `deformation.*` —— 结构形变类（长度不足）

## 严重等级：两套口径，必须按工具提示选择

工具返回的「判定口径」段落会明确告知哪套可信：

**A. 相对化等级（工具字段 `severity_relative`，看「占样品面积/边长」的比例）**
  - 与相机标定无关，**始终可信**，是默认应使用的口径
  - critical：缺陷面积 > 样品面积 6%，或缺陷长边 > 样品边长 25%
  - major：  缺陷面积 > 样品面积 2%，或缺陷长边 > 样品边长 12%
  - minor：  其余

**B. 物理等级（工具字段 `severity`，基于 mm² / mm）**
  - **仅当工具明确写「物理量可信：是」时才可使用**
  - 若写「物理量可信：否」，说明 GSD 未标定，mm² 与 mm 只是按假设值换算的估算，
    **严禁据此判定严重等级**，只能用于横向比较或描述趋势

**冲突处理**：物理量可信时两者都可用；不可信时**只用相对化等级**，
并在 decision_reason 中说明"因相机未标定，采用相对化判定口径"。

## 复核触发条件（满足任一即需人工复核）
- 相对化等级为 critical 或 major
- 工具返回的平均置信度低于 0.7
- 判据条款中对当前缺陷的适用性存在歧义
- 工具提示物理量不可信（说明判定依据本身不够硬）

## 回答要求
先给结论，再列依据（引用工具返回的具体数字与条款号），最后给处置建议。
说明判定用了哪套口径。简洁专业，使用质检术语。
"""

logger = logging.getLogger("qc_agent")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()

# 结构化结果暂存：image_path -> 最近一次检测结果
# 第三阶段会替换为持久化存储（质检单据需要长期留存以备追溯）
RESULT_STORE: dict[str, dict[str, Any]] = {}

# 报告归档目录（第四阶段替换为数据库；这里先落 JSON 文件，保证可追溯）
REPORT_DIR = settings.REPORTS_DIR


# --------------------------------------------------------------------------
# 工具与 Agent 构建
# --------------------------------------------------------------------------


def _allow_all() -> PermissionDecision:
    """质检工具是只读的，直接放行，避免 API 调用阻塞在人工确认上。"""
    return PermissionDecision(
        behavior=PermissionBehavior.ALLOW,
        message="质检工具为只读操作，自动放行。",
    )


def build_toolkit() -> Toolkit:
    """注册质检工具。

    ★ 关键点：FunctionTool 的 permission 不传时默认是 ASK，
      在 HTTP 服务里会导致工具调用一直等待人工确认（表现为请求挂住）。
    """
    return Toolkit(
        tools=[
            # 视觉工具
            FunctionTool(segment_defects, is_read_only=True, permission=_allow_all()),
            FunctionTool(get_image_info, is_read_only=True, permission=_allow_all()),
            # 知识库工具
            FunctionTool(list_taxonomy, is_read_only=True, permission=_allow_all()),
            FunctionTool(lookup_standard, is_read_only=True, permission=_allow_all()),
            FunctionTool(match_taxonomy_by_severity, is_read_only=True, permission=_allow_all()),
        ]
    )


def build_agent(*, stream: bool = False) -> Agent:
    """构建 Agent。

    stream=True 供 SSE 流式接口使用；False 供一次性返回的接口使用。
    注意 DeepSeekChatModel 的 stream 默认是 True，一次性接口必须显式设为 False，
    否则模型返回的是异步生成器而非完整 Msg。
    """
    return Agent(
        name=AGENT_NAME,
        system_prompt=SYSTEM_PROMPT,
        model=DeepSeekChatModel(
            credential=DeepSeekCredential(api_key=API_KEY),
            model=MODEL_NAME,
            stream=stream,
        ),
        toolkit=build_toolkit(),
    )


# --------------------------------------------------------------------------
# 会话管理
# --------------------------------------------------------------------------


class SessionRecord:
    __slots__ = ("agent", "turns", "total_input_tokens", "total_output_tokens")

    def __init__(self, *, stream: bool = False) -> None:
        self.agent = build_agent(stream=stream)
        self.turns = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0


SESSIONS: OrderedDict[str, SessionRecord] = OrderedDict()
# 流式会话单独一个池：模型的 stream 标志在构造时固定，无法一个实例两用。
# 前端如果用同一个 session_id 同时調用流式与非流式接口，两边各自持有 Agent 实例。
STREAM_SESSIONS: OrderedDict[str, SessionRecord] = OrderedDict()


def get_session(session_id: str | None) -> tuple[str, SessionRecord]:
    if session_id and session_id in SESSIONS:
        SESSIONS.move_to_end(session_id)
        return session_id, SESSIONS[session_id]

    if len(SESSIONS) >= MAX_SESSIONS:
        evicted_id, _ = SESSIONS.popitem(last=False)
        logger.info("会话数达上限 %d，淘汰最旧会话 %s", MAX_SESSIONS, evicted_id)

    new_id = session_id or uuid.uuid4().hex
    SESSIONS[new_id] = SessionRecord(stream=False)
    logger.info("新建会话 %s（当前会话数 %d）", new_id, len(SESSIONS))
    return new_id, SESSIONS[new_id]


def get_stream_session(session_id: str | None) -> tuple[str, SessionRecord]:
    if session_id and session_id in STREAM_SESSIONS:
        STREAM_SESSIONS.move_to_end(session_id)
        return session_id, STREAM_SESSIONS[session_id]

    if len(STREAM_SESSIONS) >= MAX_SESSIONS:
        evicted_id, _ = STREAM_SESSIONS.popitem(last=False)
        logger.info("流式会话数达上限 %d，淘汰最旧会话 %s", MAX_SESSIONS, evicted_id)

    new_id = session_id or uuid.uuid4().hex
    STREAM_SESSIONS[new_id] = SessionRecord(stream=True)
    logger.info("新建流式会话 %s（当前 %d）", new_id, len(STREAM_SESSIONS))
    return new_id, STREAM_SESSIONS[new_id]


# --------------------------------------------------------------------------
# FastAPI 应用
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("QC-Agent 启动，模型=%s，会话上限=%d", MODEL_NAME, MAX_SESSIONS)
    if API_KEY:
        logger.info("API Key 已加载，长度 %d", len(API_KEY))
    else:
        logger.error("API Key 未加载，/chat 将返回 502。查找路径: %s", _ENV_FILE)
    try:
        build_toolkit()
        logger.info("工具集加载成功")
    except Exception:  # noqa: BLE001
        logger.exception("工具集加载失败，/chat 中的工具调用将不可用")
    try:
        init_db()
        logger.info("数据库已就绪，复核队列待处理 %d 单", review_stats()["queue"]["total_open"])
    except Exception:  # noqa: BLE001
        logger.exception("数据库初始化失败，报告将无法持久化")
    yield
    SESSIONS.clear()
    STREAM_SESSIONS.clear()
    RESULT_STORE.clear()
    logger.info("QC-Agent 已停止")


app = FastAPI(
    title="QC-Agent API",
    description="工业质检多模态 Agent 系统 —— 后端服务（第二阶段：视觉工具已接入）",
    version="0.2.0",
    lifespan=lifespan,
)


# ------------------------------ 数据模型 ------------------------------


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户输入")
    session_id: str | None = Field(default=None, description="会话 ID；不传则新建")


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_input_tokens: int = 0


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    finished_reason: str | None = None
    usage: Usage = Usage()
    turn: int = 0


class InspectRequest(BaseModel):
    image_path: str = Field(..., description="待检测图像路径")
    gsd_mm_per_px: float = Field(default=0.05, gt=0, description="相机分辨率 mm/px")
    session_id: str | None = Field(
        default=None,
        description="可选。传入则会话中追加一条质检记录，便于追问",
    )


class DefectItem(BaseModel):
    """单个缺陷实例。

    ⚠️ area_px 必须是 float：OBB 用鞋带公式求旋转矩形面积，结果天然带小数
    （如 133261.39）。早期版本声明为 int，导致 Pydantic v2 抛
    `int_from_float` 校验错误使 /qc/inspect 返回 500。
    """

    model_config = ConfigDict(extra="ignore")   # 后端新增字段时不报错

    instance_id: str
    class_id: int | None = None
    class_name: str | None = None
    defect_type: str | None = None
    category: str | None = None
    bbox_px: list[int]
    polygon_px: list[list[float]] | None = None
    center_px: list[float] | None = None
    area_px: float = Field(..., description="像素面积（OBB 为多边形面积，带小数）")
    area_mm2: float
    max_length_px: float | None = None
    max_length_mm: float
    min_edge_mm: float | None = None
    aspect_ratio: float | None = None
    box_source: str | None = Field(default=None, description="obb | axis_aligned")
    severity: str = Field(..., description="物理阈值等级（需 GSD 可信）")
    severity_relative: str | None = Field(
        default=None, description="相对化等级（不依赖 GSD，推荐口径）"
    )
    area_ratio_sample: float | None = Field(
        default=None, description="缺陷面积 / 样品面积"
    )
    len_ratio_sample: float | None = Field(
        default=None, description="缺陷长边 / 样品边长"
    )
    confidence: float


class InspectResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    image_path: str
    image_size_px: list[int]
    gsd_mm_per_px: float
    defect_count: int
    total_area_mm2: float
    total_area_px: float | None = None
    total_area_ratio_of_image: float | None = None
    max_severity: str = Field(..., description="物理阈值等级（需 GSD 可信）")
    max_severity_relative: str | None = Field(
        default=None, description="相对化等级（不依赖 GSD）"
    )
    mean_confidence: float
    defects: list[DefectItem]
    backend: str
    model_task: str | None = None
    model_nc: int | None = None
    area_source: str | None = None
    sample_region: dict[str, Any] | None = Field(
        default=None, description="样品区域（相对化判定的分母来源）"
    )
    grading_mode: str | None = None
    physical_metrics_reliable: bool | None = Field(
        default=None, description="GSD 是否已标定；false 时 mm² 不可信"
    )
    session_id: str | None = None


# ---------------------- 结构化质检报告 Schema ----------------------


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

    说明：字段描述会作为 schema 的一部分传给模型，因此写得越明确，
    模型填得越准。这是"用类型系统约束模型输出"的实践。
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


class ReportRequest(BaseModel):
    image_path: str = Field(..., description="待检测图像路径")
    gsd_mm_per_px: float = Field(default=0.05, gt=0, description="相机分辨率 mm/px")
    batch_no: str = Field(default="", description="批次号（可选，用于归档检索）")
    session_id: str | None = Field(default=None, description="会话 ID（可选）")


class ReportResponse(BaseModel):
    report_id: str
    created_at: str
    batch_no: str
    image_path: str
    report: InspectionReport
    usage: Usage = Usage()
    review_id: str | None = Field(
        default=None, description="若报告需要人工复核，返回自动创建的复核任务 ID"
    )


# ---------------------- 人工复核 ----------------------


class ClaimRequest(BaseModel):
    reviewer: str = Field(..., min_length=1, description="复核员标识（工号或用户名）")


class DecideRequest(BaseModel):
    reviewer: str = Field(..., min_length=1, description="复核员标识")
    verdict: str = Field(..., description="判定结果：approved（采纳）或 rejected（驳回）")
    category: str = Field(
        default="", description="驳回原因分类，仅 rejected 时有意义：" + " / ".join(REJECT_CATEGORIES)
    )
    comment: str = Field(default="", description="复核意见")
    corrected_conclusion: str = Field(
        default="", description="人工修正后的结论（合格/不合格）。留空则沿用模型结论"
    )
    corrected_grade: str = Field(
        default="", description="人工修正后的等级（critical/major/minor/none）"
    )


class ReviewDecision(BaseModel):
    review_id: str
    report_id: str
    verdict: str
    category: str = ""
    category_label: str = ""
    final_conclusion: str
    final_grade: str
    decided_at: str
    overridden: bool = False
    message: str = ""


class SampleRequest(BaseModel):
    defect_kind: str = Field(default="crack", description="none | crack | hole | deform")
    filename: str | None = None


# ---------------------- LangGraph 编排路线 ----------------------


class GraphRunRequest(BaseModel):
    image_path: str = Field(..., description="待检测图像路径")
    gsd_mm_per_px: float = Field(default=0.05, gt=0, description="相机分辨率 mm/px")
    batch_no: str = Field(default="", description="批次号")
    thread_id: str | None = Field(
        default=None,
        description="LangGraph thread ID。留空则新建；传入已有 ID 可查看/恢复该任务",
    )


class GraphRunResponse(BaseModel):
    mode: str = Field(..., description="completed（已跑完）或 interrupted（等待人工）")
    thread_id: str
    stage: str = ""
    report_id: str | None = None
    report: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = Field(default_factory=list)
    interrupt: dict[str, Any] | None = Field(
        default=None, description="interrupted 时返回待人工处理的信息"
    )
    error: str = ""


class GraphResumeRequest(BaseModel):
    thread_id: str = Field(..., description="要恢复的 thread ID")
    verdict: str = Field(default="approved", description="approved（采纳）或 rejected（驳回）")
    reviewer: str = Field(default="", description="复核员标识")
    category: str = Field(default="", description="驳回原因分类")
    comment: str = Field(default="", description="复核意见")
    corrected_conclusion: str = Field(default="", description="人工修正后的结论")
    corrected_grade: str = Field(default="", description="人工修正后的等级")


# ------------------------------ 接口 ------------------------------


@app.get("/health", summary="健康检查")
async def health() -> dict[str, Any]:
    info: dict[str, Any] = {
        "status": "ok",
        "version": "0.5.0",
        "model": MODEL_NAME,
        "active_sessions": len(SESSIONS),
        "active_stream_sessions": len(STREAM_SESSIONS),
        "max_sessions": MAX_SESSIONS,
        "api_key_configured": bool(API_KEY),
        "tools": ["segment_defects", "get_image_info", "list_taxonomy",
                  "lookup_standard", "match_taxonomy_by_severity"],
        "env_file_exists": _ENV_FILE.exists(),
    }
    try:
        stats = review_stats()
        info["review_queue_open"] = stats["queue"]["total_open"]
        info["total_reports"] = stats["total_reports"]
        info["database"] = "ok"
    except Exception as exc:  # noqa: BLE001
        info["database"] = f"error: {exc}"

    # 判据录入情况：0/12 时所有报告都会写"无可用条款"，应尽快填写
    try:
        cs = criteria_status()
        info["criteria"] = {
            "filled": cs["filled"],
            "pending": cs["pending"],
            "total": cs["total"],
            "config_exists": cs["config_exists"],
            "config_path": cs["config_path"],
            "source": (cs.get("meta") or {}).get("source") or None,
        }
        if cs["filled"] == 0:
            info["criteria"]["hint"] = (
                "判据尚未录入，报告将标注'无可用条款'并转人工。"
                "请填写 configs/qc_criteria.yaml 后重启服务。"
            )
    except Exception as exc:  # noqa: BLE001
        info["criteria"] = f"error: {exc}"

    return info


@app.post("/chat", response_model=ChatResponse, summary="对话（Agent 可自主调用视觉工具）")
async def chat(req: ChatRequest) -> ChatResponse:
    session_id, record = get_session(req.session_id)

    user_msg = Msg(name="user", role="user", content=[TextBlock(type="text", text=req.message)])

    try:
        reply: Msg = await record.agent.reply(user_msg)
    except Exception as exc:  # noqa: BLE001
        logger.exception("会话 %s 调用失败", session_id)
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc

    usage_raw = getattr(reply, "usage", None)
    usage = Usage(
        input_tokens=getattr(usage_raw, "input_tokens", 0) or 0,
        output_tokens=getattr(usage_raw, "output_tokens", 0) or 0,
        cache_input_tokens=getattr(usage_raw, "cache_input_tokens", 0) or 0,
    )
    record.turns += 1
    record.total_input_tokens += usage.input_tokens
    record.total_output_tokens += usage.output_tokens

    reason = getattr(reply, "finished_reason", None)
    logger.info(
        "会话 %s 第 %d 轮完成：in=%d out=%d reason=%s",
        session_id, record.turns, usage.input_tokens, usage.output_tokens, reason,
    )

    return ChatResponse(
        session_id=session_id,
        reply=reply.get_text_content(),
        finished_reason=getattr(reason, "value", None) or (str(reason) if reason else None),
        usage=usage,
        turn=record.turns,
    )


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",  # 反向代理（Nginx）下必须禁用缓冲，否则 SSE 会被攒批
}

DEMO_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>QC-Agent 进度演示</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif;
         margin: 0; padding: 24px; background: #0f1419; color: #e6e6e6; }
  h1 { font-size: 18px; margin: 0 0 16px; font-weight: 600; }
  .row { display: flex; gap: 16px; flex-wrap: wrap; }
  .panel { background: #171d26; border: 1px solid #253040; border-radius: 10px;
           padding: 16px; flex: 1 1 420px; min-width: 320px; }
  .panel h2 { font-size: 13px; margin: 0 0 12px; color: #7d8b9e;
              text-transform: uppercase; letter-spacing: .08em; }
  input, button { font-size: 14px; padding: 9px 12px; border-radius: 7px;
                  border: 1px solid #2c3a4d; background: #0f1419; color: #e6e6e6; }
  input { width: 100%; margin-bottom: 10px; }
  button { background: #2563eb; border-color: #2563eb; color: #fff;
           cursor: pointer; font-weight: 500; }
  button:disabled { opacity: .5; cursor: not-allowed; }
  .phases { display: flex; flex-direction: column; gap: 7px; }
  .phase { display: flex; align-items: center; gap: 9px; padding: 7px 10px;
           border-radius: 7px; background: #0f1419; font-size: 13px;
           border-left: 3px solid #2c3a4d; transition: all .2s; }
  .phase.running { border-left-color: #f59e0b; background: #1f1a0f; }
  .phase.done { border-left-color: #10b981; color: #9ca3af; }
  .dot { width: 7px; height: 7px; border-radius: 50%; background: #2c3a4d; flex: none; }
  .phase.running .dot { background: #f59e0b; animation: pulse 1s infinite; }
  .phase.done .dot { background: #10b981; animation: none; }
  @keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: .25 } }
  .tools { font-size: 12px; color: #8b98a9; margin-top: 12px;
           border-top: 1px solid #253040; padding-top: 10px; }
  .tools div { padding: 3px 0; }
  .ok { color: #10b981; } .bad { color: #ef4444; }
  #text { white-space: pre-wrap; font-size: 14px; line-height: 1.65;
          min-height: 120px; }
  #meta { font-size: 12px; color: #7d8b9e; margin-top: 12px;
          border-top: 1px solid #253040; padding-top: 10px; }
  .caret::after { content: "▋"; color: #2563eb; animation: pulse .9s infinite; }
</style>
</head>
<body>
<h1>QC-Agent · 实时进度演示</h1>
<div class="row">
  <div class="panel">
    <h2>质检阶段</h2>
    <div class="phases" id="phases"></div>
    <div class="tools" id="tools"></div>
  </div>
  <div class="panel">
    <h2>Agent 输出</h2>
    <input id="imgpath" value="D:/agent_projects/PythonProject/data/samples/sample_crack.png">
    <button id="go">开始质检</button>
    <div id="text" style="margin-top:14px"></div>
    <div id="meta"></div>
  </div>
</div>
<script>
const $ = id => document.getElementById(id);
let phases = [], es = null, started = 0;

function renderPhases() {
  $('phases').innerHTML = phases.map((p, i) =>
    `<div class="phase ${p.status}" id="ph-${p.phase}">
       <span class="dot"></span><span>${i+1}. ${p.label}</span>
     </div>`).join('');
}
function setPhase(phase, status) {
  const el = $('ph-' + phase);
  if (el) el.className = 'phase ' + status;
}
$('go').onclick = () => {
  if (es) { es.close(); es = null; }
  const path = $('imgpath').value.trim();
  $('text').textContent = ''; $('tools').innerHTML = ''; $('meta').textContent = '';
  phases = []; $('phases').innerHTML = '';
  $('go').disabled = true; started = Date.now();

  const url = '/qc/report/stream?image_path=' + encodeURIComponent(path);
  es = new EventSource(url);

  es.addEventListener('run_start', e => {
    const d = JSON.parse(e.data);
    phases = d.phases.map(p => ({...p, status: ''}));
    renderPhases();
  });
  es.addEventListener('phase', e => {
    const d = JSON.parse(e.data);
    phases.forEach((p, i) => { if (i < d.index) p.status = 'done'; });
    renderPhases(); setPhase(d.phase, 'running');
  });
  es.addEventListener('tool_call', e => {
    const d = JSON.parse(e.data);
    $('tools').insertAdjacentHTML('beforeend',
      `<div>▶ 调用 <b>${d.tool}</b></div>`);
  });
  es.addEventListener('tool_result', e => {
    const d = JSON.parse(e.data);
    $('tools').insertAdjacentHTML('beforeend',
      `<div class="${d.ok ? 'ok' : 'bad'}">${d.ok ? '✓' : '✗'} ${d.tool} 返回</div>`);
  });
  es.addEventListener('text', e => {
    const d = JSON.parse(e.data);
    const t = $('text');
    t.textContent += d.delta;
    t.classList.add('caret');
  });
  es.addEventListener('usage', e => {
    const d = JSON.parse(e.data);
    $('meta').textContent = `tokens  输入 ${d.input_tokens} / 输出 ${d.output_tokens} / 缓存命中 ${d.cache_input_tokens}`;
  });
  es.addEventListener('report', e => {
    const d = JSON.parse(e.data);
    const r = d.report;
    $('tools').insertAdjacentHTML('beforeend',
      `<div style="margin-top:10px;padding-top:10px;border-top:1px solid #253040">
         <b>结论：${r.conclusion}</b>（等级 ${r.grade}，缺陷 ${r.defect_count} 处）
         ${r.requires_review ? '<span style="color:#f59e0b"> · 需人工复核</span>' : ''}
       </div>`);
  });
  es.addEventListener('done', e => {
    const d = JSON.parse(e.data);
    phases.forEach(p => p.status = 'done'); renderPhases();
    $('text').classList.remove('caret');
    const secs = ((Date.now() - started) / 1000).toFixed(1);
    $('meta').textContent += `   耗时 ${secs}s   工具调用 ${d.tool_calls.length} 次`;
    $('go').disabled = false; es.close(); es = null;
  });
  es.addEventListener('error', e => {
    if (e.data) { $('tools').insertAdjacentHTML('beforeend',
      `<div class="bad">✗ ${JSON.parse(e.data).message}</div>`); }
    $('go').disabled = false;
  });
  es.addEventListener('stream_closed', () => { es && es.close(); es = null; $('go').disabled = false; });
};
</script>
</body>
</html>"""


@app.get("/chat/stream", summary="对话（SSE 流式，实时推送推理进度）")
async def chat_stream(message: str, session_id: str | None = None) -> StreamingResponse:
    """SSE 流式对话。

    用 query 参数而非 request body：浏览器原生 EventSource 不支持自定义请求体，
    这是为了让前端能用最简的 EventSource 直接连。生产环境若需鉴权，
    可改用 fetch + ReadableStream 读取 POST 响应流。
    """
    sid, record = get_stream_session(session_id)
    translator = EventTranslator(run_id=uuid.uuid4().hex[:12], session_id=sid, message=message)

    user_msg = Msg(name="user", role="user", content=[TextBlock(type="text", text=message)])
    events = record.agent.reply_stream(user_msg)

    return StreamingResponse(
        translate_stream(events, translator),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.get("/qc/report/stream", summary="生成结构化报告（SSE 流式）")
async def qc_report_stream(
    image_path: str,
    gsd_mm_per_px: float = 0.05,
    batch_no: str = "",
    session_id: str | None = None,
) -> StreamingResponse:
    """流式版 /qc/report：实时推送各阶段进度，最后一条 report 事件带结构化报告。

    前端据此实现"逐段产出、边出边确认"的体验，而不是提交后干等 40 秒。
    """
    sid, record = get_stream_session(session_id)

    prompt = (
        f"请对图像 {image_path} 执行完整质检并生成报告。\n"
        f"相机分辨率 {gsd_mm_per_px} mm/px。"
        + (f"\n批次号：{batch_no}。" if batch_no else "")
        + "\n要求：先调用 segment_defects 获取实测数据，"
        "再调用 lookup_standard 查询判据条款，最后据实填写报告各字段。"
    )
    translator = EventTranslator(run_id=uuid.uuid4().hex[:12], session_id=sid, message=prompt)

    user_msg = Msg(name="user", role="user", content=[TextBlock(type="text", text=prompt)])
    events = record.agent.reply_stream(user_msg, structured_schema=InspectionReport)

    def on_final(evt: Any) -> str | None:
        """ReplyEndEvent 时把结构化报告作为最后一个业务事件下发。"""
        msg = getattr(evt, "reply_msg", None) or getattr(evt, "msg", None)
        raw = getattr(msg, "structured_output", None) if msg is not None else None
        if raw is None:
            raw = getattr(
                getattr(getattr(record.agent, "state", None), "reply_context", None),
                "structured_output",
                None,
            )
        if raw is None:
            return sse_pack("error", {"message": "未能取得结构化报告（模型可能未完成报告生成）"})
        try:
            report = InspectionReport.model_validate(raw)
        except Exception as exc:  # noqa: BLE001
            return sse_pack("error", {"message": f"报告字段校验失败: {exc}"})
        return sse_pack("report", {"report": report.model_dump(), "batch_no": batch_no})

    return StreamingResponse(
        translate_stream(events, translator, on_final=on_final),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@app.get("/demo", response_class=HTMLResponse, summary="内置演示页（验证 SSE 进度推送）")
async def demo() -> HTMLResponse:
    """一个自包含的演示页，无需前端工程即可看到阶段进度与打字机效果。"""
    return HTMLResponse(DEMO_HTML)


@app.post("/qc/inspect", response_model=InspectResponse, summary="确定性质检（不经过 LLM）")
async def qc_inspect(req: InspectRequest) -> InspectResponse:
    """直接调用分割后端并返回结构化结果。

    与 /chat 的区别：本接口不做自然语言解释，结果完全可复现，
    适合前端渲染与自动化评测。
    """
    try:
        result = annotate_defects(req.image_path, req.gsd_mm_per_px)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("质检失败: %s", req.image_path)
        raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

    RESULT_STORE[result["image_path"]] = result
    logger.info(
        "质检完成 %s：%d 处缺陷，最高等级 %s",
        req.image_path, result["defect_count"], result["max_severity"],
    )

    # 可选：把结构化结论注入会话上下文，用户随后可追问"这个缺陷怎么处理"
    if req.session_id:
        _, record = get_session(req.session_id)
        summary = (
            f"[系统] 已完成图像质检：{result['image_path']}，"
            f"缺陷 {result['defect_count']} 处，总面积 {result['total_area_mm2']} mm²，"
            f"最高等级 {result['max_severity']}。缺陷明细：{result['defects']}"
        )
        await record.agent.observe(
            Msg(name="system", role="system", content=[TextBlock(type="text", text=summary)])
        )

    return InspectResponse(**result, session_id=req.session_id)


@app.post("/qc/report", response_model=ReportResponse, summary="生成结构化质检报告（LLM + 判据库）")
async def qc_report(req: ReportRequest) -> ReportResponse:
    """走完整 Agent 链路生成结构化报告。

    与 /qc/inspect 的分工：
      * /qc/inspect  —— 纯工具调用，产出测量数据，完全可复现（给系统用）
      * /qc/report   —— Agent 读数据 + 查判据 + 出结论，产出报告（给人用）

    报告通过 structured_schema 约束字段；模型填错类型时框架会把校验错误
    回传给它自我修正（内置机制，无需自己实现重试）。
    """
    session_id, record = get_session(req.session_id)

    prompt = (
        f"请对图像 {req.image_path} 执行完整质检并生成报告。\n"
        f"相机分辨率 {req.gsd_mm_per_px} mm/px。"
        + (f"\n批次号：{req.batch_no}。" if req.batch_no else "")
        + "\n要求：先调用 segment_defects 获取实测数据，"
        "再调用 lookup_standard 查询判据条款，最后据实填写报告各字段。"
    )

    try:
        reply: Msg = await record.agent.reply(
            Msg(name="user", role="user", content=[TextBlock(type="text", text=prompt)]),
            structured_schema=InspectionReport,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("会话 %s 生成报告失败", session_id)
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc

    # 结构化结果可能在 Msg 上，也可能只在 agent state 里，两处都查
    raw = getattr(reply, "structured_output", None)
    if raw is None:
        raw = getattr(
            getattr(getattr(record.agent, "state", None), "reply_context", None),
            "structured_output",
            None,
        )

    if raw is None:
        logger.error("未能取得结构化输出，finished_reason=%s", getattr(reply, "finished_reason", None))
        raise HTTPException(
            status_code=502,
            detail=(
                "未能取得结构化输出。可能原因：模型不支持结构化输出、"
                "或工具调用未完成。回复文本："
                + reply.get_text_content()
            ),
        )

    try:
        report = InspectionReport.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"报告字段校验失败: {exc}") from exc

    usage_raw = getattr(reply, "usage", None)
    usage = Usage(
        input_tokens=getattr(usage_raw, "input_tokens", 0) or 0,
        output_tokens=getattr(usage_raw, "output_tokens", 0) or 0,
        cache_input_tokens=getattr(usage_raw, "cache_input_tokens", 0) or 0,
    )
    record.turns += 1
    record.total_input_tokens += usage.input_tokens
    record.total_output_tokens += usage.output_tokens

    report_id = f"R{uuid.uuid4().hex[:12].upper()}"
    created_at = datetime.now().isoformat(timespec="seconds")

    # 持久化：写入报告表；若 requires_review 则自动创建复核任务
    review_id = None
    try:
        saved = save_report(
            report_id=report_id,
            image_path=req.image_path,
            report=report.model_dump(),
            usage=usage.model_dump(),
            batch_no=req.batch_no,
            session_id=session_id,
            gsd_mm_per_px=req.gsd_mm_per_px,
        )
        review_id = saved.get("review_id")
        logger.info(
            "报告已入库 %s（复核任务 %s）", report_id, review_id or "无需复核"
        )
    except Exception:  # noqa: BLE001 —— 入库失败不应导致质检结果丢失
        logger.exception("报告入库失败 %s", report_id)

    logger.info(
        "报告 %s 生成完成：结论=%s 等级=%s 缺陷数=%d 需复核=%s",
        report_id, report.conclusion, report.grade, report.defect_count, report.requires_review,
    )

    return ReportResponse(
        report_id=report_id,
        created_at=created_at,
        batch_no=req.batch_no,
        image_path=req.image_path,
        report=report,
        usage=usage,
        review_id=review_id,
    )


@app.get("/qc/reports", summary="列出已归档的质检报告")
async def qc_reports(limit: int = 20, batch_no: str | None = None) -> dict[str, Any]:
    try:
        items = list_reports(limit=limit, batch_no=batch_no)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取报告失败: {exc}") from exc
    return {"total": len(items), "batch_no": batch_no, "items": items}


@app.get("/qc/reports/{report_id}", summary="查看单份报告（含复核记录）")
async def qc_report_detail(report_id: str) -> dict[str, Any]:
    try:
        rec = get_report(report_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取失败: {exc}") from exc
    if rec is None:
        raise HTTPException(status_code=404, detail=f"报告不存在: {report_id}")
    return rec


@app.get("/qc/review/queue", summary="复核队列（待处理）")
async def qc_review_queue(status: str | None = None, limit: int = 50) -> dict[str, Any]:
    try:
        items = list_reviews(status=status, limit=limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取队列失败: {exc}") from exc
    return {"total": len(items), "items": items, "reject_categories": REJECT_CATEGORIES}


@app.post("/qc/review/{review_id}/claim", summary="认领复核任务")
async def qc_review_claim(review_id: str, req: ClaimRequest) -> dict[str, Any]:
    """认领后状态变为 in_review，避免多人重复处理同一单。"""
    try:
        rec = claim_review(review_id, req.reviewer)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return rec


@app.post("/qc/review/{review_id}/decide", response_model=ReviewDecision, summary="提交复核判定")
async def qc_review_decide(review_id: str, req: DecideRequest) -> ReviewDecision:
    """提交复核结论。

    ★ 闭环的关键：人工结论会写入报告作为**权威记录**（final_conclusion），
    覆盖模型结论。驳回原因分类则进入统计，作为后续优化提示词与工具的输入。
    """
    try:
        out = decide_review(
            review_id=review_id,
            reviewer=req.reviewer,
            verdict=req.verdict,
            category=req.category,
            comment=req.comment,
            corrected_conclusion=req.corrected_conclusion,
            corrected_grade=req.corrected_grade,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    logger.info(
        "复核 %s 判定完成：%s（报告 %s，原因 %s）",
        review_id, req.verdict, out["report_id"], req.category or "无",
    )

    msg = "已采纳模型结论" if req.verdict == "approved" else "已驳回，人工结论将覆盖模型结论"
    if out["overridden"]:
        msg += "（结论已被修正）"

    return ReviewDecision(
        **out,
        category_label=REJECT_CATEGORIES.get(req.category, req.category),
        message=msg,
    )


@app.get("/qc/review/stats", summary="复核统计（队列积压 / 通过率 / 驳回原因分布）")
async def qc_review_stats() -> dict[str, Any]:
    """驳回原因分布是优化方向的依据：哪类错误最多，下轮就往哪里投入。"""
    try:
        return review_stats()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"统计失败: {exc}") from exc


# ---------------------- LangGraph 路线端点 ----------------------


@app.post("/graph/qc", response_model=GraphRunResponse, summary="[LangGraph] 执行质检流程")
async def graph_qc(req: GraphRunRequest) -> GraphRunResponse:
    """用 LangGraph 的五阶段图执行质检。

    与 `/qc/report` 的区别：
      * `/qc/report`   —— AgentScope 单 Agent 一气跑完，实现简单
      * `/graph/qc`    —— LangGraph 显式编排，每阶段状态落盘；
                          需要复核时在 gate 节点**真正挂起**（返回 interrupted），
                          可用 `/graph/qc/resume` 继续，实现断点续检

    两条路线共用同一套工具、判据库、报告 Schema 与落库逻辑，
    因此评测指标可以直接对比。
    """
    try:
        from qcagent.graph.service import get_graph
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"LangGraph 路线不可用（未安装依赖）: {exc}。"
                   "请执行 pip install langgraph langgraph-checkpoint-sqlite",
        ) from exc

    thread_id = req.thread_id or f"qc-{uuid.uuid4().hex[:12]}"
    cfg = {"configurable": {"thread_id": thread_id}}
    graph = get_graph()

    try:
        result = await asyncio.to_thread(graph.invoke, {
            "image_path": req.image_path,
            "gsd_mm_per_px": req.gsd_mm_per_px,
            "batch_no": req.batch_no,
        }, cfg)
    except Exception as exc:  # noqa: BLE001
        logger.exception("LangGraph 执行失败 thread=%s", thread_id)
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc

    inter = result.get("__interrupt__")
    trace = [
        {"stage": t.get("stage"), "status": t.get("status"), "detail": t.get("detail", "")}
        for t in (result.get("trace") or [])
    ]

    if inter:
        payload = getattr(inter[0], "value", None) or (inter[0] if isinstance(inter[0], dict) else {})
        logger.info("thread=%s 在 gate 挂起，等待人工", thread_id)
        return GraphRunResponse(
            mode="interrupted",
            thread_id=thread_id,
            stage=result.get("stage", ""),
            report=result.get("report"),
            trace=trace,
            interrupt=dict(payload or {}),
            error=result.get("error", ""),
        )

    logger.info("thread=%s 执行完成，报告 %s", thread_id, result.get("report_id"))
    return GraphRunResponse(
        mode="completed",
        thread_id=thread_id,
        stage=result.get("stage", ""),
        report_id=result.get("report_id"),
        report=result.get("report"),
        trace=trace,
        error=result.get("error", ""),
    )


@app.post("/graph/qc/resume", response_model=GraphRunResponse, summary="[LangGraph] 提交人工判定并恢复流程")
async def graph_qc_resume(req: GraphResumeRequest) -> GraphRunResponse:
    """从 gate 节点的中断处恢复。

    这是 LangGraph 相比"事后复核"的核心价值：**状态在挂起期间被冻结保存**，
    恢复时从 gate 继续执行，validate/perceive/lookup/judge 不会重跑
    —— 既不重复消耗模型费用，也保证了流程的一致性。
    """
    try:
        from langgraph.types import Command

        from qcagent.graph.service import get_graph
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=f"LangGraph 路线不可用: {exc}") from exc

    if req.verdict not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="verdict 必须是 approved 或 rejected")
    if req.verdict == "rejected" and req.category and req.category not in REJECT_CATEGORIES:
        raise HTTPException(
            status_code=400,
            detail=f"未知驳回原因分类: {req.category}（可选 {list(REJECT_CATEGORIES)}）",
        )

    cfg = {"configurable": {"thread_id": req.thread_id}}
    graph = get_graph()

    # 先确认该 thread 确实处于挂起状态，避免对已完成的 thread 重复 resume
    try:
        snapshot = graph.get_state(cfg)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取状态失败: {exc}") from exc

    if not snapshot.next:
        raise HTTPException(
            status_code=409,
            detail=f"thread {req.thread_id} 当前没有待执行的挂起点（可能已完成或不存在）",
        )

    try:
        result = await asyncio.to_thread(
            graph.invoke,
            Command(resume={
                "verdict": req.verdict,
                "reviewer": req.reviewer,
                "category": req.category,
                "comment": req.comment,
                "corrected_conclusion": req.corrected_conclusion,
                "corrected_grade": req.corrected_grade,
            }),
            cfg,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("LangGraph 恢复失败 thread=%s", req.thread_id)
        raise HTTPException(status_code=502, detail=f"{type(exc).__name__}: {exc}") from exc

    trace = [
        {"stage": t.get("stage"), "status": t.get("status"), "detail": t.get("detail", "")}
        for t in (result.get("trace") or [])
    ]
    logger.info(
        "thread=%s 人工判定 %s 后恢复完成，报告 %s",
        req.thread_id, req.verdict, result.get("report_id"),
    )
    return GraphRunResponse(
        mode="completed",
        thread_id=req.thread_id,
        stage=result.get("stage", ""),
        report_id=result.get("report_id"),
        report=result.get("report"),
        trace=trace,
        error=result.get("error", ""),
    )


@app.get("/graph/state/{thread_id}", summary="[LangGraph] 查询某个 thread 的状态")
async def graph_state(thread_id: str) -> dict[str, Any]:
    """查看任务当前处于哪个阶段、是否已挂起。用于前端"恢复任务"入口。"""
    try:
        from qcagent.graph.service import get_graph
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=f"LangGraph 路线不可用: {exc}") from exc

    graph = get_graph()
    cfg = {"configurable": {"thread_id": thread_id}}
    try:
        snap = graph.get_state(cfg)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"读取状态失败: {exc}") from exc

    values = dict(snap.values or {})
    return {
        "thread_id": thread_id,
        "next": list(snap.next or ()),
        "interrupted": bool(snap.next),
        "stage": values.get("stage", ""),
        "report_id": values.get("report_id"),
        "error": values.get("error", ""),
        "trace": values.get("trace", []),
        "has_measurement": values.get("measurement") is not None,
        "has_report": values.get("report") is not None,
    }


@app.get("/graph/info", summary="[LangGraph] 编排层与检查点库信息")
async def graph_info() -> dict[str, Any]:
    """运维观测：检查点库位置、行数、thread 数，以及图的节点拓扑。"""
    try:
        from qcagent.graph.service import checkpoint_info, get_graph
        from langgraph.checkpoint.memory import InMemorySaver  # noqa: F401
    except ImportError as exc:
        raise HTTPException(status_code=501, detail=f"LangGraph 路线不可用: {exc}") from exc

    info: dict[str, Any] = {"available": True, "checkpoint": checkpoint_info()}
    try:
        g = get_graph().get_graph()
        info["nodes"] = sorted(g.nodes.keys())
        info["edges"] = [
            {"from": e.source, "to": e.target, "conditional": bool(e.conditional)}
            for e in g.edges
        ]
    except Exception as exc:  # noqa: BLE001
        info["topology_error"] = f"{type(exc).__name__}: {exc}"
    return info


@app.post("/qc/sample", summary="生成样例图像（无真实数据时用于联调）")
async def qc_sample(req: SampleRequest) -> dict[str, Any]:
    name = req.filename or f"sample_{req.defect_kind}.png"
    out = settings.SAMPLES_DIR / name
    try:
        path = make_sample_image(str(out), defect_kind=req.defect_kind)
    except Exception as exc:  # noqa: BLE001
        # 把真实异常暴露出来：否则前端只能看到无信息量的 500
        logger.exception("生成样例图失败: %s", out)
        raise HTTPException(
            status_code=500,
            detail=(
                f"{type(exc).__name__}: {exc} | "
                f"目标路径={out} | 目标目录可写={os.access(out.parent, os.W_OK)}"
            ),
        ) from exc
    return {"image_path": path}


@app.get("/qc/samples", summary="列出已存在的样例图像路径")
async def qc_samples() -> dict[str, Any]:
    """不生成新图，只列出目录里已有的样例，便于直接拿去调 /qc/inspect。"""
    sample_dir = settings.SAMPLES_DIR
    files = sorted(str(p.resolve()) for p in sample_dir.glob("*.png")) if sample_dir.exists() else []
    return {"sample_dir": str(sample_dir), "exists": sample_dir.exists(), "files": files}


@app.delete("/session/{session_id}", summary="结束会话")
async def close_session(session_id: str) -> dict[str, Any]:
    record = SESSIONS.pop(session_id, None)
    if record is None:
        raise HTTPException(status_code=404, detail=f"会话不存在: {session_id}")
    return {
        "closed": session_id,
        "turns": record.turns,
        "total_input_tokens": record.total_input_tokens,
        "total_output_tokens": record.total_output_tokens,
    }
