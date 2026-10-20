# -*- coding: utf-8 -*-
"""事件映射层：把 AgentScope 的 AgentEvent 翻译成前端可消费的 SSE 事件（第四阶段）

为什么要单独一层（面试可讲）：
  * **前后端解耦**：前端不该知道 AgentScope 的内部事件类型。今天用 AgentScope，
    明天换别的框架，只要这一层保持 SSE 协议不变，前端零改动。
  * **阶段语义化**：AgentScope 的事件是"机制级"的（工具调用、文本增量），
    而业务要的是"阶段级"的（缺陷识别中、判据检索中）。这一层负责翻译。
  * **可测试**：纯函数映射，输入事件列表 → 输出 SSE 序列，可以离线单测，
    不需要真跑模型。

SSE 事件协议（前端只需实现这几种）：
    run_start    任务开始        {run_id, session_id, message}
    phase        阶段切换        {phase, label, status}
    tool_call    工具开始调用    {tool, args}
    tool_result  工具返回        {tool, ok, summary}
    thinking     思维链增量      {delta}         （可折叠展示）
    text         正文增量        {delta}         （打字机效果）
    usage        token 用量      {input_tokens, output_tokens, cache_input_tokens}
    report       结构化报告      {report}        （仅 /qc/report/stream）
    error        错误            {message}
    done         任务结束        {finished_reason, elapsed_s}
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator

# --------------------------------------------------------------------------
# 阶段定义：前端进度条按此顺序渲染
# --------------------------------------------------------------------------

PHASES: list[dict[str, str]] = [
    {"phase": "intake", "label": "图像校验"},
    {"phase": "perceive", "label": "缺陷识别"},
    {"phase": "knowledge", "label": "判据检索"},
    {"phase": "reasoning", "label": "结论推理"},
    {"phase": "report", "label": "报告生成"},
]

# 工具 → 所属阶段。新接工具后在这里登记即可，前端无需改动。
TOOL_PHASE_MAP: dict[str, str] = {
    "get_image_info": "intake",
    "segment_defects": "perceive",
    "lookup_standard": "knowledge",
    "list_taxonomy": "knowledge",
    "match_taxonomy_by_severity": "knowledge",
}


def sse_pack(event: str, data: dict[str, Any]) -> str:
    """打包成 SSE 帧。

    格式遵循 SSE 规范：event 行 + data 行 + 空行结尾。
    data 用 JSON 保证换行与特殊字符不会破坏协议。
    """
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def _short(value: Any, limit: int = 300) -> str:
    s = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    return s if len(s) <= limit else s[:limit] + f"...(共 {len(s)} 字符)"


def _extract_args(block: Any) -> Any:
    """工具调用参数在流式过程中是逐步拼接的 JSON 字符串。"""
    raw = getattr(block, "input", None)
    if raw is None:
        return {}
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except Exception:  # noqa: BLE001 —— 参数可能还没拼完
        return _short(raw)


def _extract_result_text(block: Any) -> str:
    out = getattr(block, "output", None)
    if out is None:
        return ""
    if isinstance(out, list):
        parts = []
        for b in out:
            t = getattr(b, "text", None)
            if t:
                parts.append(t)
        return "".join(parts)
    return str(out)


class EventTranslator:
    """有状态的事件翻译器：需要记住当前阶段、累计用量等上下文。"""

    def __init__(self, run_id: str, session_id: str, message: str) -> None:
        self.run_id = run_id
        self.session_id = session_id
        self.message = message
        self.t0 = time.monotonic()
        self.current_phase: str | None = None
        self.usage: dict[str, int] = {}
        self.tool_names: list[str] = []
        self.text_chars = 0

    # ---------------- 内部 ----------------

    def _phase_for_tool(self, tool: str) -> str | None:
        return TOOL_PHASE_MAP.get(tool)

    def _enter_phase(self, phase: str) -> str | None:
        if phase == self.current_phase:
            return None
        self.current_phase = phase
        label = next((p["label"] for p in PHASES if p["phase"] == phase), phase)
        return sse_pack("phase", {
            "phase": phase,
            "label": label,
            "status": "running",
            "index": next((i for i, p in enumerate(PHASES) if p["phase"] == phase), -1),
            "total": len(PHASES),
        })

    # ---------------- 对外 ----------------

    def start(self) -> str:
        return sse_pack("run_start", {
            "run_id": self.run_id,
            "session_id": self.session_id,
            "message": self.message,
            "phases": PHASES,
        })

    def translate(self, evt: Any) -> list[str]:
        """把一个 AgentEvent 翻译成 0~N 个 SSE 帧。"""
        name = type(evt).__name__
        frames: list[str] = []

        # ---- 工具调用开始：进入对应阶段 ----
        if name == "ToolCallStartEvent":
            block = getattr(evt, "tool_call", None) or evt
            tool = getattr(block, "name", "unknown")
            self.tool_names.append(tool)
            phase = self._phase_for_tool(tool)
            if phase:
                f = self._enter_phase(phase)
                if f:
                    frames.append(f)
            frames.append(sse_pack("tool_call", {
                "tool": tool,
                "args": _extract_args(block),
                "call_id": getattr(block, "id", None),
            }))

        # ---- 工具调用结束 ----
        elif name == "ToolCallEndEvent":
            block = getattr(evt, "tool_call", None) or evt
            frames.append(sse_pack("tool_call_end", {
                "tool": getattr(block, "name", "unknown"),
                "state": str(getattr(block, "state", "")),
            }))

        # ---- 工具结果 ----
        elif name == "ToolResultEndEvent":
            block = getattr(evt, "tool_result", None) or evt
            text = _extract_result_text(block)
            ok = "error" not in str(getattr(block, "state", "")).lower()
            frames.append(sse_pack("tool_result", {
                "tool": getattr(block, "name", "unknown"),
                "ok": ok,
                "summary": _short(text, 500),
            }))

        # ---- 正文增量 ----
        elif name == "TextBlockDeltaEvent":
            delta = getattr(evt, "delta", "") or ""
            if delta:
                self.text_chars += len(delta)
                # 正文开始产出，说明已进入推理作答阶段
                f = self._enter_phase("reasoning")
                if f:
                    frames.append(f)
                frames.append(sse_pack("text", {"delta": delta}))

        # ---- 思维链增量 ----
        elif name == "ThinkingBlockDeltaEvent":
            delta = getattr(evt, "delta", "") or ""
            if delta:
                frames.append(sse_pack("thinking", {"delta": delta}))

        # ---- 模型调用结束：拿用量 ----
        elif name == "ModelCallEndEvent":
            u = getattr(evt, "usage", None)
            if u is not None:
                self.usage = {
                    "input_tokens": int(getattr(u, "input_tokens", 0) or 0),
                    "output_tokens": int(getattr(u, "output_tokens", 0) or 0),
                    "cache_input_tokens": int(getattr(u, "cache_input_tokens", 0) or 0),
                }
                frames.append(sse_pack("usage", self.usage))

        # ---- 异常终止信号 ----
        elif name == "ExceedMaxItersEvent":
            frames.append(sse_pack("error", {
                "message": "达到最大推理轮次，任务被强制终止（可能陷入循环或任务过于复杂）",
            }))

        # ---- 需要人工确认 ----
        elif name == "RequireUserConfirmEvent":
            frames.append(sse_pack("require_confirm", {
                "message": "工具调用需要人工确认",
                "detail": _short(getattr(evt, "tool_calls", None), 500),
            }))

        return frames

    def done(self, finished_reason: Any, extra: dict[str, Any] | None = None) -> str:
        reason = getattr(finished_reason, "value", None) or (str(finished_reason) if finished_reason else None)
        data: dict[str, Any] = {
            "run_id": self.run_id,
            "finished_reason": reason,
            "elapsed_s": round(time.monotonic() - self.t0, 2),
            "tool_calls": self.tool_names,
            "text_chars": self.text_chars,
            "usage": self.usage,
        }
        if extra:
            data.update(extra)
        return sse_pack("done", data)

    def error(self, exc: BaseException) -> str:
        return sse_pack("error", {
            "message": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.monotonic() - self.t0, 2),
        })


async def translate_stream(
    events: AsyncGenerator[Any, None],
    translator: EventTranslator,
    on_final: Any = None,
) -> AsyncGenerator[str, None]:
    """把 AgentScope 的事件流翻译成 SSE 帧流。

    on_final 可选回调：收到 ReplyEndEvent 时调用，用于附加结构化报告等终态数据。
    """
    yield translator.start()
    try:
        async for evt in events:
            for frame in translator.translate(evt):
                yield frame

            if type(evt).__name__ == "ReplyEndEvent" and on_final is not None:
                extra_frame = on_final(evt)
                if extra_frame:
                    yield extra_frame

            if type(evt).__name__ == "ReplyEndEvent":
                yield translator.done(getattr(evt, "finished_reason", None))
    except Exception as exc:  # noqa: BLE001 —— 流已开始，错误必须以事件形式下发
        yield translator.error(exc)
    finally:
        yield sse_pack("stream_closed", {"run_id": translator.run_id})
