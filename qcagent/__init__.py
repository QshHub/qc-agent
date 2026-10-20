# -*- coding: utf-8 -*-
"""工业质检多模态 Agent 系统

分层结构：
    qcagent.core   检测后端、判据库、工具、事件映射、持久化
    qcagent.graph  LangGraph 五阶段编排（可选路线）
    qcagent.api    FastAPI 服务与演示页
    qcagent.eval   分层评测与端到端验收
"""

from __future__ import annotations

__version__ = "0.5.0"
