# -*- coding: utf-8 -*-
"""项目路径常量（唯一真源）

所有模块都从这里取路径，避免各自用 `Path(__file__).parent` 推算——
文件一旦移动，各自推算的层数不同，极易指错目录（本次重构就踩过）。
"""

from __future__ import annotations

from pathlib import Path

# .../项目根/qcagent/settings.py  →  上两级即项目根
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

# 目录
DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = PROJECT_ROOT / "configs"
SAMPLES_DIR = DATA_DIR / "samples"
REPORTS_DIR = DATA_DIR / "reports"

# 文件
ENV_FILE = PROJECT_ROOT / ".env"
DB_PATH = DATA_DIR / "qc_agent.db"
SAMPLE_REGIONS_PATH = DATA_DIR / "sample_regions.json"
CRITERIA_PATH = CONFIG_DIR / "qc_criteria.yaml"


def ensure_dirs() -> None:
    """确保运行期需要的目录存在。"""
    for d in (DATA_DIR, SAMPLES_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
