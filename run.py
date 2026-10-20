# -*- coding: utf-8 -*-
"""服务启动入口

    python run.py                # 默认 0.0.0.0:8000
    python run.py --port 9000    # 指定端口
    python run.py --reload       # 开发模式热重载

等价于：
    uvicorn qcagent.api.app:app --port 8000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保项目根在 sys.path 上（从任意目录执行脚本都能找到 qcagent 包）
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="启动质检 Agent 服务")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = ap.parse_args()

    import uvicorn

    if args.reload:
        uvicorn.run("qcagent.api.app:app", host=args.host, port=args.port, reload=True)
    else:
        uvicorn.run("qcagent.api.app:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
