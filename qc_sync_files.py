# -*- coding: utf-8 -*-
"""把最新版代码同步到你的项目目录

## 为什么需要这个脚本

开发过程中在 workspace 里迭代了多个文件（qc_yolo / qc_tools / qc_knowledge
/ qc_api / qc_gsd_calib 等），而项目目录里的版本可能较旧，导致：

  · 新训练的 OBB 12 类模型接不进去（qc_yolo 缺相对化判定）
  · qc_tools 缺 QC_BACKEND 切换逻辑
  · qc_gsd_calib 完全缺失 → 样品区域无法标定

## 行为

  1. 逐文件比对内容（md5），仅复制有差异的
  2. 覆盖前把旧文件备份到 `_backup_<时间戳>/`
  3. 不删除目标目录里独有的文件（如 probe.py / main.py）
  4. 复制后做一次导入自检

## 用法

    python qc_sync_files.py --check      # 只看差异，不复制
    python qc_sync_files.py              # 执行同步
    python qc_sync_files.py --src 路径   # 指定源目录
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

# 源目录（最新版代码所在）
DEFAULT_SRC = Path(
    r"C:\Users\11966\Documents\deepseek-harness\default-workspace\qc-agent"
)
# 目标目录（你的项目）
DEFAULT_DST = Path(r"D:\agent_projects\PythonProject")

# 需要同步的文件清单（按依赖顺序，被依赖的在前）
FILES = [
    "qc_severity.py",       # 相对化判定（无外部依赖）
    "qc_knowledge.py",      # 判据库（14 类）
    "qc_tools.py",          # 工具层（依赖 qc_yolo / qc_severity）
    "qc_yolo.py",           # YOLO 后端（依赖 qc_severity）
    "qc_events.py",         # SSE 事件映射
    "qc_store.py",          # 持久化
    "qc_gsd_calib.py",      # 样品区域标定
    # ---- LangGraph 路线（可选；缺失时 /graph/* 端点返回 501）----
    "qc_agent_shared.py",
    "qc_judge.py",
    "qc_graph.py",
    "qc_graph_service.py",
    # ---- 服务与评测 ----
    "qc_api.py",            # FastAPI 服务（依赖以上全部）
    "qc_eval.py",           # Agent 层评测
    "qc_eval_cases.py",     # 评测用例
    "qc_detect_eval.py",    # 检测指标评测
    "qc_train.py",          # 训练脚本
    "qc_prepare_dataset.py",  # 数据集精简
    "qc_check_gpu.py",      # 环境诊断
    "实测指标记录.md",        # 指标记录
    ".env.example",         # 配置模板
]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description="同步最新版代码到项目目录")
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--dst", default=str(DEFAULT_DST))
    ap.add_argument("--check", action="store_true", help="只看差异，不复制")
    ap.add_argument("--no-backup", action="store_true", help="不备份（不推荐）")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)

    print("=" * 78)
    print("同步代码")
    print("=" * 78)
    print(f"  源目录  {src}")
    print(f"  目标    {dst}")
    if not src.exists():
        print(f"\n❌ 源目录不存在: {src}")
        sys.exit(1)
    if not dst.exists():
        print(f"\n❌ 目标目录不存在: {dst}")
        sys.exit(1)

    # ---- 比对 ----
    rows: list[tuple[str, str, str]] = []   # (文件名, 状态, 说明)
    to_copy: list[str] = []
    for name in FILES:
        s = src / name
        d = dst / name
        if not s.exists():
            rows.append((name, "SKIP", "源目录也没有"))
            continue
        if not d.exists():
            rows.append((name, "NEW", "目标缺失"))
            to_copy.append(name)
            continue
        if md5(s) == md5(d):
            rows.append((name, "SAME", ""))
            continue
        rows.append((name, "DIFF", f"{d.stat().st_size//1024}KB → {s.stat().st_size//1024}KB"))
        to_copy.append(name)

    print(f"\n{'文件':<28}{'状态':<8}{'说明'}")
    print("-" * 78)
    for name, st, note in rows:
        mark = {"SAME": "  一致", "DIFF": "⚠️ 有差异", "NEW": "＋ 缺失", "SKIP": "— 跳过"}[st]
        print(f"  {name:<26}{mark:<10}{note}")

    print(f"\n共 {len(rows)} 个文件，需同步 {len(to_copy)} 个")

    if not to_copy:
        print("\n所有文件已是最新，无需同步。")
        return

    if args.check:
        print("\n--check 模式，未复制。去掉 --check 执行同步。")
        return

    # ---- 备份 ----
    backup = None
    if not args.no_backup:
        backup = dst / f"_backup_{datetime.now().strftime('%m%d_%H%M%S')}"
        backup.mkdir(parents=True, exist_ok=True)
        n = 0
        for name in to_copy:
            d = dst / name
            if d.exists():
                shutil.copy2(d, backup / name)
                n += 1
        print(f"\n已备份 {n} 个旧文件 -> {backup}")

    # ---- 复制 ----
    print()
    for name in to_copy:
        try:
            shutil.copy2(src / name, dst / name)
            print(f"  ✓ {name}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ {name}: {type(exc).__name__}: {exc}")

    # ---- 导入自检 ----
    print("\n" + "=" * 78)
    print("导入自检（用当前 Python）")
    print("=" * 78)
    code = (
        "import sys; sys.path.insert(0, r'%s');"
        "mods = ['qc_severity','qc_knowledge','qc_tools','qc_yolo',"
        "'qc_events','qc_store','qc_gsd_calib'];"
        "ok, bad = [], [];"
        "\nfor m in mods:\n"
        "    try:\n"
        "        __import__(m); ok.append(m)\n"
        "    except Exception as e:\n"
        "        bad.append(f'{m}: {type(e).__name__}: {e}')\n"
        "print('  正常:', ' '.join(ok) if ok else '无')\n"
        "if bad:\n"
        "    print('  失败:')\n"
        "    for b in bad: print('   ', b)\n"
        "sys.exit(1 if bad else 0)"
    ) % str(dst)
    rc = subprocess.run([sys.executable, "-c", code], check=False).returncode

    print("\n" + "=" * 78)
    if rc == 0:
        print("✅ 同步完成，所有模块可导入。")
        print("\n下一步：")
        print("  1) 更新 .env（见 .env.example）")
        print("  2) python qc_gsd_calib.py --populate-cache --split val")
        print("  3) python -m uvicorn qc_api:app --port 8000")
    else:
        print("⚠️ 部分模块导入失败，请检查上面的错误信息。")
        if backup:
            print(f"   需要回滚可从 {backup} 恢复")
    print("=" * 78)


if __name__ == "__main__":
    main()
