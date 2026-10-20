# -*- coding: utf-8 -*-
"""为 Python 3.14 安装 CUDA 版 PyTorch（官方源 cu128）

## 为什么必须用 cu128

实测 wheel 可用性（cp314 = Python 3.14）：

    cu121: cp314 无     ← 之前一直让装这个，所以永远找不到
    cu124: cp314 无
    cu126: cp314 无
    cu128: cp314 有  → torch 2.9.0 / 2.9.1 / 2.10.0 / 2.11.0
                       torchvision 0.24.0 / 0.24.1 / 0.25.0 / 0.26.0

国内镜像（阿里/清华）均未同步 cp314 的 CUDA wheel，故必须走官方源
download.pytorch.org。该源实测可访问，直链 HEAD 返回 200。

## 版本配套（必须成对，否则 import 崩）

    torch 2.10.0  ↔  torchvision 0.25.0     ← 本脚本默认
    torch 2.9.0   ↔  torchvision 0.24.0
    torch 2.9.1   ↔  torchvision 0.24.1
    torch 2.11.0  ↔  torchvision 0.26.0

## 用法

    python qc_install_torch314.py --check
    python qc_install_torch314.py --plan
    python qc_install_torch314.py                 # 默认：index-url 方式
    python qc_install_torch314.py --direct        # 直链下载后本地安装（最稳）

Windows 上建议在 Anaconda Prompt 里跑，或确认已激活 qcagent 环境。
"""

from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
import sysconfig

# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

CUDA_TAG = "cu128"
INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_TAG}"

# torch ↔ torchvision 配套表
PAIRS: dict[str, str] = {
    "2.9.0": "0.24.0",
    "2.9.1": "0.24.1",
    "2.10.0": "0.25.0",
    "2.11.0": "0.26.0",
}
DEFAULT_TORCH = "2.10.0"

# 直链（已验证 HEAD 200：2752MB / 9MB）
DIRECT: dict[str, str] = {
    "torch": f"{INDEX_URL}/torch-{DEFAULT_TORCH}%2B{CUDA_TAG}-cp314-cp314-win_amd64.whl",
    "torchvision": f"{INDEX_URL}/torchvision-{PAIRS[DEFAULT_TORCH]}%2B{CUDA_TAG}-cp314-cp314-win_amd64.whl",
}

ULTRALYTICS_DEPS = [
    "matplotlib>=3.3.0",
    "opencv-python>=4.7.0",
    "pyyaml>=5.3.1",
    "requests>=2.23.0",
    "psutil>=5.8.0",
    "polars>=0.20.0",
    "nvidia-ml-py>=12.0.0",
    "ultralytics-thop>=2.2.2",
    "ultralytics-platform>=0.1.45",
]


def _q(t: str) -> bool:
    return any(c in set(" <>&|;()$`'\"") for c in t)


def run(cmd: list[str], *, dry: bool, check: bool = True) -> int:
    print("\n  $ " + " ".join(f'"{c}"' if _q(c) else c for c in cmd))
    if dry:
        return 0
    rc = subprocess.run(cmd, check=False).returncode
    if check and rc != 0:
        print(f"  ⚠️ 返回码 {rc}")
    return rc


def ver(mod: str) -> str | None:
    try:
        return getattr(importlib.import_module(mod), "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return None


def py_tag() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def inspect() -> dict:
    print("=" * 78)
    print("当前环境")
    print("=" * 78)
    print(f"  Python      {sys.version.split()[0]}  ({py_tag()})")
    print(f"  可执行      {sys.executable}")
    print(f"  环境前缀    {sys.prefix}")

    st: dict = {}
    try:
        import torch

        st["torch"] = torch.__version__
        st["cuda_bind"] = getattr(torch.version, "cuda", None)
        st["cuda_avail"] = torch.cuda.is_available()
        st["is_cpu"] = "+cpu" in torch.__version__
        print(f"\n  torch               {torch.__version__}")
        print(f"  torch.version.cuda  {st['cuda_bind']}")
        print(f"  cuda.is_available   {st['cuda_avail']}")
        if st["cuda_avail"]:
            print(f"  GPU                 {torch.cuda.get_device_name(0)}")
    except ImportError:
        st.update(torch=None, is_cpu=True, cuda_avail=False)
        print("\n  torch               未安装")

    print(f"  torchvision         {ver('torchvision') or '未安装'}")
    print(f"  ultralytics         {ver('ultralytics') or '未安装'}")

    if sys.version_info[:2] != (3, 14):
        print(f"\n  ⚠️ 本脚本针对 Python 3.14 编写；当前是 {sys.version_info.major}.{sys.version_info.minor}")
        print(f"     若你是 3.12/3.13，可用 cu121/cu124 版本，见 qc_install_cuda.py")
    return st


def install(args: argparse.Namespace, st: dict) -> None:
    dry = args.plan
    torch_v = args.torch or DEFAULT_TORCH
    tv_v = PAIRS.get(torch_v)
    if tv_v is None:
        print(f"\n❌ 不支持的 torch 版本 {torch_v}，可选: {list(PAIRS)}")
        sys.exit(1)

    print("\n" + "=" * 78)
    print("安装计划（官方源 cu128）")
    print("=" * 78)
    print(f"  torch        {torch_v}+{CUDA_TAG}")
    print(f"  torchvision  {tv_v}+{CUDA_TAG}")
    print(f"  索引         {INDEX_URL}")
    print(f"  方式         {'直链下载后本地安装' if args.direct else 'pip index-url'}")
    print(f"  下载量       约 2.7 GB（torch）+ 9 MB（torchvision）")

    step1_needed = st.get("is_cpu") or not st.get("cuda_avail") or ver("torchvision") is None

    if not step1_needed:
        print("\n【步骤 1】跳过：torch 已是 CUDA 版且可用")

    elif args.direct and torch_v != DEFAULT_TORCH:
        # 直链只内置了默认版本的 URL，其他版本退回 pip 方式
        print(f"\n  ⚠️ --direct 仅内置 {DEFAULT_TORCH} 的直链；"
              f"改用 pip 方式安装 {torch_v}")
        args.direct = False

    # ============ 分支 A：直链下载后本地安装 ============
    if step1_needed and args.direct:
        print("\n【步骤 1】直链下载 wheel 后本地安装")
        from pathlib import Path

        cache = None
        for cand in (Path(r"E:\yolodata\wheels"),
                     Path.home() / ".cache" / "qc_wheels",
                     Path.cwd() / "wheels"):
            try:
                cand.mkdir(parents=True, exist_ok=True)
                probe = cand / ".write_test"
                probe.write_text("ok", encoding="utf-8")
                probe.unlink()
                cache = cand
                break
            except Exception:  # noqa: BLE001
                continue

        if cache is None:
            print("  ❌ 无法创建下载目录，改用 pip 方式")
            args.direct = False
        else:
            print(f"  下载目录: {cache}")
            files: list[str] = []
            for pkg, url in DIRECT.items():
                fn = url.rsplit("/", 1)[-1].replace("%2B", "+")
                dest = cache / fn
                if dest.exists() and dest.stat().st_size > 1024 * 1024:
                    print(f"  {pkg}: 已存在，跳过 -> {dest}")
                    files.append(str(dest))
                    continue

                print(f"\n  下载 {pkg}")
                print(f"    {fn}")
                rc = run(["curl", "-L", "--fail", "--retry", "2", "-C", "-",
                          "-o", str(dest), url], dry=dry, check=False)
                if dry:
                    continue

                if rc != 0 or not dest.exists() or dest.stat().st_size < 1024 * 1024:
                    print("    curl 未成功，改用 Python 下载（支持断点续传）")
                    try:
                        import urllib.request

                        have = dest.stat().st_size if dest.exists() else 0
                        headers = {"User-Agent": "pip/24.0"}
                        if have:
                            headers["Range"] = f"bytes={have}-"
                        req = urllib.request.Request(url, headers=headers)
                        mode = "ab" if have else "wb"
                        with urllib.request.urlopen(req, timeout=300) as r, open(dest, mode) as f:
                            while chunk := r.read(1024 * 1024):
                                f.write(chunk)
                        print(f"    完成 {dest.stat().st_size/1024/1024:.0f} MB")
                    except Exception as exc:  # noqa: BLE001
                        print(f"    ❌ 下载失败: {type(exc).__name__}: {exc}")
                        continue
                files.append(str(dest))

            if files and not dry:
                print("\n  本地安装（--no-deps，防止 pip 从 PyPI 补 CPU 版）")
                run([sys.executable, "-m", "pip", "install", "--no-deps",
                     "--force-reinstall", *files], dry=False, check=False)
            elif not dry:
                print("\n  ❌ wheel 未就绪，尝试改用 pip 方式")
                args.direct = False

    # ============ 分支 B：pip index-url 安装 ============
    if step1_needed and not args.direct:
        print("\n【步骤 1】pip 通过 index-url 安装")
        print("  官方索引根页是跳转页，pip 会请求 <index>/torch/ 子目录；")
        print("  实测该子目录正常（460 个 wheel），应可解析。")
        run([sys.executable, "-m", "pip", "install",
             f"torch=={torch_v}+{CUDA_TAG}",
             f"torchvision=={tv_v}+{CUDA_TAG}",
             "--index-url", INDEX_URL],
            dry=dry, check=False)

    # ---- 步骤 2：验证 ----
    if not dry and step1_needed:
        print("\n" + "=" * 78)
        print("步骤 2：验证 CUDA（关键，不通过就停）")
        print("=" * 78)
        code = (
            "import torch;"
            "print('torch      ', torch.__version__);"
            "print('cuda_bind  ', torch.version.cuda);"
            "print('available  ', torch.cuda.is_available());"
            "print('GPU        ', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A');"
            "import torchvision; print('torchvision', torchvision.__version__);"
            "print(torch.cuda.get_device_properties(0)) if torch.cuda.is_available() else None;"
            "import sys; sys.exit(0 if torch.cuda.is_available() else 1)"
        )
        rc = run([sys.executable, "-c", code], dry=False, check=False)
        if rc != 0:
            print("\n❌ CUDA 不可用，已中止。")
            print("  常见原因：")
            print("   1) 驱动版本低于 CUDA 12.8 要求（你的驱动 591.86 应该够，但需确认）")
            print("   2) 下载的不是 cu128 版本（检查文件名含 +cu128）")
            print("   3) 显存被其他进程占满")
            print("  先跑：nvidia-smi  然后：python -c \"import torch; print(torch.cuda.is_available())\"")
            return

    # ---- 步骤 3：补 torch 依赖 ----
    if not dry and step1_needed:
        print("\n【步骤 3】补齐 torch 的运行时依赖（--no-deps 装的会缺这些）")
        run([sys.executable, "-m", "pip", "install",
             "filelock", "typing-extensions", "sympy", "networkx",
             "jinja2", "fsspec"], dry=False, check=False)

    # ---- 步骤 4：--no-deps 装 ultralytics ----
    print("\n【步骤 4】ultralytics（--no-deps，防止它把 torch 换成 CPU 版）")
    run([sys.executable, "-m", "pip", "install", "--no-deps", "--upgrade", "ultralytics"],
        dry=dry, check=False)

    print("\n【步骤 5】补齐 ultralytics 的其它依赖（不含 torch）")
    for i in range(0, len(ULTRALYTICS_DEPS), 5):
        run([sys.executable, "-m", "pip", "install", *ULTRALYTICS_DEPS[i:i + 5]],
            dry=dry, check=False)

    # ---- 步骤 6：最终验证 ----
    if not dry:
        print("\n" + "=" * 78)
        print("步骤 6：最终验证（确认 ultralytics 没把 torch 换掉）")
        print("=" * 78)
        rc = run([sys.executable, "-c",
                  "import torch, torchvision, ultralytics;"
                  "print('torch', torch.__version__);"
                  "print('cuda_ok', torch.cuda.is_available());"
                  "print('ultralytics', ultralytics.__version__);"
                  "import sys; sys.exit(0 if torch.cuda.is_available() else 1)"],
                 dry=False, check=False)
        print("\n" + "=" * 78)
        if rc == 0:
            print("✅ 环境就绪。下一步：")
            print("   python qc_check_gpu.py")
            print("   python qc_train.py --epochs 3 --imgsz 640 --batch 8 --device 0")
        else:
            print("❌ 最终验证失败：CUDA 不可用。")
            print("   把上面的完整输出发我。")
        print("=" * 78)


def main() -> None:
    ap = argparse.ArgumentParser(description="Python 3.14 安装 CUDA 版 PyTorch（cu128 官方源）")
    ap.add_argument("--check", action="store_true", help="只检查状态")
    ap.add_argument("--plan", action="store_true", help="只打印命令")
    ap.add_argument("--direct", action="store_true", help="直链下载后本地安装（最稳）")
    ap.add_argument("--torch", default=DEFAULT_TORCH,
                    help=f"torch 版本，可选 {list(PAIRS)}，默认 {DEFAULT_TORCH}")
    args = ap.parse_args()

    st = inspect()

    if args.check:
        ok = bool(st.get("cuda_avail") and not st.get("is_cpu"))
        print("\n" + "=" * 78)
        print("结论:", "✅ CUDA 可用" if ok else "❌ 需要安装 CUDA 版 torch")
        print("=" * 78)
        return

    install(args, st)


if __name__ == "__main__":
    main()
