# -*- coding: utf-8 -*-
"""用清华镜像安装 CUDA 版 PyTorch（conda 频道方案）

## 为什么用 conda 频道而不是 pip

实测结论（已验证）：

  · PyPI / 清华 PyPI 镜像：**只有 CPU 版 torch**（版本号不带 +cuXXX）
  · download.pytorch.org 的 pip 索引：你环境下目录页解析失败
  · **清华 anaconda/cloud/pytorch 频道：有完整 CUDA 版包** ← 本脚本用这个

匹配到你的环境（Python 3.12）的确切包：
    pytorch-2.5.1-py3.12_cuda12.1_cudnn9_0.tar.bz2
    torchvision-0.20.1-py312_cu121.tar.bz2

## 关键：装完后不要用普通 pip 重装 ultralytics

普通 `pip install ultralytics` 会重新解析依赖，把 conda 装的 CUDA 版 torch
替换成 PyPI 的 CPU 版。必须用 `--no-deps`。

## 用法

    python qc_install_cuda.py --check     # 先看当前状态
    python qc_install_cuda.py --plan      # 打印将执行的命令
    python qc_install_cuda.py             # 执行
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import subprocess
import sys

# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

CONDA_ENV = "qcagent"

TUNA_CHANNEL = "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch"
# conda-forge 走清华镜像（pytorch 频道自身不含 numpy/filelock 等依赖）
CONDA_FORGE = "https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge"

# 精确到构建串，避免 conda 解析到不匹配的变体
PKG_TORCH = "pytorch=2.5.1=py3.12_cuda12.1_cudnn9_0"
PKG_VISION = "torchvision=0.20.1=py312_cu121"
PKG_CUDA = "pytorch-cuda=12.1"

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


def _needs_quote(token: str) -> bool:
    return any(ch in set(" <>&|;()$`'\"") for ch in token)


def run(cmd: list[str], *, dry: bool, check: bool = True) -> int:
    printable = " ".join(f'"{c}"' if _needs_quote(c) else c for c in cmd)
    print(f"\n  $ {printable}")
    if dry:
        return 0
    rc = subprocess.run(cmd, check=False).returncode
    if check and rc != 0:
        print(f"  ⚠️ 返回码 {rc}")
    return rc


def module_version(name: str) -> str | None:
    try:
        return getattr(importlib.import_module(name), "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return None


def inspect_state() -> dict:
    print("=" * 76)
    print("当前状态")
    print("=" * 76)
    print(f"  Python  {sys.version.split()[0]}")
    print(f"  可执行  {sys.executable}")

    conda = shutil.which("conda")
    print(f"  conda   {conda or '未找到（PATH 里没有 conda）'}")

    st: dict = {"conda": conda}
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

    print(f"  torchvision         {module_version('torchvision') or '未安装'}")
    print(f"  ultralytics         {module_version('ultralytics') or '未安装'}")
    return st


def find_env_python() -> str:
    """定位目标 conda 环境的 python。优先用当前解释器。"""
    return sys.executable


def install(args: argparse.Namespace, st: dict) -> None:
    dry = args.plan
    env_py = find_env_python()

    print("\n" + "=" * 76)
    print("安装计划（清华 conda 频道）")
    print("=" * 76)
    print(f"  目标环境      {CONDA_ENV}")
    print(f"  python        {env_py}")
    print(f"  torch         {PKG_TORCH}")
    print(f"  torchvision   {PKG_VISION}")
    print(f"  cuda 约束     {PKG_CUDA}")
    print(f"  频道          {TUNA_CHANNEL}")
    print(f"  依赖频道      {CONDA_FORGE}")

    conda = st.get("conda")
    if not conda:
        print("\n❌ 找不到 conda 命令。请在 Anaconda Prompt 中运行，或把 conda 加入 PATH。")
        print("   替代方案：手动执行下面两条命令（在 Anaconda Prompt 里）")
        print(f"   conda install -y -c {TUNA_CHANNEL} -c {CONDA_FORGE} "
              f"{PKG_TORCH} {PKG_VISION} {PKG_CUDA}")
        return

    # ---- 步骤 1：conda 安装 pytorch + torchvision + cuda 约束 ----
    print("\n【步骤 1】安装 CUDA 版 pytorch / torchvision（清华频道）")
    print("  注意：这一步会下载约 2.5GB，请保持网络稳定")

    # 先试 pytorch 专属频道
    rc = run(
        [conda, "install", "-y",
         "--override-channels",
         "-c", TUNA_CHANNEL,
         "-c", CONDA_FORGE,
         PKG_TORCH, PKG_VISION, PKG_CUDA],
        dry=dry, check=False,
    )

    if rc != 0 and not dry:
        print("\n  ⚠️ 首次尝试失败，放宽构建串约束后重试（让 conda 自行解析）")
        rc = run(
            [conda, "install", "-y",
             "--override-channels",
             "-c", TUNA_CHANNEL,
             "-c", CONDA_FORGE,
             "pytorch=2.5.1", "torchvision=0.20.1", PKG_CUDA],
            dry=dry, check=False,
        )

    if rc != 0 and not dry:
        print("\n  ⚠️ 仍失败。尝试不锁版本（让 conda 取最新兼容组合）")
        rc = run(
            [conda, "install", "-y",
             "--override-channels",
             "-c", TUNA_CHANNEL,
             "-c", CONDA_FORGE,
             "pytorch", "torchvision", PKG_CUDA],
            dry=dry, check=False,
        )

    if dry:
        print("\n  （dry-run，跳过后续步骤）")
        return

    # ---- 步骤 2：立即验证 ----
    print("\n" + "=" * 76)
    print("步骤 2：验证 CUDA 是否可用（关键）")
    print("=" * 76)
    verify = (
        "import torch;"
        "print('torch', torch.__version__);"
        "print('cuda_bind', torch.version.cuda);"
        "print('available', torch.cuda.is_available());"
        "print('GPU', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A');"
        "import torchvision; print('torchvision', torchvision.__version__);"
        "import sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    )
    rc2 = run([env_py, "-c", verify], dry=False, check=False)
    if rc2 != 0:
        print("\n❌ CUDA 仍不可用，已停止（不继续装 ultralytics）")
        print("   请把上面的输出发我排查：常见原因是 conda 解析到了 CPU 变体，")
        print("   或 pytorch-cuda 与驱动版本不匹配。")
        return

    # ---- 步骤 3：--no-deps 装 ultralytics（避免覆盖 torch）----
    print("\n【步骤 3】用 --no-deps 安装 ultralytics（不触碰刚装好的 torch）")
    run([env_py, "-m", "pip", "install", "--no-deps", "--upgrade", "ultralytics"],
        dry=False, check=False)

    # ---- 步骤 4：补齐 ultralytics 其它依赖 ----
    print("\n【步骤 4】补齐 ultralytics 的其它依赖（不含 torch/torchvision）")
    for i in range(0, len(ULTRALYTICS_DEPS), 5):
        run([env_py, "-m", "pip", "install", *ULTRALYTICS_DEPS[i:i + 5]],
            dry=False, check=False)

    # ---- 步骤 5：最终验证 ----
    print("\n" + "=" * 76)
    print("步骤 5：最终验证（确认 ultralytics 没把 torch 换掉）")
    print("=" * 76)
    rc3 = run([env_py, "-c", verify], dry=False, check=False)
    print("\n" + "=" * 76)
    if rc3 == 0:
        print("✅ 全部就绪。下一步：")
        print("   python qc_train.py --epochs 3 --imgsz 640 --batch 8 --device 0")
    else:
        print("❌ ultralytics 安装后 CUDA 不可用 —— torch 又被覆盖了")
        print("   修复：conda install --force-reinstall pytorch torchvision，")
        print("   然后 pip install --no-deps ultralytics（务必带 --no-deps）")
    print("=" * 76)


def main() -> None:
    global CONDA_ENV

    ap = argparse.ArgumentParser(description="用清华镜像安装 CUDA 版 PyTorch")
    ap.add_argument("--check", action="store_true", help="只检查状态")
    ap.add_argument("--plan", action="store_true", help="只打印命令（dry-run）")
    ap.add_argument("--env", default=CONDA_ENV, help="conda 环境名（仅提示用）")
    args = ap.parse_args()

    CONDA_ENV = args.env

    st = inspect_state()

    if args.check:
        healthy = bool(st.get("cuda_avail") and not st.get("is_cpu"))
        print("\n" + "=" * 76)
        print("结论:", "✅ CUDA 可用" if healthy else "❌ 需要安装 CUDA 版 torch")
        print("=" * 76)
        return

    install(args, st)


if __name__ == "__main__":
    main()
