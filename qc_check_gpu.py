# -*- coding: utf-8 -*-
"""GPU / PyTorch 环境诊断

在 PyCharm 的 qcagent 环境里运行：
    python qc_check_gpu.py

它会逐项检查并给出明确的修复命令，而不是只打印一堆版本号。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys


def hr(title: str = "") -> None:
    print("\n" + "=" * 70)
    if title:
        print(title)
        print("=" * 70)


def check_nvidia_smi() -> dict:
    """检查驱动与显卡。"""
    hr("① NVIDIA 驱动")
    exe = shutil.which("nvidia-smi")
    if not exe:
        print("  ❌ 未找到 nvidia-smi（驱动未安装或不在 PATH）")
        return {"ok": False}
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        print("  " + out.stdout.strip().replace("\n", "\n  "))
        return {"ok": out.returncode == 0}
    except Exception as exc:  # noqa: BLE001
        print(f"  ❌ nvidia-smi 调用失败: {exc}")
        return {"ok": False}


def check_torch() -> dict:
    """检查 PyTorch 与 CUDA 绑定情况 —— 诊断的核心。"""
    hr("② PyTorch 与 CUDA")
    try:
        import torch
    except ImportError:
        print("  ❌ 未安装 torch")
        return {"torch": None}

    print(f"  torch 版本        {torch.__version__}")
    print(f"  torch.version.cuda {getattr(torch.version, 'cuda', None)}")
    print(f"  cuda 是否可用      {torch.cuda.is_available()}")
    print(f"  GPU 数量          {torch.cuda.device_count()}")

    # 关键判据：版本号里带 +cpu 就是 CPU 专用轮子
    is_cpu_wheel = "+cpu" in torch.__version__
    has_cuda_bind = getattr(torch.version, "cuda", None) is not None

    if is_cpu_wheel:
        print("  ❌ 当前是 CPU-only 版 torch（版本号含 +cpu）")
        print("     → 这就是训练不用 GPU 的直接原因")
    elif not has_cuda_bind:
        print("  ❌ torch 未绑定 CUDA（torch.version.cuda 为 None）")
    elif not torch.cuda.is_available():
        print("  ⚠️ torch 有 CUDA 绑定但检测不到可用 GPU")
        print("     → 多为驱动版本低于该 CUDA 版本，或显卡被禁用")
    else:
        print("  ✅ torch 可使用 CUDA")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"     GPU{i}: {props.name}  "
                  f"{props.total_memory / 1024**3:.1f} GB  "
                  f"算力 sm_{props.major}{props.minor}")

    # 真正做一次 GPU 运算，排除"报告可用但实际失败"
    if torch.cuda.is_available():
        try:
            x = torch.randn(1000, 1000, device="cuda")
            y = x @ x
            torch.cuda.synchronize()
            print(f"  ✅ 实际 GPU 运算成功（结果范数 {y.norm().item():.2f}）")
        except Exception as exc:  # noqa: BLE001
            print(f"  ❌ GPU 运算失败: {type(exc).__name__}: {exc}")

    return {
        "torch": torch.__version__,
        "cuda_bind": getattr(torch.version, "cuda", None),
        "available": torch.cuda.is_available(),
        "is_cpu_wheel": is_cpu_wheel,
    }


def check_ultralytics() -> dict:
    hr("③ ultralytics")
    try:
        import ultralytics
    except ImportError:
        print("  ❌ 未安装 ultralytics")
        return {"installed": False}
    print(f"  版本 {ultralytics.__version__}")
    try:
        import torch

        print(f"  同环境 torch {torch.__version__}")
    except ImportError:
        pass
    return {"installed": True, "version": ultralytics.__version__}


def check_env_vars() -> None:
    hr("④ 相关环境变量")
    keys = ["CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_PATH", "TORCH_CUDA_ARCH_LIST"]
    found = False
    for k in keys:
        v = os.environ.get(k)
        if v:
            print(f"  {k} = {v}")
            found = True
    if not found:
        print("  以上变量均未设置（通常正常）")
    if os.environ.get("CUDA_VISIBLE_DEVICES") == "":
        print("  ⚠️ CUDA_VISIBLE_DEVICES 被设为空字符串，这会让 torch 看不到任何 GPU！")


def main() -> None:
    print("GPU / PyTorch 环境诊断")
    print(f"Python {sys.version.split()[0]}   {sys.executable}")

    drv = check_nvidia_smi()
    torch_info = check_torch()
    check_ultralytics()
    check_env_vars()

    hr("结论与修复")
    if not drv.get("ok"):
        print("  驱动层面有问题，先装/更新 NVIDIA 驱动。")
        return

    if torch_info.get("torch") is None:
        print("  未装 torch。建议直接装 CUDA 版：")
        print("    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        return

    if torch_info.get("is_cpu_wheel") or not torch_info.get("cuda_bind"):
        print("  ❗ 根因：PyTorch 是 CPU 版。最可能是在装 ultralytics 时被 pip 替换了。")
        print()
        print("  修复步骤（在 qcagent 环境里依次执行）：")
        print("    1) 卸载 CPU 版 torch")
        print("       pip uninstall -y torch torchvision torchaudio")
        print("    2) 装 CUDA 12.4 版 torch（RTX 3070 / 驱动 591 完全支持）")
        print("       pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
        print("    3) 验证")
        print("       python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"")
        print("       期望输出包含 +cu124 且为 True")
        print("    4) 若 ultralytics 因 torch 版本变化报错，重装它（不要带 --force-reinstall torch）")
        print("       pip install -U ultralytics")
        print()
        print("  ⚠️ 注意：不要再执行 pip install ultralytics 之外会动 torch 的命令。")
        print("     装完后立刻用 python -c 验证 torch.cuda.is_available() 仍为 True。")
    elif not torch_info.get("available"):
        print("  torch 有 CUDA 但检测不到 GPU：检查是否被别的进程占满显存")
        print("    nvidia-smi")
        print("  或更新驱动到支持该 CUDA 版本。")
    else:
        print("  ✅ GPU 环境正常。若训练仍未用 GPU，检查训练参数：")
        print("    python qc_train.py --device 0 ...")
        print("  或代码里 device 参数被设成了 'cpu' / 空字符串。")

    hr()
    print("把以上完整输出发我，我按实际情况给下一步。")


if __name__ == "__main__":
    main()
