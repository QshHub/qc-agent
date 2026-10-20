# -*- coding: utf-8 -*-
"""YOLO11-OBB 训练脚本（蜂窝催化剂缺陷检测）

## 为什么需要"数据体检"再训练

你的数据集有几个会直接影响训练成败的特征，先体检能在训练前发现问题，
而不是等 100 轮跑完发现 mAP 是 0：

  * **6 个小样本类别**（损坏 2 个框、表皮缺口 5 个、孔洞 9 个…）——
    这些类别几乎学不出来，预告指标会很难看，避免误以为是代码问题
  * **5472×3648 的超高分辨率**——默认 imgsz=640 会把细长裂缝缩到几乎不可见，
    这是最容易犯的错
  * **51+13 张空标签图**——它们是合法样本（无缺陷），但比例过高会让模型倾向"什么都不检出"
  * **多尺度/多色彩模式**（RGB 326 / 灰度 120）——影响归一化与增强策略

## 关键参数说明

| 参数 | 建议值 | 理由 |
|---|---|---|
| `imgsz` | **1024**（不是 640） | 裂缝在 640 下会缩成十几个像素；1024 是显存与精度的折中 |
| `epochs` | 100（小数据集可 150） | 446 张图，100 轮通常够收敛 |
| `batch` | 4~8 | 1024 尺寸下，8GB 显存建议 4 |
| `cache` | False | 图太大（每张约 60MB），缓存会撑爆内存 |
| `mosaic` | 0.5（默认 1.0 偏高） | 缺陷是局部结构，过度拼接会破坏上下文 |
| `degrees` | 保留默认 | OBB 标注本身已含旋转，增强旋转可能引入标签噪声 |

## 用法

    python qc_train.py --dry-run          # 只体检，不训练
    python qc_train.py                    # 体检 + 训练
    python qc_train.py --imgsz 1280 --batch 4 --epochs 150
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

DATA_ROOT = Path(r"E:\yolodata")
# 基础权重：必须是 **-obb 后缀** 的 OBB 模型。
# 实测 yolo11m-obb.pt = OBBModel（含 head.OBB 与 angle 分支），DOTA 15 类预训练。
# 注意：yolo26l.pt 是普通 detect 模型（COCO 80 类），不能用于 OBB 训练。
BASE_WEIGHTS = DATA_ROOT / "yolo11m-obb.pt"
RUNS_DIR = DATA_ROOT / "runs"


# --------------------------------------------------------------------------
# 数据体检
# --------------------------------------------------------------------------


def inspect_dataset(root: Path) -> dict[str, Any]:
    """检查数据完整性、类别分布、标注合法性。"""
    report: dict[str, Any] = {"root": str(root), "issues": [], "warnings": []}

    splits = {}
    for split in ("train", "val"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in (".bmp", ".jpg", ".png"))
        labels = sorted(lbl_dir.glob("*.txt"))

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}

        missing_lbl = img_stems - lbl_stems
        orphan_lbl = lbl_stems - img_stems

        cls_counter: Counter[int] = Counter()
        empty = 0
        bad_lines = 0
        out_of_range = 0
        total_boxes = 0
        boxes_per_img: list[int] = []

        for lbl in labels:
            text = lbl.read_text(encoding="utf-8", errors="replace").strip()
            if not text:
                empty += 1
                boxes_per_img.append(0)
                continue
            cnt = 0
            for line in text.splitlines():
                parts = line.split()
                if len(parts) < 9:
                    bad_lines += 1
                    continue
                try:
                    cid = int(float(parts[0]))
                    coords = [float(v) for v in parts[1:9]]
                except ValueError:
                    bad_lines += 1
                    continue
                if any(c < -0.01 or c > 1.01 for c in coords):
                    out_of_range += 1
                cls_counter[cid] += 1
                cnt += 1
            total_boxes += cnt
            boxes_per_img.append(cnt)

        splits[split] = {
            "images": len(images),
            "labels": len(labels),
            "empty_labels": empty,
            "total_boxes": total_boxes,
            "class_counts": dict(sorted(cls_counter.items())),
            "avg_boxes_per_image": round(total_boxes / len(labels), 2) if labels else 0,
            "boxes_per_image_hist": {
                str(k): v for k, v in sorted(Counter(boxes_per_img).items())[:10]
            },
        }

        if missing_lbl:
            report["issues"].append(f"{split}: {len(missing_lbl)} 张图缺标签文件")
        if orphan_lbl:
            report["warnings"].append(f"{split}: {len(orphan_lbl)} 个标签文件缺图")
        if bad_lines:
            report["issues"].append(f"{split}: {bad_lines} 行格式非法")
        if out_of_range:
            report["warnings"].append(f"{split}: {out_of_range} 个坐标超出 [0,1]")

    report["splits"] = splits

    # 类别名与 data.yaml 对照
    yaml_path = root / "data.yaml"
    report["data_yaml"] = str(yaml_path)
    if not yaml_path.exists():
        report["issues"].append("data.yaml 不存在")
        report["names"] = {}
    else:
        names, nc = parse_yaml_names(yaml_path)
        report["names"] = names
        report["nc"] = nc
        # 交叉检查：标签里的类别 id 是否都在 names 范围内
        all_ids: set[int] = set()
        for s in splits.values():
            all_ids |= set(s["class_counts"].keys())
        bad_ids = {i for i in all_ids if i not in names}
        if bad_ids:
            report["issues"].append(f"标签中出现未定义的类别 id: {sorted(bad_ids)}")

    # 小样本类别预警
    if "train" in splits:
        counts = splits["train"]["class_counts"]
        tiny = {cid: n for cid, n in counts.items() if n < 15}
        if tiny:
            names = report.get("names", {})
            msg = "；".join(f"{names.get(c, c)}={n}框" for c, n in sorted(tiny.items()))
            report["warnings"].append(
                f"训练集小样本类别（<15 框）：{msg}。这些类别召回率会很低，"
                f"属数据量问题，不应归因于代码"
            )

    return report


def parse_yaml_names(path: Path) -> tuple[dict[int, str], int]:
    """极简 data.yaml 解析（只取 names 与 nc），避免依赖 pyyaml。"""
    names: dict[int, str] = {}
    nc = 0
    in_names = False
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        if line.startswith("nc:"):
            try:
                nc = int(line.split(":", 1)[1].strip())
            except ValueError:
                pass
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names:
            # 形式：  0: 1.kongtaxian-Collapsed Cell
            s = line.strip()
            if ":" in s and s[0].isdigit():
                k, v = s.split(":", 1)
                try:
                    names[int(k)] = v.strip()
                except ValueError:
                    pass
            elif not s.startswith(" "):
                in_names = False
    if not nc:
        nc = len(names)
    return names, nc


def print_report(rep: dict[str, Any]) -> None:
    print("=" * 78)
    print(f"数据集体检  {rep['root']}")
    print("=" * 78)
    names = rep.get("names", {})
    print(f"data.yaml: {rep['data_yaml']}   nc={rep.get('nc')}")

    for split, s in rep["splits"].items():
        print(f"\n[{split}]")
        print(f"  图像 {s['images']}  标签 {s['labels']}  空标签(无缺陷) {s['empty_labels']}")
        print(f"  框总数 {s['total_boxes']}  平均每图 {s['avg_boxes_per_image']} 个")
        print(f"  每图框数分布: {s['boxes_per_image_hist']}")
        print("  类别分布:")
        for cid, n in sorted(s["class_counts"].items()):
            pct = n / s["total_boxes"] * 100 if s["total_boxes"] else 0
            bar = "█" * max(1, int(pct / 2))
            print(f"    {cid:>2} {names.get(cid, '?'):28s} {n:>4} 框 {pct:5.1f}%  {bar}")

    if rep["issues"]:
        print("\n❌ 必须修复的问题:")
        for i in rep["issues"]:
            print(f"   - {i}")
    if rep["warnings"]:
        print("\n⚠️ 需要注意:")
        for w in rep["warnings"]:
            print(f"   - {w}")
    if not rep["issues"] and not rep["warnings"]:
        print("\n✅ 未发现数据问题")
    print("=" * 78)


# --------------------------------------------------------------------------
# 训练
# --------------------------------------------------------------------------


def resolve_device(requested: str) -> tuple[str, dict[str, Any]]:
    """解析并校验训练设备，返回 (device 参数, 环境信息)。

    为什么必须显式检查：
      `pip install ultralytics` 可能把 CUDA 版 torch 替换成 CPU 版。
      这种情况下 `device=''` 会静默回落到 CPU，训练慢几十倍而不报错
      —— 跑一整夜才发现是最坏的情况。
    """
    info: dict[str, Any] = {"requested": requested or "auto"}
    try:
        import torch
    except ImportError:
        info["error"] = "未安装 torch"
        info["resolved"] = "cpu"
        info["warning"] = "未安装 PyTorch，无法使用 GPU。先安装 CUDA 版 torch。"
        return "cpu", info

    info["torch"] = torch.__version__
    info["cuda_bind"] = getattr(torch.version, "cuda", None)
    info["cuda_available"] = torch.cuda.is_available()
    info["is_cpu_wheel"] = "+cpu" in torch.__version__

    if requested == "cpu":
        info["resolved"] = "cpu"
        return "cpu", info

    if torch.cuda.is_available():
        idx = requested or "0"
        try:
            props = torch.cuda.get_device_properties(int(idx))
            info["gpu_name"] = props.name
            info["gpu_mem_gb"] = round(props.total_memory / 1024**3, 1)
        except Exception:  # noqa: BLE001
            pass
        info["resolved"] = idx
        return idx, info

    # 请求了 GPU 但不可用 —— 必须吵醒用户，不能静默降级
    info["resolved"] = "cpu"
    if info["is_cpu_wheel"]:
        info["warning"] = (
            "PyTorch 是 CPU-only 版本（版本号含 +cpu），无法使用 GPU。"
            "极可能是安装 ultralytics 时被 pip 替换。修复见 qc_check_gpu.py 的输出。"
        )
    elif not info["cuda_bind"]:
        info["warning"] = (
            "PyTorch 未绑定 CUDA（torch.version.cuda 为 None）。"
            "请重装 CUDA 版 torch。"
        )
    else:
        info["warning"] = (
            "torch 有 CUDA 绑定但检测不到可用 GPU：可能是驱动版本不足或显存被占满。"
            "先运行 nvidia-smi 检查。"
        )
    return "cpu", info


def cpu_safe_defaults(imgsz: int, batch: int, epochs: int) -> tuple[int, int, int, str]:
    """CPU 训练时自动调整参数，避免"能跑但跑不完"。

    实测估算（446 张训练集）：
      imgsz=640,  batch=8  → 约 7 分钟/epoch  → 50 epoch 约 6 小时
      imgsz=1024, batch=4  → 约 51 分钟/epoch → 50 epoch 约 43 小时
      imgsz=1536, batch=2  → 约 5 小时/epoch  → 不可行

    因此 CPU 下把 imgsz 压到 640、batch 提到 8（CPU 上大 batch 反而减少
    Python 循环开销），并把 epoch 上限设为 50。
    """
    notes: list[str] = []
    new_imgsz, new_batch, new_epochs = imgsz, batch, epochs

    if imgsz > 640:
        notes.append(f"imgsz {imgsz} → 640（CPU 上 1024 需 43 小时，不可行）")
        new_imgsz = 640
    if batch < 8:
        notes.append(f"batch {batch} → 8（CPU 上增大 batch 可减少循环开销）")
        new_batch = 8
    if epochs > 50:
        notes.append(f"epochs {epochs} → 50（CPU 上 100 轮约 12 小时，先跑 50）")
        new_epochs = 50

    return new_imgsz, new_batch, new_epochs, "；".join(notes)


def train(args: argparse.Namespace, data_yaml: Path) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("未安装 ultralytics：pip install ultralytics")
        sys.exit(1)

    if not BASE_WEIGHTS.exists():
        print(f"基础权重不存在: {BASE_WEIGHTS}")
        sys.exit(1)

    device, dev_info = resolve_device(args.device)

    print(f"\n{'=' * 78}")
    print("设备检查")
    print(f"{'=' * 78}")
    print(f"  torch {dev_info.get('torch')}   cuda绑定 {dev_info.get('cuda_bind')}   "
          f"可用 {dev_info.get('cuda_available')}")
    if dev_info.get("gpu_name"):
        print(f"  GPU: {dev_info['gpu_name']}  {dev_info.get('gpu_mem_gb')} GB")
    print(f"  最终使用设备: {device}")

    using_cpu = str(device).lower() == "cpu"

    if using_cpu and dev_info.get("warning"):
        print(f"\n  ⚠️ {dev_info['warning']}")

    if using_cpu and args.require_gpu:
        print("\n  ❌ 指定了 --require-gpu 但 GPU 不可用，已中止。")
        sys.exit(2)

    # ---- CPU 下自动降参数（除非用户显式 --no-auto-tune）----
    imgsz, batch, epochs = args.imgsz, args.batch, args.epochs
    if using_cpu and not args.no_auto_tune:
        imgsz, batch, epochs, note = cpu_safe_defaults(imgsz, batch, epochs)
        if note:
            print(f"\n  🔧 CPU 参数自动调整：{note}")
        print("  （如需保持原参数，加 --no-auto-tune）")

    # ---- dataloader worker 数：内存不足的主要元凶 ----
    # 每个 worker 是一个独立进程，各自 import torch（CUDA DLL 占 1~2GB 提交内存）
    # 并各自解码图像。原图 5472x3648 BMP 每张解码后 57MB，
    # workers=8 时内存/页面文件极易被撑爆（表现为 WinError 1455 / cv2 OutOfMemory）。
    workers = args.workers
    if workers is None:
        workers = 0 if using_cpu else 2
    if workers > 4:
        print(f"\n  ⚠️ workers={workers} 偏大：每个 worker 都会独立加载 torch，"
              f"原图较大时易耗尽内存。建议 ≤ 2。")

    # ---- 时间预估（让用户提前知道要等多久）----
    if using_cpu:
        iters = max(1, -(-446 // max(1, batch)))       # 训练集 446 张
        sec_per_iter = {640: 8.0, 768: 12.0, 1024: 27.0, 1280: 45.0, 1536: 80.0}
        sp = sec_per_iter.get(imgsz)
        if sp is None:
            sp = 8.0 * (imgsz / 640) ** 2              # 近似 O(面积)
        per_epoch_min = iters * sp / 60
        total_h = per_epoch_min * epochs / 60
        print(f"\n  ⏱  预估：每 epoch 约 {per_epoch_min:.0f} 分钟，"
              f"{epochs} 轮约 {total_h:.1f} 小时")
        if total_h > 12:
            print("     ⚠️ 超过 12 小时。建议减少 --epochs 或降低 --imgsz。")

    print(f"\n{'=' * 78}")
    print("开始训练")
    print(f"{'=' * 78}")
    print(f"  基础权重   {BASE_WEIGHTS}   （DOTA 15 类 OBB 预训练）")
    print(f"  数据配置   {data_yaml}")
    print(f"  imgsz={imgsz}  epochs={epochs}  batch={batch}  device={device}")
    print(f"  输出目录   {RUNS_DIR}")
    print()

    model = YOLO(str(BASE_WEIGHTS))
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        project=str(RUNS_DIR),
        name=args.name,
        exist_ok=True,
        # ---- 针对本数据集的关键调整 ----
        cache=False,          # 5472x3648 每张约 60MB，缓存会耗尽内存
        mosaic=0.5,           # 缺陷是局部结构，过度拼接破坏上下文
        close_mosaic=10,      # 最后 10 轮关闭 mosaic，让模型适应真实分布
        patience=30,          # 30 轮无提升则早停
        amp=False if using_cpu else True,  # CPU 上混合精度无收益且可能报错
        save=True,
        plots=True,
        verbose=True,
        seed=0,               # 可复现
    )

    best = Path(str(getattr(results, "save_dir", RUNS_DIR / args.name))) / "weights" / "best.pt"
    print(f"\n训练完成。最佳权重预期位于：{best}")
    if best.exists():
        print(f"确认存在，大小 {best.stat().st_size / 1024 / 1024:.1f} MB")
        # 写出可直接被 qc_yolo 使用的配置
        cfg = {
            "QC_YOLO_WEIGHTS": str(best),
            "QC_YOLO_IMGSZ": str(args.imgsz),
            "QC_YOLO_CONF": "0.25",
        }
        (RUNS_DIR / "best_weights.env").write_text(
            "\n".join(f"{k}={v}" for k, v in cfg.items()), encoding="utf-8"
        )
        print(f"\n已写出配置：{RUNS_DIR / 'best_weights.env'}")
        print("把它加到项目 .env，或直接设置环境变量 QC_YOLO_WEIGHTS 指向上面路径。")
    else:
        print("⚠️ 未找到 best.pt，请检查训练日志。")
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO11-OBB 训练（蜂窝催化剂缺陷）")
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--dry-run", action="store_true", help="只做数据体检，不训练")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024,
                    help="1024 是针对高分辨率小缺陷的经验值，不建议用默认 640")
    ap.add_argument("--batch", type=int, default=4,
                    help="RTX 3070 8GB + imgsz 1024 建议 4；显存不足降到 2 或 imgsz 768")
    ap.add_argument("--device", default="", help="留空自动；'0' 用 GPU0；'cpu' 用 CPU")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="[已默认允许，保留此参数仅为兼容] CPU 训练现在默认可行")
    ap.add_argument("--require-gpu", action="store_true",
                    help="GPU 不可用时中止训练（用于确认真的在用 GPU）")
    ap.add_argument("--no-auto-tune", action="store_true",
                    help="CPU 训练时不要自动下调 imgsz/上调 batch/限制 epochs")
    ap.add_argument("--workers", type=int, default=None,
                    help="dataloader 进程数。默认 GPU=2 / CPU=0。"
                         "原图很大或报 WinError 1455 时设为 0")
    ap.add_argument("--name", default="catalyst_obb")
    ap.add_argument("--report", default="", help="把体检报告写成 JSON")
    args = ap.parse_args()

    root = Path(args.data_root)
    rep = inspect_dataset(root)
    print_report(rep)

    if args.report:
        outp = Path(args.report)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n体检报告已写入 {outp}")

    if rep["issues"]:
        print("\n存在必须修复的问题，已中止训练。修复后重跑，或加 --force（不推荐）。")
        if not any(a == "--force" for a in sys.argv):
            return

    if args.dry_run:
        print("\n--dry-run 模式，跳过训练。")
        print("确认无误后执行：")
        print(f"  python qc_train.py --imgsz {args.imgsz} --batch {args.batch} --epochs {args.epochs}")
        return

    train(args, root / "data.yaml")


if __name__ == "__main__":
    main()
