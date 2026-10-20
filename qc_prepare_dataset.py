# -*- coding: utf-8 -*-
"""数据集精简：把 5472x3648 的 BMP 缩到训练所需尺寸

## 为什么需要这一步

你的原始图像是 **5472×3648 的 BMP**，每张解码后占 **57 MB** 内存。
559 张全解码需 **31 GB**，远超物理内存与页面文件能承受的量。

训练时 ultralytics 会把图缩到 `imgsz`（如 640），但**缩放发生在解码之后**：
先完整读入 57MB，再缩到 640。所以瓶颈在解码阶段，降低 imgsz 并不省内存。

## 关键洞察：OBB 标签是归一化坐标

`labels/*.txt` 里存的是 **0~1 归一化坐标**（如 `0.580571 0.607875 ...`）。
等比例缩放图像后，归一化坐标**完全不变**，因此：

    ★ 缩放图像不需要改动任何标签文件

这让预处理变得零风险。

## 做法

把图像等比缩放到长边 = `--max-side`（默认 1280），存为 **JPEG 质量 92**：

    内存占用  57 MB → 约 3.1 MB（约 1/18）
    磁盘占用  10.76 GB → 约 0.27 GB
    训练精度  1280 > 训练 imgsz 640，保留了充足余量

## 用法

    python qc_prepare_dataset.py                    # 生成 E:\\yolodata_1280
    python qc_prepare_dataset.py --max-side 1600    # 长边 1600
    python qc_prepare_dataset.py --check            # 只报告不生成
    python qc_prepare_dataset.py --format bmp       # 仍用 BMP（不推荐）
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

SRC = Path(r"E:\yolodata")
SPLITS = ("train", "val")


def scan(src: Path) -> dict:
    info: dict = {"splits": {}, "total_images": 0, "total_bytes": 0}
    for sp in SPLITS:
        d = src / "images" / sp
        files = sorted(p for p in d.glob("*") if p.suffix.lower() in (".bmp", ".jpg", ".jpeg", ".png"))
        size = sum(p.stat().st_size for p in files)
        lbl = src / "labels" / sp
        n_lbl = len(list(lbl.glob("*.txt"))) if lbl.exists() else 0
        info["splits"][sp] = {"images": len(files), "labels": n_lbl, "bytes": size}
        info["total_images"] += len(files)
        info["total_bytes"] += size
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="数据集精简（BMP 大图 → JPEG 小图）")
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--dst", default="")
    ap.add_argument("--max-side", type=int, default=1280,
                    help="长边缩放到多少像素；应 >= 训练 imgsz（默认 1280）")
    ap.add_argument("--quality", type=int, default=92, help="JPEG 质量 1-100")
    ap.add_argument("--format", default="jpg", choices=["jpg", "png", "bmp"])
    ap.add_argument("--check", action="store_true", help="只报告，不生成")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    if args.dst:
        dst = Path(args.dst)
    else:
        # 默认生成同级目录，如 E:\yolodata -> E:\yolodata_1280
        dst = src.parent / f"{src.name}_{args.max_side}"

    st = scan(src)
    print("=" * 76)
    print("源数据集")
    print("=" * 76)
    print(f"  路径  {src}")
    for sp, v in st["splits"].items():
        print(f"  [{sp}] 图 {v['images']}  标签 {v['labels']}  "
              f"{v['bytes']/1024**3:.2f} GB")
    print(f"  合计  {st['total_images']} 张  {st['total_bytes']/1024**3:.2f} GB")

    # 估算目标体积与内存
    print("\n" + "=" * 76)
    print("目标估算")
    print("=" * 76)
    est_mb = st["total_images"] * 0.5      # JPEG q92 在 1280 长边下约 0.3~0.8MB
    print(f"  长边      {args.max_side} px")
    print(f"  格式      {args.format.upper()} (质量 {args.quality})")
    print(f"  预估磁盘  约 {est_mb/1024:.2f} GB（原 {st['total_bytes']/1024**3:.2f} GB）")
    pix = args.max_side * (args.max_side * 0.667)
    print(f"  单张解码  约 {pix*3/1024/1024:.1f} MB（原 57 MB）")
    print(f"  全量解码  约 {st['total_images']*pix*3/1024**3:.1f} GB（原 31 GB）")

    if args.max_side < 640:
        print(f"\n  ⚠️ max_side={args.max_side} 小于训练 imgsz 640，会有损失")

    if args.check:
        print("\n--check 模式，不生成文件。")
        return

    # ---- 生成 ----
    print("\n" + "=" * 76)
    print(f"生成中 -> {dst}")
    print("=" * 76)

    if dst.exists() and not args.overwrite:
        # 允许续做：已存在的图跳过
        print("  目标目录已存在，将跳过已转换的文件（可加 --overwrite 强制重做）")

    ext = {"jpg": ".jpg", "png": ".png", "bmp": ".bmp"}[args.format]
    total_done = 0
    total_skip = 0
    failed: list[str] = []

    for sp in SPLITS:
        s_img = src / "images" / sp
        d_img = dst / "images" / sp
        d_img.mkdir(parents=True, exist_ok=True)

        # 标签直接复制（归一化坐标，缩放后无需改动）
        s_lbl = src / "labels" / sp
        d_lbl = dst / "labels" / sp
        if s_lbl.exists():
            d_lbl.mkdir(parents=True, exist_ok=True)
            for f in s_lbl.glob("*.txt"):
                shutil.copy2(f, d_lbl / f.name)

        files = sorted(p for p in s_img.glob("*")
                       if p.suffix.lower() in (".bmp", ".jpg", ".jpeg", ".png"))
        print(f"\n[{sp}] {len(files)} 张")

        for i, f in enumerate(files, 1):
            out = d_img / (f.stem + ext)
            if out.exists() and out.stat().st_size > 1024 and not args.overwrite:
                total_skip += 1
                continue
            try:
                with Image.open(f) as im:
                    im = im.convert("RGB")
                    w, h = im.size
                    s = args.max_side / max(w, h)
                    if s < 1.0:
                        im = im.resize((max(1, round(w * s)), max(1, round(h * s))),
                                       Image.LANCZOS)
                    if args.format == "jpg":
                        im.save(out, "JPEG", quality=args.quality, optimize=True)
                    elif args.format == "png":
                        im.save(out, "PNG", optimize=True)
                    else:
                        im.save(out, "BMP")
                total_done += 1
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{f.name}: {type(exc).__name__}: {exc}")

            if i % 50 == 0 or i == len(files):
                print(f"  {i}/{len(files)}  已转换 {total_done}  跳过 {total_skip}  失败 {len(failed)}")

    # ---- 写 data.yaml ----
    names_src = None
    y = src / "data.yaml"
    if y.exists():
        names_src = y.read_text(encoding="utf-8", errors="replace")

    out_yaml = dst / "data.yaml"
    if names_src:
        # 保留原 names 与 nc，只改 train/val 指向（相对 dst）
        lines = ["# 由 qc_prepare_dataset.py 生成", f"# 源: {src}", ""]
        for line in names_src.splitlines():
            s = line.strip()
            if s.startswith("train:") or s.startswith("val:"):
                continue
            lines.append(line)
        # 插入路径
        final = []
        for line in lines:
            final.append(line)
        txt = "train: images/train\nval: images/val\n" + "\n".join(
            x for x in final if x.strip() and not x.strip().startswith("#")
        )
        out_yaml.write_text(txt + "\n", encoding="utf-8")
        print(f"\n已写 {out_yaml}")

    # ---- 校验 ----
    print("\n" + "=" * 76)
    print("校验")
    print("=" * 76)
    new_st = scan(dst)
    for sp, v in new_st["splits"].items():
        ok = v["images"] == st["splits"][sp]["images"]
        print(f"  [{sp}] 图 {v['images']}/{st['splits'][sp]['images']} "
              f"{'✓' if ok else '✗ 数量不符'}  标签 {v['labels']}")
    print(f"  体积  {st['total_bytes']/1024**3:.2f} GB → {new_st['total_bytes']/1024**3:.2f} GB "
          f"(压缩 {st['total_bytes']/max(1,new_st['total_bytes']):.1f}x)")

    if failed:
        print(f"\n  ❌ {len(failed)} 张失败:")
        for x in failed[:10]:
            print(f"     {x}")

    print("\n" + "=" * 76)
    print("下一步：用新数据集训练")
    print("=" * 76)
    print(f"  python qc_train.py --data-root \"{dst}\" --epochs 100 --imgsz 640 --batch 8 --device 0 --workers 2")
    print("\n  说明：")
    print("   · 标签为归一化坐标，缩放后无需改动，已原样复制")
    print("   · max_side 1280 > 训练 imgsz 640，保留充足余量")
    print("   · 若还要更高精度，可 --max-side 1600 重新生成")


if __name__ == "__main__":
    main()
