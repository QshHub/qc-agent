# -*- coding: utf-8 -*-
"""权重深度诊断：为什么零检出？

零检出（conf=0.01 仍为 0）通常不是"没收敛"，而是结构/加载问题。逐一排查：

  1. 是否为 OBB 模型？普通 detect 模型没有 `obb` 输出，必然零检出
  2. 检测头类别数是否 = 12？若是 80（COCO）说明加载了错误权重
  3. 模型是否真的含训练权重？若 head 参数仍是随机初始化，输出会极低
  4. 模型在**训练集**图上的表现？训练集都零检出 → 几乎必然是结构问题
  5. 直接用低层 API 看原始输出，绕过 ultralytics 的置信度过滤

用法：
    python qc_diag_weights.py
    python qc_diag_weights.py --weights 路径 --imgsz 640
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

DATA_ROOT = Path(r"E:\yolodata")


def hr(t: str) -> None:
    print("\n" + "=" * 74)
    print(t)
    print("=" * 74)


def main() -> None:
    ap = argparse.ArgumentParser(description="权重深度诊断")
    ap.add_argument("--weights", default=r"E:\yolodata\best.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    import torch
    from ultralytics import YOLO

    wpath = Path(args.weights)
    hr("① 基本信息")
    print(f"  权重路径  {wpath}")
    print(f"  存在      {wpath.exists()}")
    print(f"  大小      {wpath.stat().st_size / 1024 / 1024:.2f} MB" if wpath.exists() else "")
    print(f"  torch     {torch.__version__}")
    print(f"  CUDA      {torch.cuda.is_available()}")

    if not wpath.exists():
        sys.exit(1)

    # ---- 用 torch 直接读 checkpoint，绕过 ultralytics 封装 ----
    hr("② checkpoint 原始结构（torch.load）")
    try:
        ckpt = torch.load(str(wpath), map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(str(wpath), map_location="cpu")
    if isinstance(ckpt, dict):
        print(f"  top-level keys: {list(ckpt.keys())[:12]}")
        for k in ("date", "version", "epoch", "best_fitness", "train_args", "train_metrics"):
            if k in ckpt:
                v = ckpt[k]
                s = str(v)
                print(f"  {k}: {s[:300]}")
        m = ckpt.get("model")
        if m is not None:
            print(f"  model 类型: {type(m).__name__}")
            print(f"  model.task: {getattr(m, 'task', '<无>')}")
            names = getattr(m, "names", None)
            if names is not None:
                print(f"  model.names 数量: {len(names)}")
                print(f"  model.names 前5: {list(names.items())[:5] if hasattr(names,'items') else names[:5]}")
            head = getattr(m, "model", None)
            if head is not None and len(head) > 0:
                last = head[-1]
                print(f"  最后一层(检测头)类型: {type(last).__name__}")
                for attr in ("nc", "nl", "no", "reg_max", "stride"):
                    if hasattr(last, attr):
                        print(f"    {attr} = {getattr(last, attr)}")
                # OBB 头应有 cv2/cv3 之外的 angle 分支；普通 Detect 没有
                print(f"    head 属性: {[a for a in dir(last) if not a.startswith('_')][:18]}")
    else:
        print(f"  ckpt 类型: {type(ckpt)}")

    # ---- ultralytics 视角 ----
    hr("③ ultralytics 加载后的视图")
    model = YOLO(str(wpath))
    print(f"  task        {getattr(model, 'task', '<无>')}")
    print(f"  names 数量  {len(getattr(model, 'names', {}) or {})}")
    names = getattr(model, "names", {}) or {}
    for k in sorted(names)[:14]:
        print(f"    {k}: {names[k]}")

    nn = getattr(model, "model", None)
    if nn is not None:
        print(f"  模型 wrapper: {type(nn).__name__}")
        inner = getattr(nn, "model", None)
        if inner is not None and len(inner) > 0:
            last = inner[-1]
            print(f"  最后一层: {type(last).__name__}  nc={getattr(last,'nc','?')}")
        # 参数量与"是否全为零/随机"的粗判
        try:
            sd = nn.state_dict()
            keys = [k for k in sd if "cv3" in k or "one2one" in k]
            print(f"  检测头参数张量数: {len(keys)}")
            if keys:
                k0 = keys[-1]
                t = sd[k0]
                print(f"    {k0}: shape={tuple(t.shape)} "
                      f"mean={t.float().mean():.5f} std={t.float().std():.5f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  state_dict 读取失败: {exc}")

    # ---- 推理：训练集 + 验证集，极低阈值 ----
    hr("④ 推理探测（训练集 2 张 + 验证集 2 张，conf=0.001）")
    imgs = (sorted(glob.glob(str(DATA_ROOT / "images" / "train" / "*.bmp")))[:2]
            + sorted(glob.glob(str(DATA_ROOT / "images" / "val" / "*.bmp")))[:2])
    if not imgs:
        print("  未找到图像")
        sys.exit(1)

    dev = "0" if torch.cuda.is_available() else "cpu"
    for p in imgs:
        try:
            r = model.predict(source=p, conf=0.001, imgsz=args.imgsz,
                              device=dev, verbose=False, max_det=100)[0]
        except Exception as exc:  # noqa: BLE001
            print(f"\n  {Path(p).name}: 推理异常 {type(exc).__name__}: {exc}")
            continue

        boxes = getattr(r, "boxes", None)
        obb = getattr(r, "obb", None)
        n_box = len(boxes) if boxes is not None else 0
        n_obb = len(obb) if obb is not None else 0
        print(f"\n  {Path(p).name}")
        print(f"    boxes  {n_box} 个   obb {n_obb} 个")

        if obb is not None and n_obb:
            confs = obb.conf.cpu().numpy()
            print(f"    obb 置信度: 最大 {confs.max():.5f} 中位 {float(__import__('statistics').median(confs)):.5f}")
        if boxes is not None and n_box:
            confs = boxes.conf.cpu().numpy()
            print(f"    boxes 置信度: 最大 {confs.max():.5f}")

        # 直接看原始预测张量（绕过 NMS 过滤）
        try:
            import numpy as np
            from PIL import Image

            from ultralytics.data.augment import LetterBox

            im = np.asarray(Image.open(p).convert("RGB"))
            lb = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False)
            im2 = lb(image=im)
            t = torch.from_numpy(im2.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            t = t.to(dev)
            raw = model.model(t)
            if isinstance(raw, (list, tuple)):
                r0 = raw[0]
                print(f"    原始输出类型 {type(raw).__name__}, 首项 shape {tuple(r0.shape) if hasattr(r0,'shape') else '?'}")
                if hasattr(r0, "shape") and r0.dim() == 3:
                    # shape: (B, 4+nc(+1 angle), anchors)
                    b, a, c = r0.shape
                    print(f"    -> B={b} A={a} C={c}  （C 通常 = 4 + nc [+1 angle]）")
                    # 取各通道最大值，看是否有任何激活
                    mx = r0.amax(dim=(0, 2)).detach().cpu().numpy()
                    print(f"    各通道最大值前 8: {np.round(np.sort(mx)[-8:], 3).tolist()}")
        except Exception as exc:  # noqa: BLE001
            print(f"    原始输出探测失败（不影响主诊断）: {type(exc).__name__}: {exc}")

    hr("⑤ 判定")
    print("  根据上面输出对照：")
    print("  · task 不是 'obb'           → 加载的是普通检测模型，与 OBB 标签不兼容")
    print("  · names 数量 是 80           → 加载了 COCO 权重，不是你的训练结果")
    print("  · 检测头 nc 不是 12         → 同上")
    print("  · 训练集也 0 检出           → 结构性问题（不是过拟合/欠拟合问题）")
    print("  · 训练集有检出但验证集没有   → 欠拟合，需重新训练")
    print("  · 原始输出各通道最大值全≈0  → 权重未加载或已损坏")


if __name__ == "__main__":
    main()
