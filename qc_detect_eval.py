# -*- coding: utf-8 -*-
"""检测指标评测：YOLO 预测 vs OBB 真值标签

## 这个脚本解决什么问题

前面的 `qc_eval.py` 评测的是「Agent 链路」（结构化输出、幻觉、拒答），
但**检测本身的准确率**（模型认没认出来缺陷、定位准不准）它测不了——
那需要真值标签。本脚本补上这块：

    val 集的 .txt 真值  ←→  YOLO 预测
        ↓
    类别正确率 / 漏检率 / 误检率 / IoU / 面积误差

## 为什么用"点是否在预测四边形内"做匹配

数据集是 OBB（任意四边形）。标准 IoU 需要多边形求交，实现成本高；
而质检真正关心的是"**这个真缺陷有没有被找到**"，用中心点归属判断
足以回答这个问题，且实现简单、可解释。

真正的 IoU 由 `ultralytics` 的 `model.val()` 给出（它是权威实现），
两者互补：本脚本给"漏了哪些、多报了哪些"，val() 给标准 mAP。

## 用法

    python qc_detect_eval.py                    # val 集全量
    python qc_detect_eval.py --limit 30          # 只跑 30 张（快速验证）
    python qc_detect_eval.py --split train       # 换训练集（看拟合程度）
    python qc_detect_eval.py --conf 0.25 --imgsz 1280
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

from qc_yolo import DEFAULT_CONF, DEFAULT_IMGSZ, DEFAULT_WEIGHTS, run_yolo

DATA_ROOT = Path(r"E:\yolodata")

# 类别名映射表：由模型实际的 names 动态构建（不按索引硬映射）。
# 模块加载时持有一个"按数据集 class_id"的默认表；跑评测时会用模型真实 names 覆盖。
from qc_yolo import CLASS_META, DATA_CLASS_KEYS, build_class_map  # noqa: E402

_CLASS_NAMES: dict[int, str] = {
    cid: CLASS_META[key][0] for cid, key in DATA_CLASS_KEYS.items()
}


def set_class_names_from_model(weights: str) -> dict[int, str]:
    """读取模型真实类别名，构建 idx→中文名 映射，覆盖默认表。

    为什么必须这样做：实测权重只有 11 类且缺 "9.biaopiliewen"，
    若按数据集索引硬映射，idx 8 之后全部错位一格。
    """
    global _CLASS_NAMES
    try:
        from ultralytics import YOLO

        m = YOLO(str(weights))
        names = {int(k): str(v) for k, v in (getattr(m, "names", {}) or {}).items()}
        cmap, unmapped = build_class_map(names)
        _CLASS_NAMES = {i: v["name"] for i, v in cmap.items()}
        return _CLASS_NAMES
    except Exception:  # noqa: BLE001 —— 读取失败时退回默认表，不影响主流程
        return _CLASS_NAMES


def cname(cid: int) -> str:
    return _CLASS_NAMES.get(cid, f"class_{cid}")


# --------------------------------------------------------------------------
# 真值标签解析与几何工具
# --------------------------------------------------------------------------


def read_gt_labels(label_path: Path) -> list[dict[str, Any]]:
    """解析 YOLO OBB 标签：class x1 y1 x2 y2 x3 y3 x4 y4（归一化）。

    空文件 = 该图无缺陷（正常样本），返回空列表 —— 这是有效信息，不是错误。
    """
    out: list[dict[str, Any]] = []
    if not label_path.exists():
        return out
    for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 9:
            continue
        try:
            cls = int(float(parts[0]))
            pts = np.array([float(v) for v in parts[1:9]], dtype=float).reshape(4, 2)
        except ValueError:
            continue
        out.append({"class_id": cls, "polygon_norm": pts})
    return out


def to_pixels(polygon_norm: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = polygon_norm.copy()
    pts[:, 0] *= width
    pts[:, 1] *= height
    return pts


def point_in_polygon(px: float, py: float, poly: np.ndarray) -> bool:
    """射线法判断点是否在四边形内。"""
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > py) != (y2 > py):
            x_cross = (x2 - x1) * (py - y1) / (y2 - y1 + 1e-12) + x1
            if px < x_cross:
                inside = not inside
    return inside


def poly_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


# --------------------------------------------------------------------------
# 匹配：真值 ↔ 预测
# --------------------------------------------------------------------------


def match_frame(
    gt_list: list[dict[str, Any]],
    pred_list: list[dict[str, Any]],
    width: int,
    height: int,
) -> dict[str, Any]:
    """单帧匹配。判定口径：

    * 命中（tp）：真值中心点落在某预测四边形内，且类别一致
    * 类别错：中心点落在预测框内但类别不一致（定位对、认错种类）
    * 漏检（fn）：中心点不在任何预测框内
    * 误检（fp）：没有真值中心点落在其内的预测框
    """
    gt_polys = []
    for g in gt_list:
        pts = to_pixels(g["polygon_norm"], width, height)
        gt_polys.append(
            {
                "class_id": g["class_id"],
                "poly": pts,
                "center": (float(pts[:, 0].mean()), float(pts[:, 1].mean())),
                "area": poly_area(pts),
            }
        )

    tp, miscls, fn = 0, 0, []
    miscls_classes: list[int] = []
    matched_pred_idx: set[int] = set()

    for g in gt_polys:
        cx, cy = g["center"]
        hit_idx = None
        for i, p in enumerate(pred_list):
            poly = np.array(p["polygon_px"], dtype=float)
            if point_in_polygon(cx, cy, poly):
                hit_idx = i
                break
        if hit_idx is None:
            fn.append(g)
        else:
            matched_pred_idx.add(hit_idx)
            if pred_list[hit_idx]["class_id"] == g["class_id"]:
                tp += 1
            else:
                miscls += 1
                miscls_classes.append(g["class_id"])

    fp = len(pred_list) - len(matched_pred_idx)

    # 面积误差：仅在类别一致的命中上统计
    area_errs: list[float] = []
    len_errs: list[float] = []
    for g in gt_polys:
        cx, cy = g["center"]
        for p in pred_list:
            if p["class_id"] != g["class_id"]:
                continue
            if point_in_polygon(cx, cy, np.array(p["polygon_px"], dtype=float)):
                pred_area = p["area_px"]
                if g["area"] > 0:
                    area_errs.append(abs(pred_area - g["area"]) / g["area"])
                gt_edges = [
                    float(np.linalg.norm(g["poly"][(k + 1) % 4] - g["poly"][k]))
                    for k in range(4)
                ]
                gt_max = max(gt_edges)
                if gt_max > 0:
                    pred_max = max(
                        float(np.linalg.norm(
                            np.array(p["polygon_px"][(k + 1) % 4]) - np.array(p["polygon_px"][k])
                        ))
                        for k in range(4)
                    )
                    # 长边像素 → 由调用方换算，这里只给像素比
                    len_errs.append(abs(pred_max - gt_max) / gt_max)
                break

    return {
        "gt": len(gt_polys),
        "pred": len(pred_list),
        "tp": tp,
        "misclassified": miscls,
        "fn": len(fn),
        "fp": fp,
        "fn_classes": [g["class_id"] for g in fn],
        "miscls_classes": miscls_classes,
        "area_rel_err": float(np.mean(area_errs)) if area_errs else None,
        "len_rel_err": float(np.mean(len_errs)) if len_errs else None,
    }


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO 检测指标评测")
    ap.add_argument("--split", default="val", choices=["val", "train", "正常"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--gsd", type=float, default=0.027,
                    help="mm/px，用于把面积误差换算成物理量（默认 0.027≈样品宽150mm）")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    # 用模型真实类别名构建映射（权重类别数/顺序可能与数据集不同）
    real_names = set_class_names_from_model(args.weights)
    print(f"模型类别映射（{len(real_names)} 类）:")
    for i in sorted(real_names):
        print(f"  {i:>2}: {real_names[i]}")
    print()

    img_dir = DATA_ROOT / "images" / args.split
    lbl_dir = DATA_ROOT / "labels" / args.split
    images = sorted(img_dir.glob("*.bmp"))
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(f"未找到图像: {img_dir}")
        sys.exit(1)

    print("=" * 78)
    print(f"检测评测  split={args.split}  图像 {len(images)} 张  "
          f"conf={args.conf}  imgsz={args.imgsz}")
    print(f"权重 {args.weights}")
    print(f"GSD {args.gsd} mm/px（仅用于把像素误差换算成 mm，不影响匹配）")
    print("=" * 78)

    agg = defaultdict(int)
    area_errs_all: list[float] = []
    len_errs_all: list[float] = []
    fn_by_class: dict[int, int] = defaultdict(int)
    miscls_by_class: dict[int, int] = defaultdict(int)
    gt_by_class: dict[int, int] = defaultdict(int)     # 各类别真值总数
    normal_fp = 0
    normal_imgs = 0
    normals_with_fp = 0
    t_start = time.monotonic()
    per_image: list[dict[str, Any]] = []

    for idx, img in enumerate(images, 1):
        lbl = lbl_dir / (img.stem + ".txt")
        gt = read_gt_labels(lbl)

        try:
            pred = run_yolo(str(img), args.gsd, weights=args.weights,
                            conf=args.conf, imgsz=args.imgsz)
        except Exception as exc:  # noqa: BLE001
            print(f"  [跳过] {img.name}: {type(exc).__name__}: {exc}")
            continue

        w, h = pred["image_size_px"]
        m = match_frame(gt, pred["defects"], w, h)

        agg["gt"] += m["gt"]
        agg["pred"] += m["pred"]
        agg["tp"] += m["tp"]
        agg["misclassified"] += m["misclassified"]
        agg["fn"] += m["fn"]
        agg["fp"] += m["fp"]

        # 按类别累计真值总数、漏检数、类别判错数（用于 per-class 召回率）
        for g in gt:
            gt_by_class[g["class_id"]] += 1
        for c in m["fn_classes"]:
            fn_by_class[c] += 1
        for c in m["miscls_classes"]:
            miscls_by_class[c] += 1
        if m["area_rel_err"] is not None:
            area_errs_all.append(m["area_rel_err"])
        if m["len_rel_err"] is not None:
            len_errs_all.append(m["len_rel_err"])

        is_normal = len(gt) == 0
        if is_normal:
            normal_imgs += 1
            if m["pred"] > 0:
                normals_with_fp += 1
                normal_fp += m["pred"]

        per_image.append({
            "image": img.name, "gt": m["gt"], "pred": m["pred"],
            "tp": m["tp"], "fn": m["fn"], "fp": m["fp"],
            "misclassified": m["misclassified"],
        })

        if idx % 20 == 0 or idx == len(images):
            print(f"  已处理 {idx}/{len(images)}  "
                  f"tp={agg['tp']} fn={agg['fn']} fp={agg['fp']}")

    elapsed = time.monotonic() - t_start

    # ---- 汇总指标 ----
    tp, fn, fp = agg["tp"], agg["fn"], agg["fp"]
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None

    print("\n" + "=" * 78)
    print("汇总指标")
    print("=" * 78)
    print(f"  真值缺陷总数        {agg['gt']}")
    print(f"  预测缺陷总数        {agg['pred']}")
    print(f"  命中 (TP)           {tp}")
    print(f"  漏检 (FN)           {fn}")
    print(f"  误检 (FP)           {fp}")
    print(f"  类别判错            {agg['misclassified']}  （定位命中但类别不符）")
    print(f"  Precision           {precision:.4f}" if precision is not None else "  Precision           n/a")
    print(f"  Recall              {recall:.4f}" if recall is not None else "  Recall              n/a")
    print(f"  F1                  {f1:.4f}" if f1 is not None else "  F1                  n/a")

    if area_errs_all:
        arr = np.array(area_errs_all)
        print(f"  面积相对误差 均值   {arr.mean():.2%}   中位 {np.median(arr):.2%}   P90 {np.percentile(arr, 90):.2%}")
    if len_errs_all:
        arr = np.array(len_errs_all)
        print(f"  长边相对误差 均值   {arr.mean():.2%}   中位 {np.median(arr):.2%}   P90 {np.percentile(arr, 90):.2%}")

    print(f"\n  正常图（无标签）      {normal_imgs} 张")
    if normal_imgs:
        print(f"  其中产生误检          {normals_with_fp} 张 "
              f"（{normals_with_fp / normal_imgs:.1%}），共 {normal_fp} 个误检框")

    if fn_by_class or gt_by_class:
        print("\n  各类别检出情况（按类别统计）:")
        print(f"    {'类别':<22} {'真值':>6} {'漏检':>6} {'类别错':>7} {'召回率':>8}")
        all_ids = sorted(set(gt_by_class) | set(fn_by_class) | set(miscls_by_class))
        for cid in all_ids:
            name = cname(cid)
            g = gt_by_class.get(cid, 0)
            fn_c = fn_by_class.get(cid, 0)
            mc = miscls_by_class.get(cid, 0)
            hit = g - fn_c - mc
            rec = (hit / g) if g else None
            rec_s = f"{rec:.1%}" if rec is not None else "n/a"
            flag = "  ← 样本过少" if g < 10 else ""
            print(f"    {name:<22} {g:>6} {fn_c:>6} {mc:>7} {rec_s:>8}{flag}")
        print("\n  说明：召回率 = (真值数 - 漏检 - 类别错) / 真值数")
        print("        真值数 < 10 的类别统计不稳定，不建议单独引用")

    print(f"\n  总耗时 {elapsed:.1f}s，平均 {elapsed / len(images):.2f}s/张")

    print("\n" + "-" * 78)
    print("提示：以上 TP/FP 判定用'真值中心点是否落在预测框内'，")
    print("      不依赖真实多边形求交，对'是否检出'这一问题足够；")
    print("      标准 mAP/IoU 请用 ultralytics 官方 val：")
    print(f"        yolo obb val model={args.weights} data={DATA_ROOT / 'data.yaml'} imgsz={args.imgsz}")
    print("-" * 78)

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps({
            "split": args.split, "conf": args.conf, "imgsz": args.imgsz,
            "weights": args.weights, "gsd": args.gsd,
            "images": len(images), "elapsed_s": round(elapsed, 1),
            "aggregate": dict(agg),
            "precision": precision, "recall": recall, "f1": f1,
            "area_rel_err_mean": float(np.mean(area_errs_all)) if area_errs_all else None,
            "len_rel_err_mean": float(np.mean(len_errs_all)) if len_errs_all else None,
            "normal_images": normal_imgs,
            "normal_images_with_fp": normals_with_fp,
            "fn_by_class": dict(fn_by_class),
            "per_image": per_image,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {outp}")


if __name__ == "__main__":
    main()
