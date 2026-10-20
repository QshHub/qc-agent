# -*- coding: utf-8 -*-
"""YOLO 检测后端（OBB 旋转框）

## 在系统中的位置

    segment_defects(image_path, gsd)      ← Agent 调用的工具，契约不变
        └→ annotate_defects(image_path, gsd)
              └→ run_yolo(...)            ← 本文件，替换原来的像素连通域分析

`annotate_defects` 的**返回结构完全不变**，因此 Agent、工具、报告 schema、
复核流程、评测器全都无需改动 —— 这就是把"检测后端"抽象成单一函数的价值。

## OBB 的关键细节（踩坑点）

1. **必须用 `obb.xyxyxyxy` 而不是 `obb.xywhr`**
   官方文档明确：预测框的 w/h **不一定按长边在前**，旋转角可能从短边量起。
   只有 `xyxyxyxy`（四角点）是几何无歧义的。长边从四角点自行计算。

2. **面积用旋转矩形而非轴对齐包围盒**
   OBB 的角点是旋转后的，直接按角点顺序算多边形面积（鞋带公式），
   比 `w*h` 更准确地反映实际覆盖面积；对细长裂缝尤其重要。

3. **类别映射**
   数据集 12 类与质检知识库的 12 小类不是同一套命名，
   因此用 `CLASS_MAP` 显式映射，映射缺失时保留原名 + 标记 `unmapped`，
   绝不做模糊猜测（错映射比不映射更危险）。

## 降级策略

`ultralytics` 未安装或权重缺失时，`run_yolo` 抛 `YoloUnavailable`，
由 `annotate_defects` 捕获并降级到像素后端。
这样代码在没有 GPU / 没有权重的环境下仍然可运行（便于 CI 与他人复现）。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------
# 配置（可用环境变量覆盖，便于切换实验）
# --------------------------------------------------------------------------

DEFAULT_WEIGHTS = os.getenv("QC_YOLO_WEIGHTS", r"E:\yolodata\best.pt")
DEFAULT_CONF = float(os.getenv("QC_YOLO_CONF", "0.25"))
# 推理尺寸：应与**训练时**的 imgsz 一致，否则特征尺度不匹配、精度下降。
# 当前权重在 imgsz=640 下训练（见 runs/smoke_test/args.yaml），故默认 640。
# 若将来用 1024 重新训练，记得同步改这里或设 QC_YOLO_IMGSZ=1024。
DEFAULT_IMGSZ = int(os.getenv("QC_YOLO_IMGSZ", "640"))
DEFAULT_DEVICE = os.getenv("QC_YOLO_DEVICE", "")  # "" = 自动；"cpu" / "0"

# --------------------------------------------------------------------------
# 类别映射
#
# ⚠️ 重要：不能按模型索引硬映射
#
# 实测当前权重（E:\yolodata\best.pt）：
#   task = detect（不是 obb），nc = 11（不是 12）
#   缺失的类别是数据集里的 "9.biaopiliewen"（表皮裂纹）
#   因此模型 idx 8 之后与数据集 class_id 错位一格：
#       model[8]=横向裂纹(data 9)  model[9]=长度不足(data 10)  model[10]=损坏(data 11)
#
# 所以映射按 **类别名** 匹配，而不是按索引。新增/缺失类别时只需维护 DATA_CLASS_KEYS。
# --------------------------------------------------------------------------

# 数据集 class_id → 质检知识库类别键（12 类，与 data.yaml 一一对应）
DATA_CLASS_KEYS: dict[int, str] = {
    0: "material_loss.collapse",
    1: "material_loss.channel_missing",
    2: "material_loss.face_chip",
    3: "gap_separation.face_crack",
    4: "material_loss.missing_cell_wall",
    5: "material_loss.sidewall_chip",
    6: "material_loss.corner_chip",
    7: "gap_separation.external_wall_tear",
    8: "gap_separation.skin_crack",
    9: "gap_separation.transverse_crack",
    10: "deformation.shortness",
    11: "material_loss.damage",
}

# 类别键 → (中文名, 大类)
CLASS_META: dict[str, tuple[str, str]] = {
    "material_loss.collapse": ("孔塌陷", "材料缺失类"),
    "material_loss.channel_missing": ("孔洞", "材料缺失类"),
    "material_loss.face_chip": ("端面缺口", "材料缺失类"),
    "gap_separation.face_crack": ("端面裂纹", "缝隙分离类"),
    "material_loss.missing_cell_wall": ("孔壁缺失", "材料缺失类"),
    "material_loss.sidewall_chip": ("表皮缺口", "材料缺失类"),
    "material_loss.corner_chip": ("缺角", "材料缺失类"),
    "gap_separation.external_wall_tear": ("侧壁撕裂", "缝隙分离类"),
    "gap_separation.skin_crack": ("表皮裂纹", "缝隙分离类"),
    "gap_separation.transverse_crack": ("横向裂纹", "缝隙分离类"),
    "deformation.shortness": ("长度不足", "结构形变类"),
    "material_loss.damage": ("损坏", "材料缺失类"),
}

# 中文名 → 类别键（用于把模型的 names 反查回知识库键）
NAME_TO_KEY: dict[str, str] = {cn: key for key, (cn, _) in CLASS_META.items()}


def _class_key_from_model_name(name: str) -> str | None:
    """从模型类别名解析出质检知识库类别键。

    模型里的名字形如 "1.kongtaxian-Collapsed Cell"、"10.hengxiangliewen-Transverse cracks"。
    策略：
      1) 先剥掉前缀 "N." 得到主体
      2) 用中文名精确匹配（最可靠）
      3) 用拼音/英文关键词兜底匹配
    匹配不上返回 None（调用方标记 unmapped，绝不猜测）。
    """
    body = name.strip()
    # 剥掉 "12." 这类编号前缀
    import re

    body = re.sub(r"^\d+\.", "", body).strip()

    # 中文名精确匹配：模型名里若含中文，直接命中
    for cn, key in NAME_TO_KEY.items():
        if cn in name:
            return key

    # 拼音/英文关键词兜底
    lower = body.lower()
    keyword_map: list[tuple[tuple[str, ...], str]] = [
        (("collapse", "kongtaxian", "孔塌陷"), "material_loss.collapse"),
        (("hole", "kongdong", "孔洞"), "material_loss.channel_missing"),
        (("facechip", "face chip", "duanmianquekou", "端面缺口"), "material_loss.face_chip"),
        (("facecrack", "face crack", "duanmianliewen", "端面裂纹"), "gap_separation.face_crack"),
        (("missingcellwall", "missing cell wall", "kongbiqueshi"), "material_loss.missing_cell_wall"),
        (("sidewallchip", "sidewall chip", "biaopiquekou", "表皮缺口"), "material_loss.sidewall_chip"),
        (("cornerchip", "corner chip", "quejiao", "缺角"), "material_loss.corner_chip"),
        (("externalwalltear", "external wall tear", "cebisilie", "侧壁撕裂"),
         "gap_separation.external_wall_tear"),
        (("biaopiliewen", "表破裂纹", "表皮裂纹"), "gap_separation.skin_crack"),
        (("transverse", "hengxiang", "横向裂纹"), "gap_separation.transverse_crack"),
        (("shortness", "changdubuzu", "长度不足"), "deformation.shortness"),
        (("damage", "sunhuai", "损坏"), "material_loss.damage"),
    ]
    for keys, ckey in keyword_map:
        if any(k in lower for k in keys):
            return ckey

    return None


def build_class_map(names: dict[int, str]) -> tuple[dict[int, dict[str, str]], list[int]]:
    """按模型实际的 names 构建映射表。

    Returns:
        (class_map, unmapped_ids)
    """
    cmap: dict[int, dict[str, str]] = {}
    unmapped: list[int] = []
    for idx, name in names.items():
        key = _class_key_from_model_name(str(name))
        if key is None:
            unmapped.append(int(idx))
            cmap[int(idx)] = {
                "key": f"unmapped.{idx}",
                "name": str(name),
                "category": "未映射",
            }
        else:
            cn, cat = CLASS_META[key]
            cmap[int(idx)] = {"key": key, "name": cn, "category": cat}
    return cmap, unmapped


# 兼容旧接口：默认按数据集索引（仅在没有模型 names 时使用）
CLASS_MAP: dict[int, dict[str, str]] = {
    cid: {"key": key, "name": CLASS_META[key][0], "category": CLASS_META[key][1]}
    for cid, key in DATA_CLASS_KEYS.items()
}


class YoloUnavailable(RuntimeError):
    """YOLO 后端不可用（未安装 ultralytics 或权重文件缺失）。"""


# --------------------------------------------------------------------------
# 模型单例：避免每次调用都重新加载 40MB 权重
# --------------------------------------------------------------------------

_MODEL_CACHE: dict[str, Any] = {}


def _load_model(weights: str) -> Any:
    key = str(Path(weights).resolve())
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    # 必须在 import ultralytics 之前设置：ultralytics 在模块初始化时读取该变量，
    # 决定解码失败时是否联网 pip 安装依赖（如 pi-heif）。
    # 生产环境不应在推理路径上装包：既拖慢响应，又可能因无网络而失败，
    # 还会把"图像损坏"这类真实原因掩盖成"缺少某模块"。
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")

    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise YoloUnavailable(
            "未安装 ultralytics。请执行：pip install ultralytics"
        ) from exc

    if not Path(weights).exists():
        raise YoloUnavailable(f"权重文件不存在: {weights}")

    model = YOLO(weights)
    _MODEL_CACHE[key] = model
    return model


def model_info(weights: str | None = None) -> dict[str, Any]:
    """返回模型与类别信息，供 /health 与调试使用。"""
    w = weights or DEFAULT_WEIGHTS
    try:
        model = _load_model(w)
        names = getattr(model, "names", {}) or {}
        return {
            "available": True,
            "weights": str(Path(w).resolve()),
            "num_classes": len(names),
            "names": {int(k): v for k, v in names.items()},
            "task": getattr(model, "task", None),
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "weights": w, "error": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------------
# 几何换算
# --------------------------------------------------------------------------


def _polygon_area(pts: np.ndarray) -> float:
    """鞋带公式求多边形面积（像素²）。pts 形状 (4,2)。"""
    x = pts[:, 0]
    y = pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _polygon_edges(pts: np.ndarray) -> list[float]:
    """四边形的四条边长（像素）。"""
    return [float(np.linalg.norm(pts[(i + 1) % 4] - pts[i])) for i in range(4)]


def _polygon_center(pts: np.ndarray) -> tuple[float, float]:
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _bbox_of(pts: np.ndarray) -> list[int]:
    """四边形的最小轴对齐包围盒 [x1,y1,x2,y2]，供前端画图使用。"""
    return [
        int(np.floor(pts[:, 0].min())),
        int(np.floor(pts[:, 1].min())),
        int(np.ceil(pts[:, 0].max())),
        int(np.ceil(pts[:, 1].max())),
    ]


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------


def _make_instance(
    *,
    pts: np.ndarray,
    conf: float,
    class_id: int,
    class_map: dict[int, dict[str, str]],
    names: dict[int, str],
    gsd: float,
    box_source: str,
) -> dict[str, Any]:
    """把一组四边形顶点整理成统一的缺陷实例结构。

    两种模型（obb / detect）共用此函数，保证下游拿到的字段完全一致。
    """
    area_px = _polygon_area(pts)
    edges = _polygon_edges(pts)
    max_len_px = max(edges)
    min_len_px = min(edges)
    cx, cy = _polygon_center(pts)

    mapped = class_map.get(class_id)
    raw_name = names.get(int(class_id), f"class_{class_id}")

    return {
        "instance_id": "",  # 排序后统一编号
        "class_id": int(class_id),
        "class_name": mapped["name"] if mapped else raw_name,
        "defect_type": mapped["key"] if mapped else f"unmapped.{class_id}",
        "category": mapped["category"] if mapped else "未映射",
        "mapped": bool(mapped and not str(mapped["key"]).startswith("unmapped.")),
        "confidence": round(float(conf), 4),
        "polygon_px": [[round(float(x), 2), round(float(y), 2)] for x, y in pts],
        "center_px": [round(cx, 1), round(cy, 1)],
        "bbox_px": _bbox_of(pts),
        "area_px": round(area_px, 2),
        "max_length_px": round(max_len_px, 2),
        "min_edge_px": round(min_len_px, 2),
        "area_mm2": round(area_px * gsd**2, 4),
        "max_length_mm": round(max_len_px * gsd, 4),
        "min_edge_mm": round(min_len_px * gsd, 4),
        "aspect_ratio": round(max_len_px / min_len_px, 2) if min_len_px > 0 else None,
        "box_source": box_source,          # obb | axis_aligned
        "area_is_upper_bound": box_source == "axis_aligned",
    }


def run_yolo(
    image_path: str,
    gsd_mm_per_px: float,
    *,
    weights: str | None = None,
    conf: float | None = None,
    imgsz: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """执行 OBB 推理并换算为质检指标。

    返回结构与像素后端保持一致（`annotate_defects` 的契约），额外增加
    OBB 特有字段：`polygon_px`、`center_px`、`class_id`、`class_name`。
    """
    w = weights or DEFAULT_WEIGHTS
    c = DEFAULT_CONF if conf is None else conf
    s = DEFAULT_IMGSZ if imgsz is None else imgsz
    dev = DEFAULT_DEVICE if device is None else device

    model = _load_model(w)
    names = getattr(model, "names", {}) or {}
    task = str(getattr(model, "task", "") or "").lower()

    # 类别映射按**模型的实际 names** 构建（不按索引硬映射，
    # 因为实测权重的类别数与顺序可能与数据集不一致）
    class_map, unmapped_ids = build_class_map({int(k): v for k, v in names.items()})

    predict_kwargs: dict[str, Any] = {
        "source": image_path,
        "conf": c,
        "imgsz": s,
        "verbose": False,
    }
    if dev:
        predict_kwargs["device"] = dev

    results = model.predict(**predict_kwargs)
    if not results:
        raise RuntimeError(f"YOLO 未返回结果: {image_path}")

    res = results[0]
    h, w_img = res.orig_shape[:2]

    instances: list[dict[str, Any]] = []
    out_task = task or "unknown"

    # ---- 分支 1：OBB 模型（旋转框）----
    obb = getattr(res, "obb", None)
    if obb is not None and len(obb) > 0:
        out_task = "obb"
        # xyxyxyxy: (N,4,2)；官方推荐的几何无歧义表示
        polys = obb.xyxyxyxy.cpu().numpy()
        confs = obb.conf.cpu().numpy()
        clss = obb.cls.cpu().numpy().astype(int)

        for i in range(len(polys)):
            pts = polys[i].astype(float)
            instances.append(
                _make_instance(
                    pts=pts, conf=float(confs[i]), class_id=int(clss[i]),
                    class_map=class_map, names=names, gsd=gsd_mm_per_px,
                    box_source="obb",
                )
            )

    # ---- 分支 2：普通检测模型（轴对齐框）----
    # 实测权重 task=detect，只能给矩形框。这里把矩形框转成四边形，
    # 下游（面积/长边/前端画图）逻辑无需分叉。
    else:
        boxes = getattr(res, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            out_task = "detect"
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = [float(v) for v in xyxy[i]]
                pts = np.array(
                    [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=float
                )
                inst = _make_instance(
                    pts=pts, conf=float(confs[i]), class_id=int(clss[i]),
                    class_map=class_map, names=names, gsd=gsd_mm_per_px,
                    box_source="axis_aligned",
                )
                # 轴对齐框的面积是上界，对细长缺陷高估明显，显式标注
                inst["area_is_upper_bound"] = True
                instances.append(inst)

    # 面积降序 + 统一编号
    instances.sort(key=lambda d: d["area_px"], reverse=True)
    for idx, inst in enumerate(instances, start=1):
        inst["instance_id"] = f"D{idx:02d}"

    # 面积降序 + 统一编号
    instances.sort(key=lambda d: d["area_px"], reverse=True)
    for idx, inst in enumerate(instances, start=1):
        inst["instance_id"] = f"D{idx:02d}"

    total_area_mm2 = sum(i["area_mm2"] for i in instances)
    total_area_px = sum(i["area_px"] for i in instances)
    mean_conf = (
        sum(i["confidence"] for i in instances) / len(instances) if instances else 0.0
    )
    image_area_px = float(w_img * h)

    # 相对化指标：**不依赖 GSD**，在未标定相机时仍可用于判定与对比。
    # 工业上比绝对 mm² 更通用——不同尺寸的产品可共用一套比例阈值。
    for inst in instances:
        inst["area_ratio_of_image"] = round(inst["area_px"] / image_area_px, 6)
        inst["len_ratio_of_diag"] = round(
            max(
                float(
                    np.linalg.norm(
                        np.array(inst["polygon_px"][(k + 1) % 4])
                        - np.array(inst["polygon_px"][k])
                    )
                )
                for k in range(4)
            ) / float(np.hypot(w_img, h)),
            6,
        )

    # ---- 相对化严重度（不依赖 GSD，解决未标定相机的场景）----
    from qcagent.core.severity import apply_relative_grading

    partial: dict[str, Any] = {
        "image_path": str(Path(image_path).resolve()),
        "image_size_px": [w_img, h],
        "defects": instances,
    }
    apply_relative_grading(partial)
    sample_region = partial.get("sample_region", {})
    max_sev_relative = partial.get("max_severity_relative", "none")

    return {
        "image_path": str(Path(image_path).resolve()),
        "image_size_px": [w_img, h],
        "gsd_mm_per_px": gsd_mm_per_px,
        "defect_count": len(instances),
        "total_area_mm2": round(total_area_mm2, 4),
        "total_area_px": round(total_area_px, 2),
        "total_area_ratio_of_image": round(total_area_px / image_area_px, 6),
        "mean_confidence": round(mean_conf, 4),
        "defects": instances,
        # ---- 相对化判定（推荐口径，不依赖 GSD）----
        "sample_region": sample_region,
        "max_severity_relative": max_sev_relative,
        "grading_mode": "relative",
        "relative_rules": partial.get("relative_rules"),
        # ---- 溯源字段 ----
        "backend": f"yolo_{out_task}",           # yolo_obb | yolo_detect
        "model_task": out_task,
        "model_nc": len(names),
        "model_weights": str(Path(w).resolve()),
        "infer_params": {"conf": c, "imgsz": s, "device": dev or "auto"},
        # 面积来源：旋转矩形多边形（精确）/ 轴对齐框（上界，高估细长缺陷）
        "area_source": (
            "obb_polygon" if out_task == "obb" else "axis_aligned_bbox"
        ),
        "class_names": {int(k): v for k, v in names.items()},
        "class_map": {int(k): v["name"] for k, v in class_map.items()},
        "unmapped_classes": sorted({i["class_id"] for i in instances if not i["mapped"]}),
        "model_unmapped_ids": sorted(unmapped_ids),
        # GSD 是否可信：未标定时 physical 字段仅供参考，应以 *_ratio_* 字段为准
        "physical_metrics_reliable": bool(os.getenv("QC_GSD_CALIBRATED", "") == "1"),
    }


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    print("=== 模型信息 ===")
    info = model_info()
    for k, v in info.items():
        print(f"  {k}: {v}")

    if not info.get("available"):
        print("\nYOLO 不可用，请先执行 pip install ultralytics")
        sys.exit(0)

    # 用验证集第一张图跑一次推断
    import glob

    files = sorted(glob.glob(r"E:\yolodata\images\val\*.bmp"))
    if not files:
        print("\n未找到验证集图像")
        sys.exit(0)

    gsd = float(os.getenv("QC_GSD", "0.027"))
    print(f"\n=== 推断 {Path(files[0]).name}，GSD={gsd} mm/px ===")
    r = run_yolo(files[0], gsd)
    print(f"  尺寸 {r['image_size_px']}  缺陷 {r['defect_count']} 个  "
          f"总面积 {r['total_area_mm2']} mm²  平均置信度 {r['mean_confidence']}")
    for d in r["defects"][:8]:
        print(f"    {d['instance_id']} {d['class_name']:8s} "
              f"conf={d['confidence']:.3f} 面积={d['area_mm2']:.3f}mm² "
              f"长边={d['max_length_mm']:.2f}mm 长宽比={d['aspect_ratio']}")
