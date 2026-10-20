# -*- coding: utf-8 -*-
"""质检视觉工具集（第二阶段）

设计原则（面试可讲）：
  * **接口与实现分离**：每个工具都是一个普通 Python 函数，函数体内部调用
    `annotate_defects()`。当前是纯像素分析的参考实现，未来替换为训练好的
    分割模型（ONNX / PyTorch）时，**只需重写 annotate_defects，工具契约不变**，
    Agent 侧零改动。这是"视觉模型即工具"范式的落地方式。
  * **单位在工具内部换算**：像素→毫米的换算（GSD）由工具完成，返回带单位的
    物理量。让 LLM 自己算单位是可靠性事故的高发区。
  * **每个结果都带置信度**：后续"低置信转人工复核"依赖这个字段。
  * **权限显式声明**：FunctionTool 默认 PermissionBehavior.ASK，会让 API 调用
    卡在"等待用户确认"。质检工具是只读的，显式声明 ALLOW + is_read_only。

当前实现的定位：**参考实现 / 基线**。它真的读像素、真的算连通域，不是假数据，
但它的分割能力远弱于训练好的模型 —— 这一点在简历与面试中必须说清楚。
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------

# 缺陷暗于背景的灰度阈值（合成图背景约 150±6，缺陷填充 35~85）
DARK_THRESHOLD = 100

# 过滤噪点的最小连通域面积（像素）
MIN_AREA_PX = 50

# 像素 → 物理尺寸：默认 0.05 mm/px（相当于 200dpi 级工业相机）
DEFAULT_GSD_MM_PER_PX = 0.05

# 严重度判定阈值（依据缺陷面积与最大尺寸）
SEVERITY_RULES = (
    ("critical", 50.0, 20.0),  # 面积 > 50 mm² 或 最长边 > 20 mm
    ("major", 20.0, 8.0),      # 面积 > 20 mm² 或 最长边 > 8 mm
    ("minor", 0.0, 0.0),       # 其余
)


# --------------------------------------------------------------------------
# 核心实现：可替换的分割后端
# --------------------------------------------------------------------------


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """两遍扫描连通域标记（4 邻域），并查集实现。

    返回 (labels, count)。labels 中 0 表示背景。
    这里手写是为了不引入 opencv/scipy 依赖；换真模型后此函数可整体删除。
    """
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    parent: dict[int, int] = {}
    next_id = 0

    def find(x: int) -> int:
        root = x
        while parent.get(root, root) != root:
            root = parent.get(root, root)
        while parent.get(x, x) != x:  # 路径压缩
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # 第一遍：临时标记 + 记录等价关系
    for y in range(h):
        for x in range(w):
            if not mask[y, x]:
                continue
            left = labels[y, x - 1] if x > 0 else 0
            up = labels[y - 1, x] if y > 0 else 0
            if left == 0 and up == 0:
                next_id += 1
                labels[y, x] = next_id
                parent[next_id] = next_id
            elif left != 0 and up != 0:
                labels[y, x] = min(left, up)
                union(left, up)
            else:
                labels[y, x] = left or up

    # 第二遍：统一等价类
    remap: dict[int, int] = {}
    out = np.zeros_like(labels)
    for y in range(h):
        for x in range(w):
            lab = labels[y, x]
            if lab == 0:
                continue
            root = find(lab)
            if root not in remap:
                remap[root] = len(remap) + 1
            out[y, x] = remap[root]

    return out, len(remap)


def _severity_of(area_mm2: float, max_len_mm: float) -> str:
    for name, area_th, len_th in SEVERITY_RULES:
        if area_mm2 > area_th or max_len_mm > len_th:
            return name
    return "minor"


def _pixel_backend(
    image_path: str,
    gsd_mm_per_px: float = DEFAULT_GSD_MM_PER_PX,
) -> dict[str, Any]:
    """像素连通域后端（参考实现 / 降级用）。

    不做语义分类，只按"暗于背景"切出连通域，因此 `defects` 里的实例
    没有 defect_type 字段。YOLO 不可用时作为兜底，保证系统仍能跑通。
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"图像不存在: {image_path}")

    with Image.open(path) as im:
        gray = np.asarray(im.convert("L"))
    h, w = gray.shape

    mask = gray < DARK_THRESHOLD
    labels, count = _label_components(mask)

    instances: list[dict[str, Any]] = []
    for lab in range(1, count + 1):
        ys, xs = np.where(labels == lab)
        area_px = int(ys.size)
        if area_px < MIN_AREA_PX:
            continue

        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        width_px, height_px = x2 - x1 + 1, y2 - y1 + 1

        # 最长边：取外接框长边与等效直径的较大值（细长裂缝用外接框更准）
        max_len_px = max(width_px, height_px, 2.0 * math.sqrt(area_px / math.pi))

        area_mm2 = round(area_px * gsd_mm_per_px**2, 3)
        max_len_mm = round(max_len_px * gsd_mm_per_px, 3)

        # 置信度参考值：对比度越高、形状越紧凑，判读越可信
        contrast = float(gray[mask].mean()) if mask.any() else 0.0
        background = float(gray[~mask].mean()) if (~mask).any() else 255.0
        contrast_ratio = max(0.0, min(1.0, (background - contrast) / background))
        fill_ratio = area_px / float(width_px * height_px)
        confidence = round(0.5 * contrast_ratio + 0.5 * fill_ratio, 3)

        instances.append(
            {
                "instance_id": f"D{len(instances) + 1:02d}",
                "bbox_px": [x1, y1, x2, y2],
                "area_px": area_px,
                "area_mm2": area_mm2,
                "max_length_mm": max_len_mm,
                "severity": _severity_of(area_mm2, max_len_mm),
                "confidence": confidence,
            }
        )

    instances.sort(key=lambda d: d["area_px"], reverse=True)
    for idx, inst in enumerate(instances, start=1):
        inst["instance_id"] = f"D{idx:02d}"

    mean_conf = (
        round(sum(i["confidence"] for i in instances) / len(instances), 3)
        if instances
        else 0.0
    )

    return {
        "image_path": str(path.resolve()),
        "image_size_px": [w, h],
        "gsd_mm_per_px": gsd_mm_per_px,
        "defect_count": len(instances),
        "total_area_mm2": round(sum(i["area_mm2"] for i in instances), 3),
        "max_severity": _max_severity(instances),
        "mean_confidence": mean_conf,
        "defects": instances,
        "backend": "reference_pixel_analysis_v1",
        "area_source": "connected_component",
    }


# --------------------------------------------------------------------------
# 后端调度：YOLO 优先，不可用时降级像素后端
# --------------------------------------------------------------------------

# "auto" = 优先 YOLO，不可用则降级；"yolo" = 强制 YOLO（失败即报错）；"pixel" = 强制像素
BACKEND_MODE = os.getenv("QC_BACKEND", "auto").strip().lower()


def annotate_defects(
    image_path: str,
    gsd_mm_per_px: float = DEFAULT_GSD_MM_PER_PX,
    *,
    backend: str | None = None,
    conf: float | None = None,
) -> dict[str, Any]:
    """检测后端统一入口（★ 上层唯一的替换点）。

    返回结构对两种后端保持一致，因此 Agent、工具、报告 schema、评测器
    都无需感知后端差异。差异只体现在 `backend` / `area_source` 两个溯源字段。

    Args:
        image_path: 图像路径
        gsd_mm_per_px: 每像素代表的毫米数（YOLO 后端用于像素→物理量换算）
        backend: 覆盖全局配置（auto / yolo / pixel），便于评测时对比
        conf: 仅 YOLO 后端有效，置信度阈值

    Raises:
        FileNotFoundError: 图像不存在（两种后端都先做此检查）
        YoloUnavailable: 仅当 backend="yolo" 强制指定且 YOLO 不可用时抛出
    """
    mode = (backend or BACKEND_MODE or "auto").strip().lower()
    path = Path(image_path)

    # 图像存在性检查与后端无关，提前做 —— 让错误语义统一（不因后端不同而变）
    if not path.exists():
        raise FileNotFoundError(f"图像不存在: {image_path}")

    # 可解码性预检：必须放在调用 YOLO 之前。
    # 原因：ultralytics 解码失败时会尝试自动安装解码依赖（如 pi-heif），
    # 既拖慢响应，又把真实原因（文件损坏）掩盖成 ModuleNotFoundError，
    # 导致 Agent 向用户报告"缺少某模块"，而实际是图像坏了。
    _ensure_decodable(path)

    def _do_pixel() -> dict[str, Any]:
        return _pixel_backend(image_path, gsd_mm_per_px)

    if mode == "pixel":
        return _do_pixel()

    if mode in ("auto", "yolo"):
        try:
            from qc_yolo import YoloUnavailable, run_yolo

            try:
                result = run_yolo(image_path, gsd_mm_per_px, conf=conf)
            except YoloUnavailable:
                if mode == "yolo":
                    raise
                result = None
            if result is not None:
                # 补上统一字段：severity 由面积/长边规则统一判定
                for inst in result["defects"]:
                    inst["severity"] = _severity_of(
                        inst["area_mm2"], inst["max_length_mm"]
                    )
                result["max_severity"] = _max_severity(result["defects"])
                return result
        except ImportError:
            # qc_yolo 模块本身缺失（旧部署）——降级
            if mode == "yolo":
                raise
        except Exception as exc:  # noqa: BLE001 —— 推理异常不应让整个服务不可用
            if mode == "yolo":
                raise
            print(f"[qc_tools] YOLO 推理失败，降级像素后端: {type(exc).__name__}: {exc}")

    if mode not in ("auto", "yolo", "pixel"):
        raise ValueError(f"未知 backend: {mode}（可选 auto / yolo / pixel）")

    result = _do_pixel()
    result["degraded_from"] = "yolo" if mode in ("auto", "yolo") else None
    return result


def _ensure_decodable(path: Path) -> None:
    """确认图像可被解码，否则抛出语义明确的异常。

    只读文件头（PIL 的 open 是惰性的），代价远小于整图解码，但足以
    识别截断、格式损坏、伪装扩展名等常见问题。

    这条检查是「静默放行」防线的一部分：损坏的图像必须在进入任何检测后端
    之前就失败，且失败原因必须准确，否则 Agent 会把"图像坏了"误报成别的问题。
    """
    try:
        with Image.open(path) as im:
            _ = im.size          # 触发头部解析
            im.format            # 记录格式，供后续可能的诊断使用
    except Exception as exc:  # noqa: BLE001
        raise ValueError(
            f"图像无法解码（文件可能损坏或格式不受支持）: {path.name}"
        ) from exc


def _max_severity(instances: list[dict[str, Any]]) -> str:
    order = {"critical": 3, "major": 2, "minor": 1}
    if not instances:
        return "none"
    return max((i["severity"] for i in instances), key=lambda s: order.get(s, 0))


# --------------------------------------------------------------------------
# 工具函数：暴露给 Agent 的接口
# --------------------------------------------------------------------------


def segment_defects(image_path: str, gsd_mm_per_px: float = DEFAULT_GSD_MM_PER_PX) -> str:
    """对一张工业质检图像执行缺陷分割，返回缺陷清单与尺寸。

    当用户提供图像路径并要求检测、识别、分析缺陷时调用本工具。

    **本工具以固定前缀标明执行状态，请据前缀判断结果是否可用**：
      * 以「检测失败」开头 —— 图像无法读取，**没有产生任何测量数据**，
        此时必须如实告知用户无法判定，严禁给出合格/不合格结论；
      * 以「检测完成」开头 —— 数据有效，可据此判定。

    **判定的两套口径，务必看清哪套可信**：
      * 相对化等级（推荐）—— 基于"缺陷占样品面积/边长的比例"，与相机标定无关，
        始终可信。字段名 `severity_relative`。
      * 物理等级 —— 基于 mm² 与 mm，**只有在"物理量可信=是"时才可用**。
        若本工具提示"物理量不可信"，则面积/尺寸中的 mm 单位是按假设的
        GSD 换算得来，**不得用于判定严重等级**，只能用于横向比较。

    Args:
        image_path: 待检测图像的本地文件路径，例如 D:/data/sample.png
        gsd_mm_per_px: 相机分辨率，即每像素代表多少毫米，默认 0.05
    """
    try:
        result = annotate_defects(image_path, gsd_mm_per_px)
    except FileNotFoundError:
        return (
            f"检测失败：图像不存在，未产生任何测量数据 —— {image_path}。"
            "请确认路径是否正确，在拿到有效图像前不得给出质检结论。"
        )
    except Exception as exc:  # noqa: BLE001
        return (
            f"检测失败：图像无法读取（{type(exc).__name__}: {exc}），未产生任何测量数据。"
            "在拿到有效图像前不得给出质检结论。"
        )

    defects = result["defects"]
    physical_ok = bool(result.get("physical_metrics_reliable", False))
    max_rel = result.get("max_severity_relative", "none")
    region = result.get("sample_region") or {}

    if not defects:
        return (
            f"检测完成：图像 {result['image_size_px'][0]}x{result['image_size_px'][1]} 像素，"
            f"未检出任何缺陷（后端 {result.get('backend')}）。判定为合格。"
        )

    lines: list[str] = []
    lines.append(
        f"检测完成。后端={result.get('backend')}，"
        f"共发现 {result['defect_count']} 处缺陷，"
        f"平均置信度 {result['mean_confidence']}。"
    )

    # ---- 口径声明：让模型清楚该用哪套等级 ----
    lines.append("")
    lines.append("【判定口径】")
    if physical_ok:
        lines.append(
            f"  · 物理量可信：是（GSD={result['gsd_mm_per_px']} mm/px 已标定）"
        )
        lines.append(
            f"  · 物理等级={result.get('max_severity')}；相对化等级={max_rel}"
        )
        lines.append("  · 两者均可用，如有冲突以物理等级为准")
    else:
        lines.append("  · 物理量可信：**否**（相机 GSD 未标定，mm² 与 mm 仅为估算）")
        lines.append(
            f"  · **必须使用相对化等级判定：{max_rel}**（基于缺陷占样品的比例，与标定无关）"
        )
        lines.append("  · 严禁依据下面的 mm² / mm 数值判定严重等级")

    if region:
        lines.append(
            f"  · 样品区域：{region.get('width', 0):.0f}×{region.get('height', 0):.0f} px"
            f"（来源 {region.get('source')}）"
        )

    # ---- 缺陷清单 ----
    lines.append("")
    lines.append("【缺陷清单】")
    for d in defects:
        x1, y1, x2, y2 = d["bbox_px"]
        ar = d.get("area_ratio_sample")
        lr = d.get("len_ratio_sample")
        parts = [
            f"  - {d['instance_id']} {d.get('class_name', '未分类')}"
            f"（类别键 {d.get('defect_type', '?')}）",
            f"置信度 {d['confidence']}",
            f"位置 ({x1},{y1})-({x2},{y2})",
        ]
        if ar is not None and lr is not None:
            parts.append(
                f"占样品面积 {ar:.3%}、占样品边长 {lr:.3%}"
                f" → 相对化等级 {d.get('severity_relative', '?')}"
            )
        parts.append(
            f"像素面积 {d.get('area_px')} px²、最长边 {d.get('max_length_px')} px"
        )
        if physical_ok:
            parts.append(
                f"物理面积 {d['area_mm2']} mm²、最长边 {d['max_length_mm']} mm"
                f" → 物理等级 {d['severity']}"
            )
        lines.append("；".join(parts))

    lines.append("")
    lines.append(
        f"【汇总】相对化等级={max_rel}"
        + (f"，物理等级={result.get('max_severity')}（可信）" if physical_ok
           else "（物理等级不可信，勿使用）")
    )
    return "\n".join(lines)


def get_image_info(image_path: str) -> str:
    """读取图像基本信息：尺寸、格式、灰度统计。

    当需要确认图像是否可用（例如判断是否过暗、是否读取成功）时调用。

    Args:
        image_path: 图像文件的本地路径
    """
    try:
        with Image.open(image_path) as im:
            gray = np.asarray(im.convert("L"))
            return (
                f"图像信息：{im.format} 格式，{im.size[0]}x{im.size[1]} 像素，"
                f"色彩模式 {im.mode}；灰度均值 {gray.mean():.1f}，"
                f"最小值 {gray.min()}，最大值 {gray.max()}。"
            )
    except FileNotFoundError:
        return f"读取失败：图像不存在 {image_path}"
    except Exception as exc:  # noqa: BLE001
        return f"读取失败：{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------
# 样例图生成（便于在没有真实数据时立即验证链路）
# --------------------------------------------------------------------------


def make_sample_image(
    out_path: str,
    defect_kind: str = "crack",
    size: tuple[int, int] = (640, 480),
) -> str:
    """生成一张合成的蜂窝催化剂端面图，用于在没有真实数据时测试链路。

    defect_kind 可选：none（无缺陷）/ crack（细长裂缝）/ hole（孔道缺失）/ deform（形变带）
    """
    width, height = size
    img = Image.new("L", size, color=150)
    draw = ImageDraw.Draw(img)

    # 蜂窝孔道网格
    step = 40
    radius = 14
    for cy in range(step // 2, height, step):
        for cx in range(step // 2, width, step):
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                outline=105,
                width=2,
            )

    if defect_kind == "crack":
        draw.line([(180, 120), (420, 300)], fill=40, width=10)
    elif defect_kind == "hole":
        draw.rectangle([260, 180, 330, 250], fill=35)
        draw.rectangle([400, 300, 430, 330], fill=45)
    elif defect_kind == "deform":
        draw.polygon([(150, 380), (480, 360), (490, 405), (160, 425)], fill=60)

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return str(path.resolve())


if __name__ == "__main__":
    import sys as _sys

    out_dir = Path(__file__).resolve().parent / "data" / "samples"

    # 命令行用法：
    #   python qc_tools.py            -> 生成 4 张样例图并自测分割
    #   python qc_tools.py gen crack  -> 只生成一张，打印其绝对路径
    if len(_sys.argv) >= 3 and _sys.argv[1] == "gen":
        kind = _sys.argv[2]
        p = make_sample_image(str(out_dir / f"sample_{kind}.png"), defect_kind=kind)
        print(p)
        _sys.exit(0)

    for kind in ("none", "crack", "hole", "deform"):
        p = make_sample_image(str(out_dir / f"sample_{kind}.png"), defect_kind=kind)
        print(f"\n===== {kind}: {p} =====")
        print(segment_defects(p))
