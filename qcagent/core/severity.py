# -*- coding: utf-8 -*-
"""相对化严重度判定（不依赖 GSD，解决"不知道样品物理尺寸"的阻塞）

## 为什么需要这个

绝对判定规则依赖 GSD（像素→毫米）：

    critical: 面积 > 50 mm² 或 最大尺寸 > 20 mm

但要得到 GSD 必须知道样品的**物理尺寸**。在你的场景里这个值未知，
于是整条"等级判定"链路都卡住了。

## 相对化思路

改用**缺陷尺寸相对样品尺寸**的比例来判定：

    critical: 缺陷面积 / 样品面积 > 6%  或  缺陷长边 / 样品边长 > 25%
    major:    缺陷面积 / 样品面积 > 2%  或  缺陷长边 / 样品边长 > 12%
    minor:    其余

这样做的好处：

  1. **完全绕开 GSD**——比值与像素↔毫米的换算无关
  2. **跨产品尺寸通用**——换一种规格的产品不必重设阈值
  3. **可解释**——"占样品面积 6%" 比 "50 平方毫米" 更容易被质量人员接受

代价：不能给出 mm² 绝对值。若将来标定了 GSD，两种口径可并存。

## 样品区域的确定

两种途径，按可靠性排序：
  1) 显式传入 `sample_bbox_px`（来自 qc_gsd_calib 的自动检测）
  2) 由缺陷框外接包围盒 + 余量估计（粗略，但无需额外信息）

默认走 2)，并可缓存 1) 的结果。
"""

from __future__ import annotations

from qcagent import settings  # noqa: E402  统一路径常量

import json
import os
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------
# 相对化阈值（占样品尺寸的比例）
# --------------------------------------------------------------------------

RELATIVE_RULES = (
    # (等级, 面积占样品比例阈值, 长边占样品边长比例阈值)
    ("critical", 0.060, 0.250),
    ("major", 0.020, 0.120),
    ("minor", 0.0, 0.0),
)

# 严重程度排序（用于取最大值）
_ORDER = {"critical": 3, "major": 2, "minor": 1, "none": 0}


def grade_by_ratio(area_ratio: float, len_ratio: float) -> str:
    """按相对比例判定等级。

    Args:
        area_ratio: 缺陷面积 / 样品面积
        len_ratio:  缺陷长边 / 样品边长
    """
    for name, area_th, len_th in RELATIVE_RULES:
        if area_ratio > area_th or len_ratio > len_th:
            return name
    return "minor"


def grade_absolute(area_mm2: float, max_len_mm: float) -> str:
    """按绝对物理量判定（原规则，需 GSD）。保留以便标定后使用。"""
    rules = (
        ("critical", 50.0, 20.0),
        ("major", 20.0, 8.0),
        ("minor", 0.0, 0.0),
    )
    for name, area_th, len_th in rules:
        if area_mm2 > area_th or max_len_mm > len_th:
            return name
    return "minor"


# --------------------------------------------------------------------------
# 样品区域估计
# --------------------------------------------------------------------------

_CACHE_PATH = settings.SAMPLE_REGIONS_PATH
_cache: dict[str, dict[str, Any]] | None = None


def _load_cache() -> dict[str, dict[str, Any]]:
    global _cache
    if _cache is None:
        try:
            _cache = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            _cache = {}
    return _cache


def save_sample_region(image_path: str, bbox: list[float]) -> None:
    """登记某张图的样品外接框（由标定工具或人工确认后写入）。"""
    cache = _load_cache()
    cache[str(Path(image_path).resolve())] = {"bbox": bbox}
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def estimate_sample_region(
    image_size_px: tuple[int, int],
    defects: list[dict[str, Any]],
    image_path: str | None = None,
) -> dict[str, Any]:
    """确定样品在图中的区域，返回 {bbox, width, height, source}。

    优先级：
      1) 缓存里已登记的精确框
      2) 由缺陷框并集外扩估计（粗略）
    """
    if image_path:
        cached = _load_cache().get(str(Path(image_path).resolve()))
        if cached and cached.get("bbox"):
            b = cached["bbox"]
            return {
                "bbox": b,
                "width": b[2] - b[0],
                "height": b[3] - b[1],
                "source": "cached_calibration",
            }

    W, H = image_size_px
    if not defects:
        # 无缺陷时无法从缺陷反推，退化为"整幅图"（此时任何缺陷比例都趋近 0）
        return {"bbox": [0, 0, W, H], "width": W, "height": H, "source": "whole_image"}

    xs1 = min(d["bbox_px"][0] for d in defects)
    ys1 = min(d["bbox_px"][1] for d in defects)
    xs2 = max(d["bbox_px"][2] for d in defects)
    ys2 = max(d["bbox_px"][3] for d in defects)
    w = max(1.0, xs2 - xs1)
    h = max(1.0, ys2 - ys1)

    # 缺陷通常只占样品局部，按经验外扩 3 倍（含余量），并限制在图像范围内
    pad_x = w * 1.0
    pad_y = h * 1.0
    bx1 = max(0.0, xs1 - pad_x)
    by1 = max(0.0, ys1 - pad_y)
    bx2 = min(float(W), xs2 + pad_x)
    by2 = min(float(H), ys2 + pad_y)

    return {
        "bbox": [bx1, by1, bx2, by2],
        "width": bx2 - bx1,
        "height": by2 - by1,
        "source": "estimated_from_defects",
        "note": "由缺陷并集外扩估算，精度有限；建议用 qc_gsd_calib 标定后调用 save_sample_region 固化",
    }


# --------------------------------------------------------------------------
# 对 run_yolo 的结果做相对化升级
# --------------------------------------------------------------------------


def apply_relative_grading(result: dict[str, Any]) -> dict[str, Any]:
    """给 run_yolo 的返回结果补充相对化指标与等级。

    原地修改并返回 result，新增字段：
      · sample_region            样品区域与来源
      · defects[].area_ratio_sample   缺陷面积 / 样品面积
      · defects[].len_ratio_sample    缺陷长边 / 样品边长
      · defects[].severity_relative   相对化等级
      · max_severity_relative
      · grading_mode             "relative" | "absolute"
    """
    defects = result.get("defects") or []
    W, H = result.get("image_size_px") or [1, 1]

    region = estimate_sample_region(
        (int(W), int(H)), defects, image_path=result.get("image_path")
    )
    result["sample_region"] = region

    sample_area = region["width"] * region["height"]
    sample_side = max(region["width"], region["height"])

    severities: list[str] = []
    for d in defects:
        area_ratio = d["area_px"] / sample_area if sample_area > 0 else 0.0
        len_ratio = d["max_length_px"] / sample_side if sample_side > 0 else 0.0
        d["area_ratio_sample"] = round(area_ratio, 6)
        d["len_ratio_sample"] = round(len_ratio, 6)
        sev = grade_by_ratio(area_ratio, len_ratio)
        d["severity_relative"] = sev
        severities.append(sev)

    result["max_severity_relative"] = (
        max(severities, key=lambda s: _ORDER.get(s, 0)) if severities else "none"
    )
    result["grading_mode"] = "relative"
    result["relative_rules"] = [
        {"grade": g, "area_ratio_gt": a, "len_ratio_gt": l} for g, a, l in RELATIVE_RULES
    ]
    return result


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    print("=== 相对化阈值 ===")
    for g, a, l in RELATIVE_RULES:
        print(f"  {g:8s} 面积占样品 > {a:.1%}  或  长边占样品边长 > {l:.1%}")

    print("\n=== 等级判定示例 ===")
    cases = [
        (0.001, 0.02, "极小缺陷"),
        (0.010, 0.08, "小缺陷"),
        (0.025, 0.15, "中缺陷"),
        (0.070, 0.30, "大缺陷"),
        (0.200, 0.50, "严重缺陷"),
    ]
    for ar, lr, label in cases:
        print(f"  {label:8s} 面积占比 {ar:6.1%}  长边占比 {lr:6.1%} → "
              f"{grade_by_ratio(ar, lr)}")

    print("\n=== 样品区域估计示例（3 个缺陷框散落在局部）===")
    demo_defects = [
        {"bbox_px": [2000, 1500, 2100, 1560]},
        {"bbox_px": [2200, 1600, 2280, 1650]},
        {"bbox_px": [2050, 1700, 2120, 1760]},
    ]
    r = estimate_sample_region((5472, 3648), demo_defects)
    print(f"  样品区域 {[round(v) for v in r['bbox']]}  来源 {r['source']}")
    print(f"  样品尺寸 {r['width']:.0f} × {r['height']:.0f} px")
