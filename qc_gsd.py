# -*- coding: utf-8 -*-
"""GSD（Ground Sample Distance）标定工具

## 为什么需要这个

YOLO 输出的是**像素**尺度（框的四角点、边长、面积）。
但质检判定规则是**物理**尺度（mm²、mm）：

    critical: 面积 > 50 mm² 或 最大尺寸 > 20 mm
    major:    面积 > 20 mm² 或 最大尺寸 > 8 mm

两者之间必须由 GSD（每像素代表多少毫米）桥接：

    物理面积(mm²) = 像素面积(px²) × GSD²
    物理长度(mm)  = 像素长度(px)  × GSD

## 两种产地

1. **相机标定**：已知工作距离与镜头焦距，或拍过标定块 → GSD 直接可得
2. **实物反推**：知道被摄样品的物理尺寸（蜂窝催化剂端面边长）→
       GSD = 样品物理边长(mm) ÷ 样品在图中的像素边长(px)

本模块提供第 2 种的估算函数，因为工业现场往往"知道产品多大"但"不知道 GSD"。
"""

from __future__ import annotations

from typing import Any

from PIL import Image


def gsd_from_physical(
    image_path: str,
    physical_width_mm: float,
    *,
    aspect_ratio: float | None = None,
) -> dict[str, Any]:
    """由样品的物理边长反推 GSD。

    Args:
        image_path: 样品图像路径
        physical_width_mm: 样品在图像中的实际横向尺寸（mm）
        aspect_ratio: 样品长宽比。若给出，则纵向 GSD 按此推算；
                      不给出则假设图像横纵比例一致（方形像素）

    Returns:
        含 gsd_mm_per_px、各项中间量的 dict
    """
    with Image.open(image_path) as im:
        width_px, height_px = im.size

    if physical_width_mm <= 0:
        raise ValueError(f"physical_width_mm 必须为正数，收到 {physical_width_mm}")

    gsd_x = physical_width_mm / width_px
    gsd_y = gsd_x / aspect_ratio if aspect_ratio else gsd_x

    return {
        "image_size_px": [width_px, height_px],
        "physical_width_mm": physical_width_mm,
        "gsd_mm_per_px": round((gsd_x + gsd_y) / 2, 6),
        "gsd_x": round(gsd_x, 6),
        "gsd_y": round(gsd_y, 6),
        "physical_height_mm": round(height_px * gsd_y, 3),
        "method": "physical_width / image_width_px",
        "note": (
            "该值假定样品铺满图像宽度。若样品在图中只占一部分，"
            "需把 physical_width_mm 换成'样品实际占据的那段宽度对应的物理尺寸'。"
        ),
    }


def sanity_check(gsd: float, image_size_px: tuple[int, int]) -> dict[str, Any]:
    """对 GSD 做合理性检查，避免标定错误导致等级全错。

    工业相机的常见 GSD 区间约 0.005~1.0 mm/px（对应 5μm~1mm 分辨率）。
    超出该区间大概率是标定错了，提前预警比事后发现等级全错要好。
    """
    width_px, height_px = image_size_px
    warnings: list[str] = []

    if gsd <= 0:
        warnings.append(f"GSD 必须为正，当前 {gsd}")
    if gsd < 0.005:
        warnings.append(f"GSD={gsd} mm/px 小于 0.005，相当于分辨率优于 5μm，请确认单位是否误用 cm 或 m")
    if gsd > 1.0:
        warnings.append(f"GSD={gsd} mm/px 大于 1.0，相当于 1 像素 1 毫米，请确认是否漏了小数位")

    # 反推视野，便于人工核对
    fov_w = width_px * gsd
    fov_h = height_px * gsd
    return {
        "gsd_mm_per_px": gsd,
        "field_of_view_mm": [round(fov_w, 1), round(fov_h, 1)],
        "warnings": warnings,
        "ok": not warnings,
    }


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    sample = r"E:\yolodata\images\val"
    import glob

    files = sorted(glob.glob(sample + r"\*.bmp"))
    if not files:
        print(f"未找到图像：{sample}")
        sys.exit(0)

    img = files[0]
    print(f"样例图: {img}\n")

    print("=== 已知样品物理宽度时的 GSD 反推 ===")
    for width_mm in (100, 150, 200, 300):
        r = gsd_from_physical(img, width_mm)
        print(f"  样品宽 {width_mm:>4} mm → GSD {r['gsd_mm_per_px']:.6f} mm/px "
              f"（视野 {r['physical_height_mm']:.0f} mm 高）")

    print("\n=== 合理性检查 ===")
    for gsd in (0.0002, 0.027, 0.05, 2.0):
        c = sanity_check(gsd, (5472, 3648))
        status = "OK" if c["ok"] else "WARN"
        print(f"  GSD={gsd:<8} [{status}] 视野 {c['field_of_view_mm']} mm"
              + ("  " + "; ".join(c["warnings"]) if c["warnings"] else ""))
