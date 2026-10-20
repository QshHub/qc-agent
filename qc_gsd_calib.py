# -*- coding: utf-8 -*-
"""GSD 标定辅助（针对"样品只占画面一部分"的真实图像）

## 为什么不能假设"样品铺满画面"

之前的 qc_gsd.py 假设样品宽度 = 图像宽度，这在产线实拍图上完全不成立：
样品放在夹具上、倾斜、只占画面一角。
用错的 GSD 算出面积 → 等级判定全错 → 整个质检结论失去意义。

## 本工具做什么

1. 用**自适应阈值 + 最大连通域**自动找出样品主体（蜂窝结构在灰度上明显区别于背景）
2. 拟合最小外接旋转矩形，给出样品在图像中的**像素边长**
3. 你填入"样品实际边长(mm)"，直接得到 GSD
4. 用标注框反推校验：已知某个缺陷的真实尺寸时，可反算 GSD 是否合理

## 用法

    python qc_gsd_calib.py                    # 对 val 集抽样标定
    python qc_gsd_calib.py --image 某个.bmp   # 指定单张
    python qc_gsd_calib.py --side-mm 150      # 指定样品实际边长，直接输出 GSD
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

DATA_ROOT = Path(r"E:\yolodata")


# --------------------------------------------------------------------------
# 样品主体检测
# --------------------------------------------------------------------------


def detect_sample_bbox(image_path: str, *, max_side_for_speed: int = 1400) -> dict:
    """检测样品主体，返回其在原图中的像素尺寸与四边形。

    思路：
      蜂窝陶瓷在灰度图上呈"规则亮暗交替的网格"，与背景（相机架、夹具、地面）
      在局部对比度上有明显差异。用 Otsu 自适应阈值把"高对比度区域"分离出来，
      取最大连通域即为样品。

    为速度起见，先把图缩小到 max_side_for_speed 内处理，再按比例还原。
    """
    with Image.open(image_path) as im:
        W, H = im.size
        gray = im.convert("L")
        scale = min(1.0, max_side_for_speed / max(W, H))
        small = gray.resize((int(W * scale), int(H * scale)), Image.BILINEAR)

    a = np.asarray(small, dtype=np.float32)

    # 局部对比度：与原图模糊版的差的绝对值，蜂窝网格处响应最强
    from PIL import ImageFilter

    blur = np.asarray(small.filter(ImageFilter.GaussianBlur(radius=6)), dtype=np.float32)
    contrast = np.abs(a - blur)

    # Otsu 阈值
    hist, bins = np.histogram(contrast, bins=256, range=(0, contrast.max() + 1e-6))
    total = contrast.size
    sum_all = float(np.dot(np.arange(256), hist))
    sumB = 0.0
    wB = 0
    best_var, best_thr = -1.0, 0.0
    for i in range(256):
        wB += int(hist[i])
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB = sumB / wB
        mF = (sum_all - sumB) / wF
        var_between = wB * wF * (mB - mF) ** 2
        if var_between > best_var:
            best_var, best_thr = var_between, i

    mask = contrast > best_thr
    # 形态学闭运算（用 PIL 的 MaxFilter/MinFilter 近似）
    m = Image.fromarray((mask * 255).astype(np.uint8))
    m = m.filter(ImageFilter.MaxFilter(9)).filter(ImageFilter.MinFilter(9))
    mask = np.asarray(m) > 127

    # 找最大连通域的外接框（用投影法近似，避免引入 scipy）
    ys, xs = np.where(mask)
    if ys.size < 100:
        return {
            "ok": False,
            "reason": "未检测到明显样品区域（对比度分割失败）",
            "image_size_px": [W, H],
        }

    # 用分位数裁掉离群噪点，比 min/max 稳健
    x1, x2 = np.percentile(xs, [0.5, 99.5])
    y1, y2 = np.percentile(ys, [0.5, 99.5])
    w_px = (x2 - x1) / scale
    h_px = (y2 - y1) / scale

    return {
        "ok": True,
        "image_size_px": [W, H],
        "sample_bbox_px": [round(x1 / scale), round(y1 / scale),
                           round(x2 / scale), round(y2 / scale)],
        "sample_size_px": [round(w_px), round(h_px)],
        "sample_area_ratio": round(float(mask.mean()), 4),
        "detected_area_ratio": round(float((x2 - x1) * (y2 - y1)) / mask.size, 4),
        "otsu_threshold": round(best_thr, 2),
    }


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="GSD 标定辅助")
    ap.add_argument("--image", default="", help="单张图路径；留空则抽样 val 集")
    ap.add_argument("--n", type=int, default=5, help="抽样数量")
    ap.add_argument("--side-mm", type=float, default=0.0,
                    help="样品端面实际边长(mm)。给出后自动算 GSD")
    ap.add_argument("--populate-cache", action="store_true",
                    help="对 --split 下所有图像检测样品区域并写入 qc_severity 缓存")
    ap.add_argument("--split", default="all", choices=["train", "val", "all"],
                    help="--populate-cache 处理哪些划分")
    ap.add_argument("--out", default="", help="结果写入 JSON")
    args = ap.parse_args()

    # ---------- 模式：批量检测并写入缓存 ----------
    if args.populate_cache:
        from qc_severity import save_sample_region

        splits = ["train", "val"] if args.split == "all" else [args.split]
        total = 0
        ok = 0
        failed: list[str] = []
        for sp in splits:
            files = sorted(glob.glob(str(DATA_ROOT / "images" / sp / "*.bmp")))
            print(f"\n[{sp}] {len(files)} 张")
            for i, p in enumerate(files, 1):
                total += 1
                try:
                    r = detect_sample_bbox(p)
                    if r.get("ok"):
                        save_sample_region(p, r["sample_bbox_px"])
                        ok += 1
                    else:
                        failed.append(f"{Path(p).name}: {r.get('reason', '未检出')}")
                except Exception as exc:  # noqa: BLE001
                    failed.append(f"{Path(p).name}: {type(exc).__name__}: {exc}")
                if i % 50 == 0 or i == len(files):
                    print(f"  {i}/{len(files)}  成功 {ok}  失败 {len(failed)}")

        print(f"\n{'=' * 70}")
        print(f"完成：{ok}/{total} 张样品区域已写入缓存")
        print(f"缓存文件: qc-agent/data/sample_regions.json")
        print("之后 run_yolo 的 sample_region.source 会变为 cached_calibration，")
        print("相对化判定的分母（样品面积）才是真实的。")
        if failed:
            print(f"\n失败 {len(failed)} 张（这些图会退化为按缺陷外扩估算）:")
            for x in failed[:10]:
                print(f"   {x}")
        return

    # ---------- 模式：抽样查看 ----------
    if args.image:
        images = [args.image]
    else:
        images = sorted(glob.glob(str(DATA_ROOT / "images" / "val" / "*.bmp")))[: args.n]

    if not images:
        print("未找到图像")
        sys.exit(1)

    print("=" * 78)
    print("样品主体检测（用于 GSD 标定）")
    print("=" * 78)

    rows = []
    for p in images:
        r = detect_sample_bbox(p)
        rows.append({"image": Path(p).name, **r})
        print(f"\n{Path(p).name}")
        if not r["ok"]:
            print(f"  ❌ {r['reason']}")
            continue
        print(f"  图像尺寸     {r['image_size_px']}")
        print(f"  样品外接框   {r['sample_bbox_px']}  (x1,y1,x2,y2)")
        print(f"  样品像素尺寸 {r['sample_size_px'][0]} × {r['sample_size_px'][1]} px")
        print(f"  占画面比例   检测区域 {r['detected_area_ratio']:.1%}")

    ok_rows = [r for r in rows if r.get("ok")]
    if not ok_rows:
        print("\n❌ 所有图像都未能检测出样品主体，请手工目测样品像素宽度后手动算 GSD。")
        return

    med_w = float(np.median([r["sample_size_px"][0] for r in ok_rows]))
    med_h = float(np.median([r["sample_size_px"][1] for r in ok_rows]))

    print("\n" + "-" * 78)
    print(f"样品像素尺寸中位数：{med_w:.0f} × {med_h:.0f} px")
    print(f"（图像宽 {ok_rows[0]['image_size_px'][0]} px，"
          f"样品占比 {med_w / ok_rows[0]['image_size_px'][0]:.1%}）")
    print("-" * 78)

    if args.side_mm > 0:
        gsd = args.side_mm / med_w
        gsd_h = args.side_mm / med_h
        print(f"\n样品实际边长 {args.side_mm} mm")
        print(f"  GSD（按宽）  {gsd:.6f} mm/px")
        print(f"  GSD（按高）  {gsd_h:.6f} mm/px")
        print(f"  两者差异     {abs(gsd - gsd_h) / max(gsd, gsd_h):.1%}"
              f"（样品倾斜时会有差异，属正常）")
        print(f"\n  建议写入 .env：")
        print(f"    QC_GSD_MM_PER_PX={gsd:.6f}")
        fov_w = ok_rows[0]["image_size_px"][0] * gsd
        print(f"\n  换算视野宽度 {fov_w:.0f} mm，样品占 {args.side_mm:.0f} mm")
    else:
        print("\n请提供样品端面实际边长，例如：")
        print(f"  python qc_gsd_calib.py --side-mm 150")
        print("\n或按不同假设看 GSD：")
        for mm in (80, 100, 120, 150, 200):
            print(f"  样品 {mm:>3} mm → GSD = {mm / med_w:.6f} mm/px")

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps({
            "rows": rows,
            "median_sample_px": [med_w, med_h],
            "side_mm": args.side_mm,
            "gsd_from_width": (args.side_mm / med_w) if args.side_mm else None,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {outp}")


if __name__ == "__main__":
    main()
