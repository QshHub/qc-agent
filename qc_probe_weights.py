# -*- coding: utf-8 -*-
"""权重有效性快检（在投入全量评测前先看一眼）

## 为什么需要这一步

训练产物缺少 `results.csv` 与 `weights/`，说明训练可能在 5 轮中途中断。
这种权重有两种可能：

  A) 学到了东西，只是没跑到最佳 —— 可以直接评测
  B) 根本没学到（输出全是背景/置信度极低）—— 评测会得到 0 检出，白跑

本脚本用 val 集的 3~5 张图，把**原始置信度分布**打出来。
若最大置信度长期低于 0.1，基本可判定模型未收敛。

## 用法

    python qc_probe_weights.py                      # 默认 5 张 val 图
    python qc_probe_weights.py --conf 0.01          # 极低阈值，看模型到底输出什么
    python qc_probe_weights.py --weights E:\\yolodata\\best.pt
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


def main() -> None:
    ap = argparse.ArgumentParser(description="权重有效性快检")
    ap.add_argument("--weights", default=r"E:\yolodata\best.pt")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--conf", type=float, default=0.01,
                    help="探测用极低阈值，看模型真实输出分布")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        print(f"缺少依赖: {exc}")
        print("需要 ultralytics 与 torch")
        sys.exit(1)

    print("=" * 76)
    print("权重有效性快检")
    print("=" * 76)
    print(f"  权重      {args.weights}")
    print(f"  torch     {torch.__version__}")
    print(f"  CUDA 可用 {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU       {torch.cuda.get_device_name(0)}")

    wpath = Path(args.weights)
    if not wpath.exists():
        print(f"\n❌ 权重不存在: {wpath}")
        sys.exit(1)

    model = YOLO(str(wpath))
    names = getattr(model, "names", {}) or {}
    print(f"\n  类别数    {len(names)}")
    print(f"  类别名    {list(names.values())[:4]} ...")
    print(f"  任务      {getattr(model, 'task', None)}")

    # 训练时是否有加载过数据？看 model.args 里的 imgsz
    tr_args = getattr(model, "args", None)
    if tr_args is not None:
        for k in ("imgsz", "epochs", "batch", "device", "data"):
            if hasattr(tr_args, k):
                print(f"  训练参数 {k:8s} {getattr(tr_args, k)}")

    images = sorted(glob.glob(str(DATA_ROOT / "images" / args.split / "*.bmp")))[: args.n]
    if not images:
        print(f"\n未找到图像: {DATA_ROOT / 'images' / args.split}")
        sys.exit(1)

    print(f"\n{'=' * 76}")
    print(f"探测 {len(images)} 张图，conf={args.conf}（极低阈值，看原始输出）")
    print(f"{'=' * 76}")

    dev = args.device or ("0" if torch.cuda.is_available() else "cpu")
    all_confs: list[float] = []
    total_boxes = 0

    for p in images:
        try:
            res = model.predict(source=p, conf=args.conf, imgsz=args.imgsz,
                                device=dev, verbose=False)[0]
        except Exception as exc:  # noqa: BLE001
            print(f"\n{Path(p).name}: 推理失败 {type(exc).__name__}: {exc}")
            continue

        obb = getattr(res, "obb", None)
        n = len(obb) if obb is not None else 0
        total_boxes += n
        confs = [float(c) for c in obb.conf.cpu().numpy()] if n else []
        cls = [int(c) for c in obb.cls.cpu().numpy()] if n else []
        all_confs.extend(confs)

        top = f"{max(confs):.3f}" if confs else "—"
        print(f"\n{Path(p).name}")
        print(f"  检出框 {n} 个  最高置信度 {top}")
        if n:
            for c, k in sorted(zip(confs, cls), reverse=True)[:6]:
                nm = names.get(k, k)
                print(f"    conf={c:.4f}  class={k} ({nm})")

    print(f"\n{'=' * 76}")
    print("结论")
    print(f"{'=' * 76}")
    print(f"  总检出框数     {total_boxes}")
    if all_confs:
        import statistics

        print(f"  置信度 最大     {max(all_confs):.4f}")
        print(f"  置信度 中位     {statistics.median(all_confs):.4f}")
        print(f"  置信度 >0.25    {sum(1 for c in all_confs if c > 0.25)} 个")
        print(f"  置信度 >0.50    {sum(1 for c in all_confs if c > 0.50)} 个")

        if max(all_confs) < 0.10:
            print("\n  ❌ 判定：模型基本未收敛（最高置信度 < 0.1）")
            print("     5 轮 smoke test 通常不足以学出可用模型。")
            print("     建议：至少训 100 轮，且确认 imgsz 与数据分辨率匹配。")
        elif sum(1 for c in all_confs if c > 0.25) == 0:
            print("\n  ⚠️ 判定：有输出但置信度都低于 0.25")
            print("     可在评测时降低 --conf（如 0.1）看召回，但正式指标需重新训练。")
        else:
            print("\n  ✅ 判定：模型有可用输出，可以跑全量评测")
            print(f"     python qc_detect_eval.py --conf 0.25 --imgsz {args.imgsz}")
    else:
        print("\n  ❌ 判定：所有图都零检出（即便 conf=0.01）")
        print("     模型确定未学到缺陷特征，必须重新训练。")

    print("\n  参考：完整评测")
    print(f"    python qc_detect_eval.py --weights {args.weights} --conf 0.25 --imgsz {args.imgsz}")


if __name__ == "__main__":
    main()
