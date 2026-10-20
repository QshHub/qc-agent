# -*- coding: utf-8 -*-
"""评测集定义（第六阶段）

设计原则：
  * **用例与执行分离**：本文件只描述"输入 + 期望"，不含任何执行逻辑。
    这样扩充评测集时不需要碰评测代码，也便于把评测集单独交给他人标注。
  * **期望值可推导**：期望等级由 qc_tools 的规则推导得出，而非手工填写，
    避免规则改了而评测集没同步（这类不一致会直接污染评测结果）。
  * **按类别组织**：normal / param_variant / adversarial 三类分别对应
    "正常链路""参数鲁棒性""该拒答时是否拒答"，前两类算准确率，第三类单独看拒答率。
  * **每例显式声明后端**：合成图形只为检验**规则引擎与拒答逻辑**，
    故默认走确定性像素后端（pixel），不依赖 GPU 或训练好的权重；
    真实模型的检测质量另由 qc_detect_eval.py 在真实产线数据上单独评测。
    两件事混在一套用例里会让结果无法解释——模型一换，期望值全错。
  * **样本量诚实标注**：当前 n 很小，脚本会显式提示，不做统计学过度解读。

扩充评测集的方式：往 CASES 里加 dict 即可。真实产线数据标注完成后，
把 image_path 指向真实图片、backend 设为 "yolo"、期望值按实测填写即可。
"""

from __future__ import annotations

from typing import Any

from qcagent.core.tools import DEFAULT_GSD_MM_PER_PX, _severity_of

# 期望等级由规则推导：与 qc_tools.SEVERITY_RULES 同源，杜绝"评测集与实现不一致"
def expected_grade(area_mm2: float, max_len_mm: float) -> str:
    return _severity_of(area_mm2, max_len_mm)


def _case(
    case_id: str,
    kind: str,
    image_kind: str,
    *,
    file_name: str | None = None,
    gsd: float = DEFAULT_GSD_MM_PER_PX,
    expected_defects: int | None = None,
    min_defects: int | None = None,
    expected_grade: str | None = None,
    expected_conclusion: str | None = None,
    expect_refusal: bool = False,
    backend: str = "pixel",
    note: str = "",
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "kind": kind,
        "image_kind": image_kind,
        # file_name 用于指向"不该存在的文件"这类特殊用例；
        # 为空时按 sample_{image_kind}.png 约定解析
        "file_name": file_name,
        "gsd_mm_per_px": gsd,
        # backend 显式声明本用例测哪个后端：
        #   pixel（默认）—— 测规则引擎与拒答逻辑。合成图形结果确定、与模型无关，
        #                   不需要 GPU 或训练好的权重，任何环境都能复现
        #   yolo          —— 测真实模型链路，需 QC_YOLO_WEIGHTS 可用
        # 期望值随后端不同而不同，混用会让评测结果失去意义。
        "backend": backend,
        "expected_defects": expected_defects,
        # min_defects：只要求"至少检出 N 处"，用于真实图像。
        # 模型迭代会改变具体数量，写死等于把评测绑死在某一版权重上。
        "min_defects": min_defects,
        "expected_grade": expected_grade,
        "expected_conclusion": expected_conclusion,
        "expect_refusal": expect_refusal,
        "note": note,
    }


# --------------------------------------------------------------------------
# 评测集
#
# 期望值来源：qc_tools 的参考分割后端在默认 GSD=0.05 下实测得出
#   none   -> 0 处缺陷                       -> 合格
#   crack  -> 1 处, 6.92 mm², 12.3 mm       -> major (尺寸超 8mm)
#   hole   -> 2 处, 15.006 mm², 4.006 mm    -> minor
#   deform -> 1 处, 37.765 mm², 17.05 mm    -> major (面积超 20mm²)
# --------------------------------------------------------------------------

CASES: list[dict[str, Any]] = [
    # ---------- A. 正常链路（结论正确性） ----------
    _case("A01-no-defect", "normal", "none",
          expected_defects=0, expected_grade="none", expected_conclusion="合格",
          note="无缺陷图，应判合格且不臆造问题"),
    _case("A02-crack-major", "normal", "crack",
          expected_defects=1, expected_grade="major", expected_conclusion="需复核",
          note="裂缝，面积未超阈值但尺寸超 8mm，应判 major 并转复核"),
    _case("A03-hole-minor", "normal", "hole",
          expected_defects=2, expected_grade="minor", expected_conclusion="合格",
          note="两处孔道缺失，均在 minor 阈值内，可放行"),
    _case("A04-deform-major", "normal", "deform",
          expected_defects=1, expected_grade="major", expected_conclusion="需复核",
          note="形变带面积 37.765mm² 超 20mm² 阈值，应判 major"),

    # ---------- B. 参数变体（鲁棒性：GSD 变化应导致等级变化） ----------
    _case("B01-crack-gsd-double", "param_variant", "crack", gsd=0.10,
          expected_defects=1, expected_grade="critical", expected_conclusion="需复核",
          note="GSD 翻倍 -> 面积变 4 倍、尺寸变 2 倍，应升级为 critical"),
    _case("B02-hole-gsd-coarse", "param_variant", "hole", gsd=0.20,
          expected_defects=2, expected_grade="critical", expected_conclusion="需复核",
          note="GSD 变 4 倍：孔道缺失等效直径 16.0mm、外接框 14.0mm，"
               "取较大者 16.0mm 已超 major 阈值；实测最大尺寸 20.03mm 越过 20mm，"
               "故判 critical（此期望值经实跑校正，原始推算遗漏了等效直径项）"),
    _case("B03-deform-gsd-fine", "param_variant", "deform", gsd=0.02,
          expected_defects=1, expected_grade="minor", expected_conclusion="合格",
          note="GSD 变小，形变带降级为 minor"),

    # ---------- C. 对抗性：该拒答时必须拒答（不计入准确率，单独看拒答率） ----------
    _case("C01-nonexistent-file", "adversarial", "none",
          file_name="__this_file_does_not_exist__.png", expect_refusal=True,
          note="图像不存在：工具必须报错，Agent 不得把'没读到图'当成'没有缺陷'而放行"),
    _case("C02-corrupt-image", "adversarial", "corrupt",
          file_name="sample_corrupt.png", expect_refusal=True,
          note="损坏文件，读取失败，Agent 不得编造结果"),

    # ---------- D. 真实产线图像（覆盖 YOLO 链路） ----------
    #
    # 合成图形只能检验规则引擎，覆盖不到训练好的模型。这一类用真实数据集图像，
    # 期望值刻意只用**结构性质**（至少检出 1 处、等级落在合法集合内），
    # 不写死具体数量——模型迭代后数量会变，写死会让评测每隔一版就红一次。
    #
    # 路径由 QC_EVAL_DATA_ROOT 指定（默认 E:/yolodata），找不到时自动跳过。
    _case("D01-real-face-crack", "real_image", "real",
          file_name="real:bplw003.bmp", backend="yolo", min_defects=1,
          note="真实端面裂纹图（val 集），YOLO 应至少检出 1 处"),
    _case("D02-real-face-crack2", "real_image", "real",
          file_name="real:bpqk004.bmp", backend="yolo", min_defects=1,
          note="真实端面缺陷图（val 集），YOLO 应至少检出 1 处"),
    _case("D03-real-face-crack3", "real_image", "real",
          file_name="real:bpqk010.bmp", backend="yolo", min_defects=1,
          note="真实端面缺陷图（val 集），YOLO 应至少检出 1 处"),
]


def resolve_cases(sample_dir: str, *, backend_override: str | None = None) -> list[dict[str, Any]]:
    """把 image_kind 解析成具体文件路径。

    路径解析放在这里而不是 CASES 里，是为了让评测集本身与部署路径解耦
    —— 换机器、换数据目录时不用改评测集定义。

    真实图像用例用 `real:文件名` 前缀标记，指向 QC_EVAL_DATA_ROOT
    （默认 E:/yolodata）下的 images/val。数据不存在时保留用例但标记 missing，
    由执行器跳过——这样在没数据的环境里评测不会直接崩掉。

    Args:
        backend_override: 强制覆盖所有用例的后端（如传 "yolo" 可在同一套用例上
            对比模型链路表现，但此时期望值不再适用，只应观察结果不做判定）。
    """
    import os
    from pathlib import Path

    base = Path(sample_dir)
    data_root = Path(os.getenv("QC_EVAL_DATA_ROOT", r"E:/yolodata"))

    out: list[dict[str, Any]] = []
    for c in CASES:
        item = dict(c)
        name = c.get("file_name") or f"sample_{c['image_kind']}.png"

        if name.startswith("real:"):
            fname = name.split(":", 1)[1]
            p = (data_root / "images" / "val" / fname)
            item["image_path"] = str(p)
            item["missing"] = not p.exists()
            if item["missing"]:
                item["skip_reason"] = f"real_image_missing({p})"
        else:
            item["image_path"] = str((base / name).resolve())
            item["missing"] = False

        if backend_override:
            item["backend"] = backend_override
        out.append(item)
    return out


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    print(f"评测集共 {len(CASES)} 条\n")
    by_kind: dict[str, int] = {}
    for c in CASES:
        by_kind[c["kind"]] = by_kind.get(c["kind"], 0) + 1
    for k, n in by_kind.items():
        print(f"  {k:16s} {n} 条")
    print()
    for c in CASES:
        exp = "拒答" if c["expect_refusal"] else f"{c['expected_conclusion']}({c['expected_grade']})"
        print(f"  {c['case_id']:22s} {c['image_kind']:8s} gsd={c['gsd_mm_per_px']:<5} 期望={exp:18s} {c['note']}")
