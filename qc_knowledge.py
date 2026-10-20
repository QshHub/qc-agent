# -*- coding: utf-8 -*-
"""缺陷判据知识库（已对齐真实数据集）

## ⚠️ 判据来源说明（重要）

`REAL_DATASET_TAXONOMY` 的 12 个类别与 `E:\\yolodata\\data.yaml` 严格对应，
这是 YOLO 实际输出的类别。

但它们的**条款号与判定阈值目前为空（PENDING）**，原因：

  * 条款号必须来自真实的**企业标准 / 国标 / 客户技术协议**，不能编造；
  * 旧版里的 `Q/QC-JS-001` 等编号是示例占位符，**不可用于生产**；
  * 判定阈值（多少 mm² 判废）是业务决策，需质量部门确认。

在补全之前，`lookup_standard()` 会明确返回「判据待补充」，
Agent 据此应拒绝给出合格/不合格结论、转人工判定。
**这比让模型编一个条款号安全得多。**

补全方式：把 `REAL_DATASET_TAXONOMY` 各项的 `standard` / `clause` /
`criteria` / `disposal` 填上真实内容即可，代码无需改动。

## 与旧版的关系

`LEGACY_TAXONOMY`（孔道缺失、轴向裂缝等）保留作**演示与回归测试基准**，
`lookup_standard()` 同时支持两套。真实判据补全后可删除旧版。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

PENDING = "待补充"

# --------------------------------------------------------------------------
# 真实数据集类别（与 E:\yolodata\data.yaml 一一对应）
# --------------------------------------------------------------------------

REAL_DATASET_TAXONOMY: dict[str, dict[str, Any]] = {
    # ---- 材料缺失类 ----
    "material_loss.collapse": {
        "category": "材料缺失类", "name": "孔塌陷", "class_id": 0,
        "standard": PENDING, "clause": PENDING, "criterion": PENDING, "disposal": PENDING,
    },
    "material_loss.channel_missing": {
        "category": "材料缺失类", "name": "孔洞", "class_id": 1,
        "standard": PENDING, "clause": PENDING, "criterion": PENDING, "disposal": PENDING,
    },
    "material_loss.face_chip": {
        "category": "材料缺失类", "name": "端面缺口", "class_id": 2,
        "standard": PENDING, "clause": PENDING, "criterion": PENDING, "disposal": PENDING,
    },
    "material_loss.missing_cell_wall": {
        "category": "材料缺失类", "name": "孔壁缺失", "class_id": 4,
        "standard": PENDING, "clause": PENDING, "criterion": PENDING, "disposal": PENDING,
    },
    "material_loss.sidewall_chip": {
        "category": "材料缺失类", "name": "表皮缺口", "class_id": 5,
        "standard": PENDING, "clause": PENDING, "criterion": PENDING, "disposal": PENDING,
    },
    "material_loss.corner_chip": {
        "category": "材料缺失类", "name": "缺角", "class_id": 6,
        "standard": PENDING, "clause": PENDING, "criterion": PENDING, "disposal": PENDING,
    },
    "material_loss.damage": {
        "category": "材料缺失类", "name": "损坏", "class_id": 11,
        "standard": PENDING, "clause": PENDING, "criterion": PENDING, "disposal": PENDING,
    },
    # ---- 缝隙分离类 ----
    "gap_separation.face_crack": {
        "category": "缝隙分离类", "name": "端面裂纹", "class_id": 3,
        "standard": PENDING, "clause": PENDING, "criterion": PENDING, "disposal": PENDING,
    },
    "gap_separation.external_wall_tear": {
        "category": "缝隙分离类", "name": "侧壁撕裂", "class_id": 7,
        "standard": PENDING, "clause": PENDING, "criterion": PENDING, "disposal": PENDING,
    },
    "gap_separation.skin_crack": {
        "category": "缝隙分离类", "name": "表皮裂纹", "class_id": 8,
        "standard": PENDING, "clause": PENDING, "criterion": PENDING, "disposal": PENDING,
    },
    "gap_separation.transverse_crack": {
        "category": "缝隙分离类", "name": "横向裂纹", "class_id": 9,
        "standard": PENDING, "clause": PENDING, "criterion": PENDING, "disposal": PENDING,
    },
    # ---- 结构形变类 ----
    "deformation.shortness": {
        "category": "结构形变类", "name": "长度不足", "class_id": 10,
        "standard": PENDING, "clause": PENDING, "criterion": PENDING, "disposal": PENDING,
    },
}


# --------------------------------------------------------------------------
# 旧版示例类别（仅用于演示与回归测试，非真实业务判据）
# --------------------------------------------------------------------------

LEGACY_TAXONOMY: dict[str, dict[str, Any]] = {
    # ---- 材料缺失类 ----
    # 注意：以下 3 个键曾与真实数据集类别同名（channel_missing / face_chip / collapse），
    # 已加 legacy. 前缀以避免遮蔽真实类别。真实类别在 REAL_DATASET_TAXONOMY 中。
    "legacy.material_loss.channel_missing": {
        "category": "材料缺失类",
        "name": "孔道缺失",
        "standard": "Q/QC-JS-001",
        "clause": "5.2.1",
        "criterion": "端面存在整孔或半孔缺失，单处面积不超过 20 mm²",
        "disposal": "单处超标时判定不合格，整批加严抽检；未超标时记录并降级使用",
    },
    "legacy.material_loss.face_chip": {
        "category": "材料缺失类",
        "name": "端面缺损",
        "standard": "Q/QC-JS-001",
        "clause": "5.2.2",
        "criterion": "端面边缘崩缺深度不超过 1.5 mm，且不影响密封面",
        "disposal": "涉及密封面时直接判废；非密封面可修磨后复检",
    },
    "material_loss.coating_peel": {
        "category": "材料缺失类",
        "name": "涂层脱落",
        "standard": "Q/QC-JS-001",
        "clause": "5.2.3",
        "criterion": "活性涂层局部脱落面积不超过孔壁面积的 5%",
        "disposal": "超标时判定不合格；未超标时记录脱落位置并跟踪后续批次",
    },
    "legacy.material_loss.collapse": {
        "category": "材料缺失类",
        "name": "局部塌陷",
        "standard": "Q/QC-JS-001",
        "clause": "5.2.4",
        "criterion": "不允许出现结构性塌陷，任何尺寸的塌陷均判不合格",
        "disposal": "直接判废，并追溯成型工序压力参数",
    },
    # ---- 缝隙分离类 ----
    "gap_separation.axial_crack": {
        "category": "缝隙分离类",
        "name": "轴向裂缝",
        "standard": "Q/QC-JS-002",
        "clause": "6.1.1",
        "criterion": "轴向裂缝最大长度不超过 20 mm，且不得贯穿整个通道",
        "disposal": "贯穿性裂缝直接判废；未贯穿时标记为关注项并加强后续工序检查",
    },
    "gap_separation.hoop_crack": {
        "category": "缝隙分离类",
        "name": "环向裂缝",
        "standard": "Q/QC-JS-002",
        "clause": "6.1.2",
        "criterion": "环向裂缝累计弧长不超过端面周长的 10%",
        "disposal": "超标时判不合格；未超标时记录并观察使用",
    },
    "gap_separation.interface_debond": {
        "category": "缝隙分离类",
        "name": "界面剥离",
        "standard": "Q/QC-JS-002",
        "clause": "6.1.3",
        "criterion": "基体与涂层界面不允许出现剥离",
        "disposal": "直接判废，追溯涂覆工艺与烧结曲线",
    },
    "gap_separation.edge_crack": {
        "category": "缝隙分离类",
        "name": "边缘开裂",
        "standard": "Q/QC-JS-002",
        "clause": "6.1.4",
        "criterion": "边缘开裂长度不超过 5 mm，且不得向内部延伸",
        "disposal": "向内部延伸时判废；否则修边后复检",
    },
    # ---- 结构形变类 ----
    "deformation.overall": {
        "category": "结构形变类",
        "name": "整体变形",
        "standard": "Q/QC-JS-003",
        "clause": "7.1.1",
        "criterion": "端面平面度偏差不超过 0.5 mm",
        "disposal": "超标时判不合格，追溯烧结支撑与装窑方式",
    },
    "deformation.channel_twist": {
        "category": "结构形变类",
        "name": "孔道扭曲",
        "standard": "Q/QC-JS-003",
        "clause": "7.1.2",
        "criterion": "孔道轴线偏移不超过孔道边长的 10%",
        "disposal": "超标时判不合格；未超标时记录并评估背压影响",
    },
    "deformation.face_tilt": {
        "category": "结构形变类",
        "name": "端面倾斜",
        "standard": "Q/QC-JS-003",
        "clause": "7.1.3",
        "criterion": "端面与轴线垂直度偏差不超过 0.3 mm",
        "disposal": "超标时磨平后复检，影响装配时判不合格",
    },
    "deformation.dimension_out": {
        "category": "结构形变类",
        "name": "尺寸超差",
        "standard": "Q/QC-JS-003",
        "clause": "7.1.4",
        "criterion": "外形尺寸偏差应在图纸公差带内（一般 ±0.5 mm）",
        "disposal": "超差判不合格；临界值复测确认后放行",
    },
}

# 严重等级定义：两套口径并存
#
# 物理口径（若 GSD 可信）与相对口径（始终可用）应给出同一等级。
# 相机未标定时只能用相对口径，此处在 condition 里同时写明两者，
# 避免 Agent 只看到物理阈值就据此误判。
GRADE_RULES: list[dict[str, Any]] = [
    {
        "grade": "critical",
        "label": "严重",
        "condition": "缺陷面积 > 样品面积 6% 或 缺陷长边 > 样品边长 25%"
                     "（若 GSD 已标定，等价于面积 > 50 mm² 或最大尺寸 > 20 mm）",
        "action": "判不合格，禁止放行，追溯工艺",
    },
    {
        "grade": "major",
        "label": "较重",
        "condition": "缺陷面积 > 样品面积 2% 或 缺陷长边 > 样品边长 12%"
                     "（若 GSD 已标定，等价于面积 > 20 mm² 或最大尺寸 > 8 mm）",
        "action": "转人工复核，确认后判废或降级",
    },
    {
        "grade": "minor",
        "label": "轻微",
        "condition": "缺陷面积 ≤ 样品面积 2% 且 缺陷长边 ≤ 样品边长 12%",
        "action": "记录归档，可放行",
    },
]


# --------------------------------------------------------------------------
# 外部判据配置加载
#
# 判据属于**业务数据**，不该硬编码在代码里——标准会修订、不同产品线口径不同。
# 因此支持从 configs/qc_criteria.yaml（或 .json）覆盖内置默认值。
#
# 优先级：配置文件 > REAL_DATASET_TAXONOMY 内置默认（默认全部为 PENDING）
# 留空的字段保持 PENDING，lookup_standard 会如实返回"尚未录入"。
# --------------------------------------------------------------------------

CRITERIA_PATH = Path(__file__).resolve().parent / "configs" / "qc_criteria.yaml"
_CRITERIA_META: dict[str, Any] = {}


def _read_criteria_file(path: Path) -> dict[str, Any]:
    """读取配置。优先 PyYAML，缺失时回退 json（同目录同名 .json）。"""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # PyYAML，ultralytics 已依赖

            return yaml.safe_load(text) or {}
        except ImportError:
            alt = path.with_suffix(".json")
            if alt.exists():
                import json

                return json.loads(alt.read_text(encoding="utf-8", errors="replace"))
            print(
                f"[qc_knowledge] 未安装 PyYAML，跳过 {path.name}。"
                f"可改用同名 .json 或 pip install pyyaml"
            )
            return {}
        except Exception as exc:  # noqa: BLE001
            print(f"[qc_knowledge] 解析 {path.name} 失败: {type(exc).__name__}: {exc}")
            return {}
    if path.suffix.lower() == ".json":
        import json

        return json.loads(text)
    return {}


def load_criteria(path: Path | None = None, *, verbose: bool = False) -> dict[str, Any]:
    """把外部配置合并进 REAL_DATASET_TAXONOMY。

    Returns:
        统计信息 {filled, pending, total, source}
    """
    global _CRITERIA_META

    p = path or CRITERIA_PATH
    data = _read_criteria_file(p)
    if not data:
        return {"filled": 0, "pending": len(REAL_DATASET_TAXONOMY),
                "total": len(REAL_DATASET_TAXONOMY), "source": None}

    _CRITERIA_META = dict(data.get("meta") or {})

    filled = 0
    for key, cfg in (data.get("criteria") or {}).items():
        if key not in REAL_DATASET_TAXONOMY or not isinstance(cfg, dict):
            continue
        item = REAL_DATASET_TAXONOMY[key]
        std = str(cfg.get("standard") or "").strip()
        clause = str(cfg.get("clause") or "").strip()
        crit = str(cfg.get("criterion") or "").strip()
        disp = str(cfg.get("disposal") or "").strip()
        # 只有 standard 与 criterion 都有值才算"已录入"
        if std and crit:
            item["standard"] = std
            item["clause"] = clause or "-"
            item["criterion"] = crit
            item["disposal"] = disp or "按企业质量流程处置"
            filled += 1
        else:
            item["standard"] = PENDING
            item["clause"] = PENDING
            item["criterion"] = PENDING
            item["disposal"] = PENDING

    # 同步等级阈值到 qc_severity（若给了）
    th = data.get("grade_thresholds") or {}
    if th:
        try:
            import qc_severity

            rules = []
            for grade in ("critical", "major"):
                cfg = th.get(grade) or {}
                a = cfg.get("area_ratio")
                l = cfg.get("len_ratio")
                if a is not None and l is not None:
                    rules.append((grade, float(a), float(l)))
            if rules:
                rules.append(("minor", 0.0, 0.0))
                qc_severity.RELATIVE_RULES = tuple(rules)
                if verbose:
                    print(f"[qc_knowledge] 已应用自定义等级阈值: {rules}")
        except Exception as exc:  # noqa: BLE001
            print(f"[qc_knowledge] 同步等级阈值失败: {type(exc).__name__}: {exc}")

    if verbose:
        print(f"[qc_knowledge] 判据配置 {p.name}: 已录入 {filled}/{len(REAL_DATASET_TAXONOMY)} 类")
    return {
        "filled": filled,
        "pending": len(REAL_DATASET_TAXONOMY) - filled,
        "total": len(REAL_DATASET_TAXONOMY),
        "source": str(p) if p.exists() else None,
        "meta": _CRITERIA_META,
    }


def reload_criteria(path: Path | None = None) -> dict[str, Any]:
    """热重载判据配置（服务运行中修改配置后可调用）。"""
    return load_criteria(path, verbose=True)


def criteria_status() -> dict[str, Any]:
    """报告判据录入情况，供 /health 或运维检查。"""
    filled = [k for k, v in REAL_DATASET_TAXONOMY.items() if v["standard"] != PENDING]
    pending = [k for k, v in REAL_DATASET_TAXONOMY.items() if v["standard"] == PENDING]
    return {
        "total": len(REAL_DATASET_TAXONOMY),
        "filled": len(filled),
        "pending": len(pending),
        "filled_keys": filled,
        "pending_keys": pending,
        "config_path": str(CRITERIA_PATH),
        "config_exists": CRITERIA_PATH.exists(),
        "meta": _CRITERIA_META,
    }


# 模块导入时自动加载一次
load_criteria(verbose=False)


# --------------------------------------------------------------------------
# 合并视图：对外统一入口
#
# 真实数据集类别在前（优先匹配），旧版演示类别在后（保证回归测试可用）。
# 两者键名不冲突，因此简单合并即可。
# --------------------------------------------------------------------------

DEFECT_TAXONOMY: dict[str, dict[str, Any]] = {
    **REAL_DATASET_TAXONOMY,
    **LEGACY_TAXONOMY,
}


def list_taxonomy() -> str:
    """列出全部缺陷类别及其判据，供 Agent 归类时对照。

    当需要确定某个缺陷属于哪一类、或需要引用判据条款时调用。
    """
    lines = ["缺陷分类体系与判据："]
    for section, book in (("真实产线类别", REAL_DATASET_TAXONOMY),
                          ("演示用示例类别（非生产判据）", LEGACY_TAXONOMY)):
        lines.append(f"\n===== {section} =====")
        current_cat = None
        for key, item in book.items():
            if item["category"] != current_cat:
                current_cat = item["category"]
                lines.append(f"\n【{current_cat}】")
            if item["standard"] == PENDING:
                lines.append(f"  - {item['name']}（类别键 {key}）｜判据：尚未录入，需人工判定")
            else:
                lines.append(
                    f"  - {item['name']}（类别键 {key}）"
                    f"｜判据 {item['standard']} {item['clause']}：{item['criterion']}"
                    f"｜处置：{item['disposal']}"
                )
    lines.append("\n严重等级定义（按实测尺寸，与判据条款相互独立）：")
    for g in GRADE_RULES:
        lines.append(f"  - {g['grade']}（{g['label']}）：{g['condition']} → {g['action']}")
    return "\n".join(lines)


def lookup_standard(defect_key_or_name: str) -> str:
    """按缺陷类别键或中文名查询判据条款与处置要求。

    ★ 若该类别的判据尚未录入，会**明确返回「判据待补充」**并要求人工判定。
      这是刻意的设计：宁可让 Agent 说"无法判定"，也不让它编造条款号。

    Args:
        defect_key_or_name: 类别键（如 gap_separation.hoop_crack）或中文名（如 横向裂纹）
    """
    query = defect_key_or_name.strip()

    # 精确键命中
    item = DEFECT_TAXONOMY.get(query)
    # 中文名命中
    if item is None:
        for v in DEFECT_TAXONOMY.values():
            if v["name"] == query:
                item = v
                break
    # 模糊命中
    if item is None:
        for v in DEFECT_TAXONOMY.values():
            if query in v["name"] or v["name"] in query:
                item = v
                break

    if item is None:
        names = "、".join(v["name"] for v in DEFECT_TAXONOMY.values())
        return f"未找到匹配的缺陷类别「{query}」。现有类别：{names}"

    if item["standard"] == PENDING:
        return (
            f"缺陷类别：{item['category']} - {item['name']}\n"
            f"判据条款：尚未录入（该类别已在分类体系中定义，但质量判定阈值与条款号待补充）\n"
            f"处理要求：**不得据此给出合格/不合格结论**，应转人工判定，"
            f"并说明'该类别的判定依据尚未配置'。"
        )

    return (
        f"缺陷类别：{item['category']} - {item['name']}\n"
        f"判据条款：{item['standard']} {item['clause']}\n"
        f"判定标准：{item['criterion']}\n"
        f"处置要求：{item['disposal']}"
    )


def match_taxonomy_by_severity(severity: str) -> str:
    """按严重等级返回对应的处置要求。

    Args:
        severity: critical / major / minor
    """
    for g in GRADE_RULES:
        if g["grade"] == severity.strip().lower():
            return f"{g['grade']}（{g['label']}）：{g['condition']} → 处置：{g['action']}"
    return f"未知等级「{severity}」。有效等级：critical / major / minor"


if __name__ == "__main__":
    import sys

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    print(list_taxonomy())
    print("\n" + "=" * 60)
    print(lookup_standard("轴向裂缝"))
    print("\n" + "=" * 60)
    print(lookup_standard("gap_separation.axial_crack"))
    print("\n" + "=" * 60)
    print(lookup_standard("不存在的缺陷"))
