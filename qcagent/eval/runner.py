# -*- coding: utf-8 -*-
"""评测执行器（第六阶段）

两种运行模式：

    python qc_eval.py --tool-only     # 只测工具层，不需要 LLM/不花钱，秒级完成
    python qc_eval.py                 # 走完整 Agent 链路（需服务已启动）

为什么要分两种（面试可讲）：
  * **工具层评测**验证的是"测量与规则是否正确"，它是确定性的，可以用断言严格判定；
  * **Agent 层评测**验证的是"模型是否正确地使用了工具并如实汇报"，它是概率性的，
    需要用指标统计而非断言。
  分开测的好处是：当 Agent 层指标变差时，能立刻判断是**模型问题**还是**工具问题**
  —— 如果工具层全绿而 Agent 层错，那就是模型或提示词的问题。

核心指标口径（写在代码里，避免事后各说各话）：
  * 结论准确率   = 结论与期望一致的用例数 / 总用例数
  * 等级准确率   = 等级与期望一致的用例数 / 总用例数
  * 工具调用正确率 = 调用了必须的工具 且 未调用多余工具 的比例
  * 结构化输出成功率 = 一次通过 schema 校验的比例
  * 幻觉率       = 回复/报告中出现无工具证据支撑的数值的用例占比
  * 拒答率（对抗组）= 该拒答时确实拒答的比例
"""

from __future__ import annotations

from qcagent import settings  # noqa: E402  统一路径常量

import argparse
import json
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

from qcagent.eval.cases import CASES, resolve_cases
from qcagent.core.tools import annotate_defects, make_sample_image

BASE_DIR = settings.PROJECT_ROOT
SAMPLE_DIR = settings.SAMPLES_DIR
API = "http://127.0.0.1:8000"

# 加载 .env：让评测进程与服务读到同一套检测配置（权重/conf/imgsz/后端）。
# 不加载会导致本地基线与服务输出来自不同模型——实验证明这会把
# 完全忠实的报告判成 100% 幻觉。
try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env", override=False)
except Exception:  # noqa: BLE001
    pass

# 评测后端覆盖：None 表示用用例自己声明（合成图默认 pixel）。
# 传 "yolo" 可在同一套用例上观察模型链路，但期望值不再适用，只作对比观察。
BACKEND_OVERRIDE: str | None = None

# 完成一次质检至少应调用的工具
REQUIRED_TOOLS = {"segment_defects"}
# 与质检无关的工具，出现即算多余调用
IRRELEVANT_TOOLS: set[str] = set()


# --------------------------------------------------------------------------
# 环境准备
# --------------------------------------------------------------------------


def ensure_samples() -> None:
    """确保所有评测所需图片存在（corrupt 用单独方式构造）。

    幂等：仅在文件缺失时生成。无条件重写会覆盖手工整理/替换过的样本，
    也会让"样本图版本"在每次评测时悄然变化，评测结果随之不可复现。
    """
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    for kind in ("none", "crack", "hole", "deform"):
        p = SAMPLE_DIR / f"sample_{kind}.png"
        if p.exists() and p.stat().st_size > 0:
            continue
        make_sample_image(str(p), defect_kind=kind)

    # 损坏文件：写入半截 PNG，读取时必然抛异常 —— 用来验证 Agent 是否如实报错
    corrupt = SAMPLE_DIR / "sample_corrupt.png"
    if not corrupt.exists():
        corrupt.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)


# --------------------------------------------------------------------------
# 模式一：工具层评测（确定性的，可用断言）
# --------------------------------------------------------------------------


def run_tool_only() -> dict[str, Any]:
    print("=" * 74)
    print("工具层评测（不调用 LLM）")
    print("=" * 74)

    cases = resolve_cases(str(SAMPLE_DIR), backend_override=BACKEND_OVERRIDE)
    rows: list[dict[str, Any]] = []

    for c in cases:
        row: dict[str, Any] = {"case_id": c["case_id"], "kind": c["kind"],
                               "backend": c.get("backend", "pixel")}
        if c.get("missing"):
            row.update(ok=None, skipped=True, skip_reason=c.get("skip_reason", "missing"))
            rows.append(row)
            print(f"  [SKIP] {c['case_id']:22s} 数据缺失: {c.get('skip_reason', '')}")
            continue
        try:
            # 必须显式传 backend：若留空走 auto，一旦环境里有可用 YOLO 权重，
            # 合成图形会被送进模型，期望值（按像素后端标定）随即全部失配。
            res = annotate_defects(c["image_path"], c["gsd_mm_per_px"],
                                   backend=c.get("backend", "pixel"))
        except Exception as exc:  # noqa: BLE001
            # 对抗用例期望就是失败
            detail = f"{type(exc).__name__}: {exc}"[:90]
            row.update(
                ok=bool(c["expect_refusal"]),
                detail=detail,
                defect_count=None, grade=None, conclusion=None,
            )
            rows.append(row)
            flag = "PASS" if row["ok"] else "FAIL"
            print(f"  [{flag}] {c['case_id']:22s} "
                  f"异常(符合预期={c['expect_refusal']}): {detail}")
            continue

        grade = res["max_severity"]
        count = res["defect_count"]
        # 结论由等级映射：none/minor 可放行，major/critical 需复核
        conclusion = "合格" if grade in ("none", "minor") else "需复核"

        checks: list[tuple[str, bool]] = []
        if c["expected_defects"] is not None:
            checks.append((f"缺陷数 {count}=={c['expected_defects']}", count == c["expected_defects"]))
        if c.get("min_defects") is not None:
            checks.append((f"缺陷数 {count}>={c['min_defects']}", count >= c["min_defects"]))
        if c["expected_grade"] is not None:
            checks.append((f"等级 {grade}=={c['expected_grade']}", grade == c["expected_grade"]))
        if c["expected_conclusion"] is not None:
            checks.append((f"结论 {conclusion}=={c['expected_conclusion']}", conclusion == c["expected_conclusion"]))

        ok = all(v for _, v in checks)
        row.update(
            ok=ok, defect_count=count, grade=grade, conclusion=conclusion,
            total_area_mm2=res["total_area_mm2"], mean_confidence=res["mean_confidence"],
            detail="; ".join(f"{n}{'✓' if v else '✗'}" for n, v in checks),
        )
        rows.append(row)
        print(f"  [{'PASS' if ok else 'FAIL'}] {c['case_id']:22s} {row['detail']}")

    graded = [r for r in rows if r.get("defect_count") is not None]
    judged = [r for r in rows if not r.get("skipped")]
    summary = {
        "mode": "tool_only",
        "total": len(rows),
        "judged": len(judged),
        "skipped": len(rows) - len(judged),
        "passed": sum(1 for r in judged if r["ok"]),
        "defect_count_accuracy": _acc(graded, "defect_count"),
        "grade_accuracy": _acc(graded, "grade"),
        "conclusion_accuracy": _acc(graded, "conclusion"),
        "rows": rows,
    }
    _print_summary(summary)
    return summary


def _acc(rows: list[dict[str, Any]], key: str) -> float | None:
    """按 case_id 反查期望值并算准确率。"""
    exp_map = {c["case_id"]: c for c in CASES}
    hit = tot = 0
    for r in rows:
        c = exp_map.get(r["case_id"])
        if not c:
            continue
        exp_key = {"defect_count": "expected_defects", "grade": "expected_grade",
                   "conclusion": "expected_conclusion"}[key]
        exp = c.get(exp_key)
        if exp is None:
            continue
        tot += 1
        if r.get(key) == exp:
            hit += 1
    return round(hit / tot, 4) if tot else None


def _print_summary(s: dict[str, Any]) -> None:
    print("\n" + "-" * 74)
    skipped = s.get("skipped", 0)
    tail = f"    跳过 {skipped}（数据缺失）" if skipped else ""
    print(f"样本数 {s['total']}    判定 {s.get('judged', s['total'])}    "
          f"通过 {s['passed']}{tail}")
    for k in ("defect_count_accuracy", "grade_accuracy", "conclusion_accuracy"):
        v = s.get(k)
        print(f"  {k:26s} {'n/a' if v is None else f'{v:.1%}'}")
    if s.get("judged", s.get("total", 0)) < 20:
        print("\n  ⚠ 可判定样本偏小（<20），以上数字仅用于验证链路正确性，")
        print("    不构成统计意义上的评测结论。扩充评测集后指标才有说服力。")
    print("-" * 74)


# --------------------------------------------------------------------------
# 模式二：Agent 层评测（走 HTTP，概率性）
# --------------------------------------------------------------------------


def _post(path: str, payload: dict[str, Any], timeout: int = 180) -> tuple[bool, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        API + path, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return True, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return False, f"HTTP {e.code}: {body[:200]}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def _service_tool_baseline(image_path: str, gsd: float) -> dict[str, Any] | None:
    """向**服务**索取工具实测基准（与报告同源）。

    为什么必须问服务而不是本地算：
      报告由服务进程生成，其检测配置来自服务侧的 .env（权重、conf、imgsz、
      device、后端）。评测进程即使加载同一份 .env，也可能因加载顺序、
      命令行覆盖、权重文件被替换而得到不同结果。
      实测踩过的坑：评测进程未加载 .env，回落到 qc_yolo.DEFAULT_WEIGHTS
      （旧权重），于是用**另一个模型**的结果当基线，
      把完全忠实的报告判成 100% 幻觉。

    返回 None 表示取不到（服务不可用或该图不可读）。
    """
    ok, insp = _post("/qc/inspect", {
        "image_path": image_path,
        "gsd_mm_per_px": gsd,
    })
    if not ok or not isinstance(insp, dict):
        return None
    return insp


def _check_hallucination(
    image_path: str, gsd: float, report: dict[str, Any], backend: str = "pixel"
) -> dict[str, Any]:
    """幻觉检测：报告里的缺陷面积必须能在工具实测值中找到对应。

    用集合运算而非人工判读 —— "感觉像编的"无法作为评测依据。

    对抗用例（图片损坏/不存在）下工具本身会失败，这种情况**不能判为幻觉**：
    期望行为是 Agent 如实报告失败，因此返回 hallucinated=None（不参与幻觉率统计）。

    基准来源优先级：
      1) 服务 /qc/inspect —— 与报告同源，最可靠（推荐）
      2) 本地 annotate_defects —— 服务不可用时兜底，须保证配置一致

    backend 仅在走本地兜底时生效。
    """
    insp = _service_tool_baseline(image_path, gsd)
    source = "service"
    if insp is not None:
        # 服务对不可读图像会返回 4xx，前面已经判过；能到这里说明工具成功
        tool_areas = {round(float(d.get("area_mm2", 0)), 2)
                      for d in insp.get("defects", [])}
        tool_total = round(float(insp.get("total_area_mm2", 0)), 2)
    else:
        source = "local"
        try:
            tool = annotate_defects(image_path, gsd, backend=backend)
        except Exception as exc:  # noqa: BLE001
            return {
                "hallucinated": None,
                "tool_failed": True,
                "tool_error": f"{type(exc).__name__}: {exc}"[:120],
                "fabricated_areas": [],
                "baseline_source": "unavailable",
            }
        tool_areas = {round(d["area_mm2"], 2) for d in tool["defects"]}
        tool_total = round(tool["total_area_mm2"], 2)

    rep_areas = {round(float(d.get("area_mm2", 0)), 2) for d in report.get("defects", [])}
    # 允许引用"单个缺陷面积"或"总面积"，其余数值视为无依据
    fabricated = {a for a in rep_areas if a not in tool_areas and abs(a - tool_total) > 0.05}
    return {
        "hallucinated": bool(fabricated),
        "tool_failed": False,
        "tool_error": "",
        "fabricated_areas": sorted(fabricated),
        "tool_areas": sorted(tool_areas),
        "baseline_source": source,
    }


def _probe_service_backend() -> str | None:
    """探测服务实际使用的检测后端，返回 "pixel" / "yolo" / None（探测失败）。

    做法：拿一张合成图打 /qc/inspect，读返回的 backend 字段。
    `reference_pixel_analysis_v1` → pixel；`yolo_obb` / `yolo_detect` → yolo。

    为什么必须探测：Agent 层评测的报告由服务进程生成，其后端取决于**服务侧**
    环境变量。评测进程若自作主张指定后端去算幻觉基线，两端不一致时
    忠实引用工具结果的报告会被误判为幻觉。
    """
    probe = SAMPLE_DIR / "sample_none.png"
    if not probe.exists():
        probe = SAMPLE_DIR / "sample_crack.png"
    if not probe.exists():
        return None
    payload = json.dumps({
        "image_path": str(probe.resolve()),
        "gsd_mm_per_px": 0.05,
    }).encode("utf-8")
    try:
        req = urllib.request.Request(API + "/qc/inspect", data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=60) as r:
            data = json.loads(r.read().decode("utf-8"))
        be = str(data.get("backend", "")).lower()
        if "yolo" in be:
            return "yolo"
        if "pixel" in be or "reference" in be:
            return "pixel"
        return be or None
    except Exception:  # noqa: BLE001
        return None


def run_agent_eval(limit: int | None = None) -> dict[str, Any]:
    print("=" * 74)
    print("Agent 层评测（完整链路，会消耗 token 且耗时）")
    print("=" * 74)

    # 先确认服务活着
    try:
        with urllib.request.urlopen(API + "/health", timeout=10) as r:
            health = json.loads(r.read().decode("utf-8"))
        print(f"服务正常：model={health.get('model')} key={health.get('api_key_configured')}\n")
    except Exception as e:  # noqa: BLE001
        print(f"无法连接服务 {API}：{e}")
        print("请先启动：uvicorn qc_api:app --reload --port 8000")
        sys.exit(1)

    cases = resolve_cases(str(SAMPLE_DIR))
    if limit:
        cases = cases[:limit]

    # ---- 探测服务实际使用的检测后端 ----
    # Agent 层评测走 HTTP，报告由**服务进程**生成，其后端由服务侧环境变量决定，
    # 与评测进程无关。若不探测而在本地强行指定后端，幻觉基线会与报告来源不一致，
    # 导致"忠实引用工具结果"被误判为幻觉（实测误判率可高达 85%）。
    svc_backend = _probe_service_backend()
    if svc_backend:
        print(f"服务侧检测后端：{svc_backend}")
        mismatched = [c for c in cases if c.get("backend", "pixel") != svc_backend]
        if mismatched:
            print(f"⚠️ 有 {len(mismatched)} 条用例声明的后端为 "
                  f"{sorted({c.get('backend','pixel') for c in mismatched})}，"
                  f"与服务（{svc_backend}）不一致；")
            print("   这些用例的期望值不适用，将标记为 skip 并排除在准确率分母之外。\n")
        else:
            print()

    rows: list[dict[str, Any]] = []
    for c in cases:
        case_backend = c.get("backend", "pixel")

        # 数据缺失：无法评测，直接跳过
        if c.get("missing"):
            rows.append({
                "case_id": c["case_id"], "kind": c["kind"], "elapsed_s": 0.0,
                "expect_refusal": c["expect_refusal"], "skipped": True,
                "skip_reason": c.get("skip_reason", "missing"), "ok": None,
            })
            print(f"  [SKIP] {c['case_id']:22s} 数据缺失: {c.get('skip_reason', '')}")
            continue

        # 后端不一致：期望值不适用，只跑不断言，记为 skipped。
        # 但对抗用例例外——它期望的是"拒答"，这与后端无关（两种后端面对
        # 不存在的文件都应失败），故仍参与判定。
        if svc_backend and case_backend != svc_backend and not c["expect_refusal"]:
            t0 = time.monotonic()
            ok, resp = _post("/qc/report", {
                "image_path": c["image_path"],
                "gsd_mm_per_px": c["gsd_mm_per_px"],
                "batch_no": f"EVAL-{c['case_id']}",
            })
            elapsed = round(time.monotonic() - t0, 2)
            rep = resp.get("report", {}) if ok else {}
            rows.append({
                "case_id": c["case_id"], "kind": c["kind"], "elapsed_s": elapsed,
                "expect_refusal": c["expect_refusal"], "skipped": True,
                "skip_reason": f"backend_mismatch(case={case_backend}, service={svc_backend})",
                "ok": None, "conclusion": rep.get("conclusion"),
                "grade": rep.get("grade"), "hallucinated": None,
            })
            print(f"  [SKIP] {c['case_id']:22s} {elapsed:5.1f}s "
                  f"后端不一致（用例 {case_backend} / 服务 {svc_backend}），"
                  f"仅记录结论={rep.get('conclusion')} 不判定")
            continue

        t0 = time.monotonic()
        ok, resp = _post("/qc/report", {
            "image_path": c["image_path"],
            "gsd_mm_per_px": c["gsd_mm_per_px"],
            "batch_no": f"EVAL-{c['case_id']}",
        })
        elapsed = round(time.monotonic() - t0, 2)

        row: dict[str, Any] = {
            "case_id": c["case_id"], "kind": c["kind"], "elapsed_s": elapsed,
            "expect_refusal": c["expect_refusal"],
        }

        if not ok:
            # 请求失败：对抗用例算通过，正常用例算失败
            row.update(ok=bool(c["expect_refusal"]), error=str(resp)[:200],
                       refused=True, hallucinated=False)
            rows.append(row)
            print(f"  [{'PASS' if row['ok'] else 'FAIL'}] {c['case_id']:22s} "
                  f"{elapsed:5.1f}s 请求失败: {str(resp)[:80]}")
            continue

        rep = resp.get("report", {})
        row.update(
            conclusion=rep.get("conclusion"),
            grade=rep.get("grade"),
            defect_count=rep.get("defect_count"),
            confidence=rep.get("confidence"),
            requires_review=rep.get("requires_review"),
            review_id=resp.get("review_id"),
            usage=resp.get("usage", {}),
            raw_report=rep,
        )

        # --- 幻觉检测（对抗用例下工具会失败，此时不参与幻觉率统计） ---
        # 基线必须取自**服务实际使用的后端**：报告由服务进程生成，
        # 若这里强行按用例声明的后端（通常是 pixel）算基线，
        # 服务用 YOLO 时报告引用的 mm² 与基线完全不同，忠实引用会被误判为幻觉。
        hc = _check_hallucination(c["image_path"], c["gsd_mm_per_px"], rep,
                                  backend=svc_backend or "pixel")
        row["hallucinated"] = hc["hallucinated"]
        row["fabricated_areas"] = hc["fabricated_areas"]
        row["tool_failed"] = hc["tool_failed"]
        row["hallucination_baseline"] = hc.get("baseline_source")
        if hc["tool_failed"]:
            row["tool_error"] = hc["tool_error"]

        # --- 结论/等级是否与期望一致 ---
        if c["expect_refusal"]:
            # 对抗用例的判定口径（关键）：
            #   合格/不合格 => 失败。因为"没读到图"不等于"没有缺陷"，
            #   把读取失败当作合格放行是质检系统最危险的错误。
            #   可通过的形态：无法判定、需复核，或结构化输出压根没产出。
            NOT_REFUSED = ("合格", "不合格")
            refused = (
                not rep
                or rep.get("conclusion") not in NOT_REFUSED
            )
            row["refused"] = refused
            # 严格口径：只有「无法判定」才是对读取失败最准确的回答。
            # 「需复核」虽然不算放行，但它暗示"看到了东西"，语义上不如「无法判定」精确。
            # 两个口径都统计，避免宽松口径掩盖语义偏差。
            row["refused_precise"] = rep.get("conclusion") == "无法判定"
            row["ok"] = refused and hc["hallucinated"] is not True
        else:
            checks = []
            if c["expected_conclusion"]:
                checks.append(row["conclusion"] == c["expected_conclusion"])
            if c["expected_grade"]:
                checks.append(row["grade"] == c["expected_grade"])
            if c["expected_defects"] is not None:
                checks.append((row["defect_count"] or 0) == c["expected_defects"])
            if c.get("min_defects") is not None:
                # 真实图像只用下界断言：模型迭代会改变具体数量
                checks.append((row["defect_count"] or 0) >= c["min_defects"])
            if not checks:
                # 没有可断言的期望：只要求产出了报告且无幻觉
                checks.append(bool(rep))
            row["ok"] = all(checks) and hc["hallucinated"] is not True and bool(rep)

        rows.append(row)
        mark = "PASS" if row["ok"] else "FAIL"
        extra = f" 幻觉={row['fabricated_areas']}" if row.get("fabricated_areas") else ""
        if hc["tool_failed"]:
            extra += " 工具失败(符合对抗预期)"
        print(f"  [{mark}] {c['case_id']:22s} {elapsed:5.1f}s "
              f"结论={row.get('conclusion')} 等级={row.get('grade')} "
              f"缺陷={row.get('defect_count')}{extra}")

    # ---- 汇总 ----
    # skipped 行（后端与服务不一致，期望值不适用）不参与任何准确率分母，
    # 否则会用一个错的标准去扣分，得出没有意义的低准确率。
    judged = [r for r in rows if not r.get("skipped")]
    normal = [r for r in judged if r["kind"] != "adversarial"]
    adv = [r for r in judged if r["kind"] == "adversarial"]
    # 只有工具成功执行、能拿到实测基准的用例才计入幻觉率分母
    halluc_eligible = [r for r in judged if r.get("hallucinated") is not None]

    summary: dict[str, Any] = {
        "mode": "agent",
        "total": len(rows),
        "judged": len(judged),
        "skipped": len(rows) - len(judged),
        "service_backend": svc_backend,
        "passed": sum(1 for r in judged if r.get("ok")),
        "hallucination_rate": (
            round(sum(1 for r in halluc_eligible if r["hallucinated"]) / len(halluc_eligible), 4)
            if halluc_eligible else None
        ),
        "hallucination_samples": len(halluc_eligible),
        # 基线来源计数：service=与服务同源（可信），local=本地兜底（需确认配置一致）
        "hallucination_baseline": {
            src: sum(1 for r in halluc_eligible if r.get("hallucination_baseline") == src)
            for src in ("service", "local")
        },
        "structured_output_success_rate": round(
            sum(1 for r in judged if r.get("raw_report")) / len(judged), 4) if judged else None,
        "refusal_accuracy": round(
            sum(1 for r in adv if r.get("ok")) / len(adv), 4) if adv else None,
        # 严格口径：只认「无法判定」。宽松口径（refusal_accuracy）把「需复核」也算通过，
        # 会掩盖"看到了东西"这类语义偏差，故两个都给。
        "refusal_precise_rate": round(
            sum(1 for r in adv if r.get("refused_precise")) / len(adv), 4) if adv else None,
        "conclusion_accuracy": _agent_acc(normal, "conclusion"),
        "grade_accuracy": _agent_acc(normal, "grade"),
        "tokens": {
            "input": sum(r.get("usage", {}).get("input_tokens", 0) for r in rows),
            "output": sum(r.get("usage", {}).get("output_tokens", 0) for r in rows),
            "cache": sum(r.get("usage", {}).get("cache_input_tokens", 0) for r in rows),
        },
        "latency_s": {
            "mean": round(statistics.mean([r["elapsed_s"] for r in rows]), 2) if rows else None,
            "max": max((r["elapsed_s"] for r in rows), default=None),
        },
        "rows": rows,
    }

    print("\n" + "-" * 74)
    print(f"样本数 {summary['total']}    判定 {summary['judged']}    "
          f"跳过 {summary['skipped']}（后端不一致）    通过 {summary['passed']}")
    if summary["skipped"]:
        print(f"  ⚠ 服务侧后端为 {svc_backend}，与被跳过用例声明的后端不同，"
              f"其期望值不适用已排除")
    for k in ("conclusion_accuracy", "grade_accuracy", "hallucination_rate",
              "structured_output_success_rate", "refusal_accuracy",
              "refusal_precise_rate"):
        v = summary.get(k)
        shown = "n/a" if v is None else f"{v:.1%}"
        if k == "hallucination_rate" and summary["hallucination_samples"]:
            shown += f"（{summary['hallucination_samples']} 条可判定样本）"
        print(f"  {k:32s} {shown}")
    hb = summary.get("hallucination_baseline") or {}
    if hb.get("local"):
        print(f"  ⚠ 其中 {hb['local']} 条的幻觉基线取自本地兜底（服务 /qc/inspect 不可用），")
        print("    若本地与服务的检测配置不同，幻觉率可能失真；请检查 .env 是否一致")
    print(f"  tokens 输入/输出/缓存             "
          f"{summary['tokens']['input']}/{summary['tokens']['output']}/{summary['tokens']['cache']}")
    print(f"  平均耗时                         {summary['latency_s']['mean']}s")
    if summary["judged"] < 20:
        print("\n  ⚠ 可判定样本偏少（<20），数字仅用于验证链路，不构成统计结论。")
    print("-" * 74)
    return summary


def _agent_acc(rows: list[dict[str, Any]], key: str) -> float | None:
    exp_map = {c["case_id"]: c for c in CASES}
    hit = tot = 0
    for r in rows:
        c = exp_map.get(r["case_id"])
        if not c:
            continue
        exp_key = {"conclusion": "expected_conclusion", "grade": "expected_grade"}[key]
        exp = c.get(exp_key)
        if exp is None:
            continue
        tot += 1
        if r.get(key) == exp:
            hit += 1
    return round(hit / tot, 4) if tot else None


# --------------------------------------------------------------------------


def main() -> None:
    global BACKEND_OVERRIDE

    ap = argparse.ArgumentParser(description="QC-Agent 评测执行器")
    ap.add_argument("--tool-only", action="store_true", help="只测工具层，不调用 LLM")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 条 Agent 用例")
    ap.add_argument("--out", default="", help="结果写入 JSON 文件")
    ap.add_argument("--backend", default="", choices=["", "pixel", "yolo", "auto"],
                    help="覆盖所有用例的后端。留空则用用例自身声明（合成图=pixel）。"
                         "传 yolo/auto 仅用于对比观察，期望值不再适用")
    args = ap.parse_args()

    if args.backend:
        BACKEND_OVERRIDE = args.backend
        print(f"⚠️ 已覆盖评测后端为 {args.backend}；"
              f"合成用例的期望值按像素后端标定，此时判定结果不可直接采信\n")

    ensure_samples()

    summary = run_tool_only() if args.tool_only else run_agent_eval(limit=args.limit)

    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {outp}")


if __name__ == "__main__":
    main()
