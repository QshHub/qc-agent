# -*- coding: utf-8 -*-
"""端到端验收脚本：一次跑完 Agent 系统的四条主链路

不需要记 curl / PowerShell 语法，直接：
    python qc_acceptance.py

会依次验证：
    ① /health          服务与依赖状态
    ② /qc/inspect      确定性质检（模型是否正确接入）
    ③ /qc/report       Agent 端到端（工具调用 + 结构化报告）
    ④ /qc/review/*     人工复核闭环（队列 / 认领 / 判定 / 统计）

只依赖标准库，不引入额外包。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

BASE = "http://127.0.0.1:8000"
DEFAULT_IMAGE = r"E:/yolodata/images/val/bplw003.bmp"


def hr(t: str) -> None:
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


def req(method: str, path: str, body: dict | None = None, timeout: int = 300) -> tuple[bool, Any]:
    url = BASE + path
    data = json.dumps(body, ensure_ascii=False).encode("utf-8") if body is not None else None
    r = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return True, (json.loads(raw) if raw else {})
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')[:400]}"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {e}"


def main() -> None:
    ap = argparse.ArgumentParser(description="QC-Agent 端到端验收")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--gsd", type=float, default=0.0545)
    ap.add_argument("--skip-review", action="store_true", help="跳过复核闭环")
    args = ap.parse_args()

    passed: list[str] = []
    failed: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        (passed if ok else failed).append(name)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    # ---------------- ① 健康检查 ----------------
    hr("① /health  服务与依赖状态")
    ok, h = req("GET", "/health", timeout=15)
    if not ok:
        print(f"  ❌ 服务未启动或不可达：{h}")
        print("\n  请先启动：")
        print("    cd D:\\agent_projects\\PythonProject")
        print("    python -m uvicorn qc_api:app --port 8000")
        sys.exit(1)

    print(f"  version              {h.get('version')}")
    print(f"  模型                 {h.get('model')}")
    print(f"  API Key 已配置       {h.get('api_key_configured')}")
    print(f"  工具                 {h.get('tools')}")
    print(f"  数据库               {h.get('database')}")
    print(f"  历史报告 / 待复核    {h.get('total_reports')} / {h.get('review_queue_open')}")
    check("服务可用", h.get("status") == "ok")
    check("API Key 已加载", bool(h.get("api_key_configured")))
    check("工具已注册", len(h.get("tools") or []) >= 2)
    check("数据库就绪", h.get("database") == "ok")

    # ---------------- ② 确定性质检 ----------------
    hr("② /qc/inspect  确定性质检（验证模型接入）")
    t0 = time.monotonic()
    ok, insp = req("POST", "/qc/inspect",
                   {"image_path": args.image, "gsd_mm_per_px": args.gsd})
    elapsed = time.monotonic() - t0
    if not ok:
        print(f"  ❌ 请求失败：{insp}")
        print("\n  常见原因：")
        print("   · qc_api.py 是旧版（DefectItem.area_px 为 int，OBB 面积是 float → 500）")
        print("   · 未重新同步：python qc_sync_files.py")
        print("   · .env 里 QC_YOLO_WEIGHTS 路径不对")
        failed.append("确定性质检")
    else:
        print(f"  后端            {insp.get('backend')}")
        print(f"  模型任务/类别   {insp.get('model_task')} / {insp.get('model_nc')}")
        print(f"  缺陷数          {insp.get('defect_count')}")
        print(f"  面积来源        {insp.get('area_source')}")
        print(f"  物理量可信      {insp.get('physical_metrics_reliable')}")
        print(f"  综合等级        物理={insp.get('max_severity')}  "
              f"相对={insp.get('max_severity_relative')}")
        sr = insp.get("sample_region") or {}
        print(f"  样品区域        {sr.get('source')} "
              f"{sr.get('width', 0):.0f}x{sr.get('height', 0):.0f}px")
        print(f"  耗时            {elapsed:.2f}s")

        defects = insp.get("defects") or []
        if defects:
            print(f"\n  {'实例':<6}{'类别':<10}{'conf':>7}{'面积px':>12}"
                  f"{'占样品':>9}{'等级':>8}")
            print("  " + "-" * 56)
            for d in defects[:8]:
                ar = d.get("area_ratio_sample")
                print(f"  {d.get('instance_id', ''):<6}{d.get('class_name') or '?':<10}"
                      f"{d.get('confidence', 0):>7.3f}{d.get('area_px', 0):>12.1f}"
                      f"{(f'{ar:.3%}' if ar is not None else '-'):>9}"
                      f"{d.get('severity_relative') or '-':>8}")

        check("确定性质检成功", True)
        check("使用 YOLO 后端", "yolo" in str(insp.get("backend", "")))
        check("检出缺陷", (insp.get("defect_count") or 0) > 0,
              f"检出 {insp.get('defect_count')} 个")
        check("样品区域已标定",
              sr.get("source") == "cached_calibration",
              f"来源={sr.get('source')}")

    # ---------------- ③ Agent 端到端 ----------------
    hr("③ /qc/report  Agent 端到端（工具调用 + 结构化报告）")
    t0 = time.monotonic()
    ok, rep = req("POST", "/qc/report", {
        "image_path": args.image,
        "gsd_mm_per_px": args.gsd,
        "batch_no": f"ACC-{int(time.time()) % 100000}",
    })
    elapsed = time.monotonic() - t0
    if not ok:
        print(f"  ❌ 请求失败：{rep}")
        print("  可能原因：模型不支持结构化输出、或工具调用未完成")
        failed.append("Agent 端到端")
    else:
        r = rep.get("report") or {}
        print(f"  report_id       {rep.get('report_id')}")
        print(f"  review_id       {rep.get('review_id')}")
        print(f"  结论            {r.get('conclusion')}")
        print(f"  等级            {r.get('grade')}")
        print(f"  缺陷数          {r.get('defect_count')}")
        print(f"  置信度          {r.get('confidence')}")
        print(f"  需复核          {r.get('requires_review')}")
        print(f"  判定依据        {(r.get('standard_basis') or '')[:80]}")
        print(f"  处置建议        {(r.get('disposal') or '')[:80]}")
        usage = rep.get("usage") or {}
        print(f"  token           输入 {usage.get('input_tokens')} / "
              f"输出 {usage.get('output_tokens')} / 缓存 {usage.get('cache_input_tokens')}")
        print(f"  耗时            {elapsed:.1f}s")

        check("Agent 生成报告", bool(r.get("conclusion")))
        check("结论合法", r.get("conclusion") in ("合格", "不合格", "需复核", "无法判定"),
              str(r.get("conclusion")))
        check("需复核时返回 review_id",
              (not r.get("requires_review")) or bool(rep.get("review_id")))

        if not args.skip_review and rep.get("review_id"):
            rid = rep["review_id"]

            # ---------------- ④ 复核闭环 ----------------
            hr("④ 人工复核闭环")
            ok, q = req("GET", "/qc/review/queue?limit=5")
            check("复核队列可读", ok, f"共 {q.get('total')} 条" if ok else str(q)[:100])

            ok, c = req("POST", f"/qc/review/{rid}/claim", {"reviewer": "acceptance_bot"})
            check("认领任务", ok, str(c.get("status")) if ok else str(c)[:120])

            ok, d = req("POST", f"/qc/review/{rid}/decide", {
                "reviewer": "acceptance_bot",
                "verdict": "rejected",
                "category": "clause_wrong",
                "comment": "验收测试：模拟条款引用错误",
                "corrected_conclusion": "不合格",
                "corrected_grade": "critical",
            })
            if ok:
                print(f"  判定            {d.get('verdict')}  原因={d.get('category_label')}")
                print(f"  人工结论        {d.get('final_conclusion')} / {d.get('final_grade')}")
                print(f"  是否覆盖模型    {d.get('overridden')}")
            check("提交判定", ok, "" if ok else str(d)[:150])

            ok, det = req("GET", f"/qc/reports/{rep.get('report_id')}")
            if ok:
                print(f"  模型原始结论    {det.get('conclusion')} / {det.get('grade')}")
                print(f"  人工权威结论    {det.get('final_conclusion')} / {det.get('final_grade')}")
                print(f"  已复核          {det.get('reviewed')}")
                check("人工结论已覆盖模型结论", det.get("final_conclusion") == "不合格")
                check("模型原始判断仍保留", det.get("conclusion") in ("需复核", "不合格", "无法判定"))
            else:
                check("回读报告", False, str(det)[:120])

            ok, st = req("GET", "/qc/review/stats")
            if ok:
                print(f"\n  统计  报告总数={st.get('total_reports')}  "
                      f"队列={st.get('queue')}")
                print(f"        判定={st.get('decided')}  通过率={st.get('approval_rate')}")
                cats = st.get("reject_categories") or []
                if cats:
                    print(f"        驳回原因分布: {[(c['label'], c['count']) for c in cats[:4]]}")
                check("统计可读", True)
            else:
                check("统计可读", False, str(st)[:120])

    # ---------------- 汇总 ----------------
    hr("验收结果")
    print(f"  通过 {len(passed)} 项，失败 {len(failed)} 项")
    if failed:
        print("\n  失败项：")
        for f in failed:
            print(f"   · {f}")
    else:
        print("\n  ✅ 全部通过 —— Agent 系统端到端可用")
        print("\n  下一步：")
        print("   · 打开演示页看进度推送： http://127.0.0.1:8000/demo")
        print("   · 打开 API 文档：        http://127.0.0.1:8000/docs")
    print("=" * 78)

    sys.exit(0 if not failed else 2)


if __name__ == "__main__":
    main()
