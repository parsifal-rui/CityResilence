#!/usr/bin/env python
"""
对比两份抽取结果 JSON，按文章逐条比较 events 的：
- 数量差异
- event_text / event_type / date_from / date_to / location 内容差异
输出便于人工校验的报告。
"""
import json
import sys
from pathlib import Path


def load_results(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    return data.get("results", data)


def event_key(e):
    return (
        e.get("event_text"),
        e.get("event_type"),
        e.get("date_from"),
        e.get("date_to"),
        _loc_str(e.get("location")),
    )


def _loc_str(loc):
    if loc is None:
        return ""
    return "/".join(loc) if isinstance(loc, list) else str(loc)


def compare_events(events_a, events_b, label_a="A", label_b="B"):
    lines = []
    n_a, n_b = len(events_a), len(events_b)
    if n_a != n_b:
        lines.append(f"  【数量不同】 {label_a}={n_a} 条，{label_b}={n_b} 条")
    else:
        lines.append(f"  数量: {n_a} 条（一致）")

    common = min(n_a, n_b)
    for i in range(common):
        ea, eb = events_a[i], events_b[i]
        diffs = []
        if ea.get("event_text") != eb.get("event_text"):
            diffs.append(f"event_text: {label_a}={ea.get('event_text')!r} | {label_b}={eb.get('event_text')!r}")
        if ea.get("event_type") != eb.get("event_type"):
            diffs.append(f"event_type: {label_a}={ea.get('event_type')!r} | {label_b}={eb.get('event_type')!r}")
        if ea.get("date_from") != eb.get("date_from"):
            diffs.append(f"date_from: {label_a}={ea.get('date_from')!r} | {label_b}={eb.get('date_from')!r}")
        if ea.get("date_to") != eb.get("date_to"):
            diffs.append(f"date_to: {label_a}={ea.get('date_to')!r} | {label_b}={eb.get('date_to')!r}")
        loc_a, loc_b = _loc_str(ea.get("location")), _loc_str(eb.get("location"))
        if loc_a != loc_b:
            diffs.append(f"location: {label_a}=[{loc_a}] | {label_b}=[{loc_b}]")

        if diffs:
            lines.append(f"  [{i}] {ea.get('event_id', '?')} vs {eb.get('event_id', '?')} —— 差异:")
            for d in diffs:
                lines.append(f"      • {d}")
        else:
            lines.append(f"  [{i}] 一致 — event_text={ea.get('event_text')!r}, type={ea.get('event_type')}, date={ea.get('date_from')}~{ea.get('date_to')}, location={_loc_str(ea.get('location'))}")

    if n_a > common:
        lines.append(f"  【仅{label_a}多出】 共 {n_a - common} 条:")
        for i in range(common, n_a):
            e = events_a[i]
            lines.append(f"      [{i}] event_text={e.get('event_text')!r}, type={e.get('event_type')}, date={e.get('date_from')}~{e.get('date_to')}, location={_loc_str(e.get('location'))}")
    if n_b > common:
        lines.append(f"  【仅{label_b}多出】 共 {n_b - common} 条:")
        for i in range(common, n_b):
            e = events_b[i]
            lines.append(f"      [{i}] event_text={e.get('event_text')!r}, type={e.get('event_type')}, date={e.get('date_from')}~{e.get('date_to')}, location={_loc_str(e.get('location'))}")

    return "\n".join(lines)


def main():
    root = Path(__file__).resolve().parent
    default_a = root / "deepseek_event_schema_results.json"
    default_b = root / "deepseek_event_schema_rag_results.json"
    path_a = Path(sys.argv[1]) if len(sys.argv) > 1 else default_a
    path_b = Path(sys.argv[2]) if len(sys.argv) > 2 else default_b

    if not path_a.is_file():
        print(f"文件不存在: {path_a}", file=sys.stderr)
        sys.exit(1)
    if not path_b.is_file():
        print(f"文件不存在: {path_b}", file=sys.stderr)
        sys.exit(1)

    results_a = load_results(path_a)
    results_b = load_results(path_b)
    name_a, name_b = path_a.name, path_b.name

    report_lines = [
        f"对比: {name_a} (A) vs {name_b} (B)",
        "关注: event_text, event_type, time(date_from/date_to), location",
        "=" * 60,
    ]
    diff_summary = []

    n = max(len(results_a), len(results_b))
    for idx in range(n):
        ra = results_a[idx] if idx < len(results_a) else None
        rb = results_b[idx] if idx < len(results_b) else None
        if ra is None or rb is None:
            report_lines.append(f"\nArticle {idx}: 仅一方有结果，跳过")
            continue
        if "error" in ra:
            report_lines.append(f"\nArticle {idx} (A 报错): {ra.get('error', '')[:80]}...")
            continue
        if "error" in rb:
            report_lines.append(f"\nArticle {idx} (B 报错): {rb.get('error', '')[:80]}...")
            continue

        events_a = (ra.get("extraction") or {}).get("events", [])
        events_b = (rb.get("extraction") or {}).get("events", [])

        block = compare_events(events_a, events_b, label_a="A", label_b="B")
        if "数量不同" in block or "差异:" in block or "仅A多出" in block or "仅B多出" in block:
            diff_summary.append(idx)
        report_lines.append(f"\n--- Article {idx} (article_id={ra.get('article_id', '')}, publish_date={ra.get('publish_date', '')}) ---")
        report_lines.append(block)

    report_lines.append("\n" + "=" * 60)
    report_lines.append(f"【需人工校验的文章 index】共 {len(diff_summary)} 篇: {diff_summary}")

    out_path = root / "compare.txt"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"报告已写入: {out_path}")
    print(f"有数量或内容差异的文章 index: {diff_summary}")


if __name__ == "__main__":
    main()
