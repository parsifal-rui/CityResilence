"""
验收：与 deepseek_event_schema_modified 中事件字段要求（约 272–280 行）
及关系约束（约 323–325 行）对齐的最小质检。

事件
- evidence 必须能在原文中找到（归一化后子串匹配）。
- event_type / city / date / location / attributes.severity / attributes.time_uncertain
  按 schema 要求检查。

关系（仅 extraction.relations）
- 端点 event_id 必须属于本篇文章事件列表（不得新增事件、不得用不存在 id）。
- 填写 source_event_name / target_event_name，且与对应事件的 event_name 一致（便于核对）。

用法：
  python eval_v2.py --results_json results/run1/results.json --out_json out/eval_summary_v2.json
  python eval_v2.py --results_json ... --articles_csv data/article/articles_deduped.csv --out_json ...
  （deduped 类 CSV 无 article_id 时按行序对齐 A000/A001…，须与 results 生成顺序一致）
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
ALLOWED_EVENT_TYPES = frozenset({"Driver", "Modulator", "Hazard", "Impact"})
ALLOWED_SEVERITY = frozenset({"", "mild", "moderate", "severe"})
ALLOWED_RELATION_TYPES = frozenset({"引发", "加剧", "削弱", "增强"})
MAX_EVIDENCE_CHARS = 120


def norm_text(s: Any) -> str:
    s = "" if s is None else str(s)
    s = s.lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[“”\"'‘’]", "", s)
    s = re.sub(r"[，,。．\.！!？?\-—_；;：:\(\)\[\]\{\}<>《》]", "", s)
    return s


def is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


def parse_yyyy_mm_dd(s: Any) -> Optional[Tuple[int, int, int]]:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not DATE_RE.match(s):
        return None
    try:
        d = datetime.strptime(s, "%Y-%m-%d").date()
        return (d.year, d.month, d.day)
    except Exception:
        return None


def norm_aid(x: Any) -> str:
    if is_missing(x):
        return ""
    s = str(x).strip()
    m = re.fullmatch(r"[Aa]?0*(\d+)", s)
    if m:
        return str(int(m.group(1)))
    return s.lower()


def load_results_json(path: str | Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and isinstance(obj.get("results"), list):
        return obj["results"]
    if isinstance(obj, dict) and isinstance(obj.get("records"), list):
        return obj["records"]
    if isinstance(obj, list):
        return obj
    raise ValueError("results_json 应为 list，或含 results/records 的 dict")


def load_articles_csv_index(articles_csv: str | Path) -> Dict[str, str]:
    """
    正文列名：content / text / article / body / full_text。
    ID 列名：article_id / id / doc_id；若均无，则按数据行顺序第 i 行对应
    results 里 index=i、article_id 形如 A000 的篇（与 norm_aid 对齐）。
    """
    idx: Dict[str, str] = {}
    with open(articles_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("articles_csv 无表头")
        id_keys = ["article_id", "id", "doc_id"]
        text_keys = ["content", "text", "article", "body", "full_text"]
        for row_i, row in enumerate(reader):
            aid = next((str(row[k]).strip() for k in id_keys if k in row and not is_missing(row.get(k))), None)
            txt = next((str(row[k]) for k in text_keys if k in row and not is_missing(row.get(k))), None)
            if not txt:
                continue
            if aid:
                idx[aid] = txt
                idx[norm_aid(aid)] = txt
            else:
                idx[str(row_i)] = txt
                sym = f"A{row_i:03d}"
                idx[sym] = txt
                idx[sym.lower()] = txt
                idx[norm_aid(sym)] = txt
    return idx


def article_body(rec: dict, articles_index: Optional[Dict[str, str]]) -> str:
    aid = rec.get("article_id")
    if articles_index:
        for k in (str(aid).strip() if aid is not None else "", norm_aid(aid)):
            if k in articles_index:
                return articles_index[k]
    return str(rec.get("content") or rec.get("full_text") or rec.get("content_preview") or "")


def evidence_in_body(evd: str, body: str) -> bool:
    evd = (evd or "").strip()
    if not evd or not body:
        return False
    return norm_text(evd) in norm_text(body)


def check_event_schema(ev: dict) -> List[str]:
    """对应 schema 272–280；返回 issue 码列表（空表示该维度均通过）。"""
    issues: List[str] = []
    et = ev.get("event_type")
    if is_missing(et) or str(et).strip() not in ALLOWED_EVENT_TYPES:
        issues.append("event_type_bad")

    city = ev.get("city")
    if city is not None and not isinstance(city, str):
        issues.append("city_type_bad")
    elif isinstance(city, str) and city.strip() == "":
        issues.append("city_empty_string")

    for key in ("date_from", "date_to"):
        v = ev.get(key)
        if not is_missing(v) and parse_yyyy_mm_dd(v) is None:
            issues.append(f"{key}_format_bad")
    df = parse_yyyy_mm_dd(ev.get("date_from"))
    dt = parse_yyyy_mm_dd(ev.get("date_to"))
    if df and dt:
        if datetime(df[0], df[1], df[2]).date() > datetime(dt[0], dt[1], dt[2]).date():
            issues.append("date_order_bad")

    loc = ev.get("location")
    if not isinstance(loc, list):
        issues.append("location_not_list")
    else:
        if not all(isinstance(x, str) for x in loc):
            issues.append("location_elem_not_str")

    attrs = ev.get("attributes")
    if not isinstance(attrs, dict):
        issues.append("attributes_not_dict")
    else:
        if "severity" not in attrs:
            issues.append("severity_missing")
        else:
            sev = attrs.get("severity")
            if sev is None:
                issues.append("severity_null")
            elif str(sev).strip() not in ALLOWED_SEVERITY:
                issues.append("severity_bad")
        if "time_uncertain" not in attrs:
            issues.append("time_uncertain_missing")
        elif not isinstance(attrs.get("time_uncertain"), bool):
            issues.append("time_uncertain_not_bool")

    return issues


def check_relation_rules(
    rel: dict,
    event_ids: set[str],
    id_to_name: Dict[str, str],
) -> List[str]:
    """对应关系约束 323–325 + evidence 在原文（调用方单独记）。"""
    issues: List[str] = []
    sid = str(rel.get("source_event_id", "")).strip()
    tid = str(rel.get("target_event_id", "")).strip()
    if is_missing(rel.get("source_event_id")):
        issues.append("rel_missing_source_event_id")
    if is_missing(rel.get("target_event_id")):
        issues.append("rel_missing_target_event_id")
    if sid and sid not in event_ids:
        issues.append("rel_source_id_unknown")
    if tid and tid not in event_ids:
        issues.append("rel_target_id_unknown")

    sn = rel.get("source_event_name")
    tn = rel.get("target_event_name")
    if is_missing(sn):
        issues.append("rel_missing_source_event_name")
    if is_missing(tn):
        issues.append("rel_missing_target_event_name")

    if sid in id_to_name and not is_missing(sn):
        if str(sn).strip() != id_to_name[sid]:
            issues.append("rel_source_name_mismatch")
    if tid in id_to_name and not is_missing(tn):
        if str(tn).strip() != id_to_name[tid]:
            issues.append("rel_target_name_mismatch")

    rt = rel.get("relation_type")
    if is_missing(rt) or str(rt).strip() not in ALLOWED_RELATION_TYPES:
        issues.append("rel_relation_type_bad")

    evd = rel.get("evidence")
    if is_missing(evd):
        issues.append("rel_evidence_missing")
    elif len(str(evd).strip()) > MAX_EVIDENCE_CHARS:
        issues.append("rel_evidence_too_long")

    return issues


def evaluate(
    records: List[Dict[str, Any]],
    *,
    articles_index: Optional[Dict[str, str]] = None,
    max_examples: int = 15,
) -> Dict[str, Any]:
    ev_issue_stats: Counter = Counter()
    rel_issue_stats: Counter = Counter()
    examples: Dict[str, List[dict]] = defaultdict(list)

    def ex(issue: str, payload: dict) -> None:
        if len(examples[issue]) < max_examples:
            examples[issue].append(payload)

    n_articles = len(records)
    n_events = 0
    n_rels = 0
    n_evidence_ok = 0
    n_schema_ok = 0
    n_event_all_ok = 0
    n_rel_id_name_ok = 0
    n_rel_evidence_ok = 0
    n_rel_all_ok = 0

    for rec in records:
        ext = rec.get("extraction") or {}
        events = ext.get("events") if isinstance(ext.get("events"), list) else []
        relations = ext.get("relations") if isinstance(ext.get("relations"), list) else []
        body = article_body(rec, articles_index)

        event_ids: set[str] = set()
        id_to_name: Dict[str, str] = {}
        for ev in events:
            if not isinstance(ev, dict):
                continue
            eid = ev.get("event_id")
            if not is_missing(eid):
                eid_s = str(eid).strip()
                if eid_s in event_ids:
                    ev_issue_stats["event_duplicate_id"] += 1
                    ex("event_duplicate_id", {"article_id": rec.get("article_id"), "event_id": eid_s})
                event_ids.add(eid_s)
                id_to_name[eid_s] = str(ev.get("event_name", "")).strip()

        for ev in events:
            if not isinstance(ev, dict):
                continue
            n_events += 1
            eid = ev.get("event_id")
            evd = ev.get("evidence")

            schema_issues = check_event_schema(ev)
            for it in schema_issues:
                ev_issue_stats[it] += 1
                ex(it, {"article_id": rec.get("article_id"), "event_id": eid, "event": ev})

            schema_ok = len(schema_issues) == 0
            if schema_ok:
                n_schema_ok += 1

            ev_ok = False
            if not is_missing(evd):
                if len(str(evd).strip()) > MAX_EVIDENCE_CHARS:
                    ev_issue_stats["event_evidence_too_long"] += 1
                    ex("event_evidence_too_long", {"article_id": rec.get("article_id"), "event_id": eid})
                elif not body:
                    ev_issue_stats["event_evidence_no_body"] += 1
                    ex("event_evidence_no_body", {"article_id": rec.get("article_id"), "event_id": eid})
                elif evidence_in_body(str(evd), body):
                    ev_ok = True
                    n_evidence_ok += 1
                else:
                    ev_issue_stats["event_evidence_not_in_text"] += 1
                    ex("event_evidence_not_in_text", {"article_id": rec.get("article_id"), "event_id": eid, "evidence": str(evd)[:200]})
            else:
                ev_issue_stats["event_evidence_missing"] += 1
                ex("event_evidence_missing", {"article_id": rec.get("article_id"), "event_id": eid})

            if schema_ok and ev_ok:
                n_event_all_ok += 1

        for rel in relations:
            if not isinstance(rel, dict):
                rel_issue_stats["relation_not_dict"] += 1
                continue
            n_rels += 1
            rel_issues = check_relation_rules(rel, event_ids, id_to_name)
            for it in rel_issues:
                rel_issue_stats[it] += 1
                ex(it, {"article_id": rec.get("article_id"), "relation": rel})

            rel_evd = rel.get("evidence")
            ev_in = False
            if not is_missing(rel_evd) and body and len(str(rel_evd).strip()) <= MAX_EVIDENCE_CHARS:
                if evidence_in_body(str(rel_evd), body):
                    ev_in = True
                    n_rel_evidence_ok += 1
            if not ev_in and not is_missing(rel_evd) and body:
                if len(str(rel_evd).strip()) <= MAX_EVIDENCE_CHARS:
                    rel_issue_stats["rel_evidence_not_in_text"] += 1
                    ex("rel_evidence_not_in_text", {"article_id": rec.get("article_id"), "relation": rel})
            elif is_missing(rel_evd):
                pass
            elif not body:
                rel_issue_stats["rel_evidence_no_body"] += 1

            id_ok = not any(
                x in rel_issues
                for x in (
                    "rel_missing_source_event_id",
                    "rel_missing_target_event_id",
                    "rel_source_id_unknown",
                    "rel_target_id_unknown",
                    "rel_missing_source_event_name",
                    "rel_missing_target_event_name",
                    "rel_source_name_mismatch",
                    "rel_target_name_mismatch",
                    "rel_relation_type_bad",
                    "rel_evidence_missing",
                    "rel_evidence_too_long",
                )
            )
            if id_ok:
                n_rel_id_name_ok += 1
            if id_ok and ev_in:
                n_rel_all_ok += 1

    def rate(num: int, den: int) -> float:
        return round(num / max(1, den), 6)

    return {
        "counts": {
            "articles": n_articles,
            "events": n_events,
            "relations": n_rels,
        },
        "events": {
            "evidence_in_text_rate": rate(n_evidence_ok, n_events),
            "schema_ok_rate": rate(n_schema_ok, n_events),
            "evidence_and_schema_ok_rate": rate(n_event_all_ok, n_events),
            "issue_counts": dict(ev_issue_stats.most_common()),
        },
        "relations": {
            "id_and_names_ok_rate": rate(n_rel_id_name_ok, n_rels),
            "evidence_in_text_rate": rate(n_rel_evidence_ok, n_rels),
            "id_names_and_evidence_ok_rate": rate(n_rel_all_ok, n_rels),
            "issue_counts": dict(rel_issue_stats.most_common()),
        },
        "examples": {k: v for k, v in examples.items() if v},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_json", required=True)
    ap.add_argument("--articles_csv", default=None)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--max_examples", type=int, default=15)
    args = ap.parse_args()

    records = load_results_json(args.results_json)
    idx = load_articles_csv_index(args.articles_csv) if args.articles_csv else None
    summary = evaluate(records, articles_index=idx, max_examples=args.max_examples)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(text, encoding="utf-8")
        print(f"Saved → {args.out_json}")
    else:
        print(text)


if __name__ == "__main__":
    main()
