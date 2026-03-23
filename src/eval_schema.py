import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


def _norm_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[“”\"'‘’]", "", s)
    s = re.sub(r"[，,。．\.！!？?\-—_；;：:\(\)\[\]\{\}<>《》]", "", s)
    return s


def _parse_yyyy_mm_dd(s: Any) -> Optional[date]:
    if not isinstance(s, str) or not DATE_RE.match(s.strip()):
        return None
    try:
        return datetime.strptime(s.strip(), "%Y-%m-%d").date()
    except Exception:
        return None


def load_allowed_event_types(entities_by_type_json: str) -> List[str]:
    with open(entities_by_type_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError("entities_by_type.json must be a dict {event_type: [entities...]}")
    return sorted([k for k in obj.keys() if isinstance(k, str) and k.strip()])


def load_allowed_relation_types(graph_database_export_xlsx: str) -> List[str]:
    try:
        import pandas as pd  # type: ignore
    except Exception as e:
        raise RuntimeError("需要 pandas 才能读取 xlsx：请先安装 pandas + openpyxl") from e
    df = pd.read_excel(graph_database_export_xlsx)
    col = None
    for c in ["relation_1", "relation", "predicate", "rel", "关系"]:
        if c in df.columns:
            col = c
            break
    if col is None:
        raise ValueError(f"xlsx 未找到关系列。现有列：{list(df.columns)}")
    rels = []
    for v in df[col].tolist():
        if isinstance(v, str):
            vv = v.strip()
            if vv:
                rels.append(vv)
    return sorted(set(rels))


def load_results_json(results_json: str) -> List[Dict[str, Any]]:
    with open(results_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "records" in obj and isinstance(obj["records"], list):
        return obj["records"]
    if isinstance(obj, dict) and "results" in obj and isinstance(obj["results"], list):
        return obj["results"]
    if isinstance(obj, list):
        return obj
    raise ValueError("results.json 结构不符合预期：应为 list[record] 或 {records/results:[...]}")


def _pick_first(d: Dict[str, Any], keys: Sequence[str]) -> Optional[Any]:
    for k in keys:
        if k in d:
            return d.get(k)
    return None


def load_articles_csv_index(articles_csv: str) -> Dict[str, str]:
    """
    不默认调用。仅在你显式传 --articles_csv 时读取（文件很大）。
    输出: {article_id: full_text}
    """
    path = Path(articles_csv)
    if not path.exists():
        raise FileNotFoundError(str(path))

    idx: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("articles_cleaned.csv 无表头")
        id_candidates = ["article_id", "id", "doc_id"]
        text_candidates = ["content", "text", "article", "body", "full_text"]

        for row in reader:
            aid = _pick_first(row, id_candidates)
            txt = _pick_first(row, text_candidates)
            if aid is None or txt is None:
                continue
            aid = str(aid).strip()
            if not aid:
                continue
            idx[aid] = str(txt)
    return idx


@dataclass
class EventIssue:
    issue: str
    detail: str


def evaluate_records(
    records: List[Dict[str, Any]],
    allowed_event_types: Sequence[str],
    allowed_relation_types: Sequence[str],
    *,
    articles_index: Optional[Dict[str, str]] = None,
    rng_seed: int = 2026,
    sample_evidence_k: int = 50,
    min_event_text_len: int = 5,
    max_event_text_len: int = 220,
    max_examples_per_issue: int = 20,
    future_days_tolerance: int = 30,
) -> Dict[str, Any]:
    rng = random.Random(rng_seed)
    allowed_event_types_set = set(allowed_event_types)
    allowed_relation_types_set = set(allowed_relation_types)

    today = date.today()
    min_date = date(1900, 1, 1)
    max_date = today.fromordinal(today.toordinal() + int(future_days_tolerance))

    record_count = len(records)
    event_count = 0
    relation_count = 0

    field_missing = Counter()
    field_type_errors = Counter()
    field_norm_errors = Counter()

    suspicious_stats = Counter()
    issue_examples: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    def add_example(issue: str, payload: Dict[str, Any]) -> None:
        if len(issue_examples[issue]) < max_examples_per_issue:
            issue_examples[issue].append(payload)

    def score_add(issue: str, payload: Dict[str, Any], score: int = 1) -> None:
        suspicious_stats[issue] += 1
        add_example(issue, payload)
        payload["_score"] = payload.get("_score", 0) + score

    per_article_event_ids: Dict[str, set] = defaultdict(set)
    per_article_event_ids_dup: Dict[str, set] = defaultdict(set)
    per_article_events_norm_text: Dict[str, Counter] = defaultdict(Counter)

    events_by_article: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)

    # first pass: collect events & basic checks
    for rec in records:
        extraction = (rec or {}).get("extraction") or {}
        events = extraction.get("events") or []
        if not isinstance(events, list):
            continue
        for ev in events:
            if not isinstance(ev, dict):
                continue
            event_count += 1
            aid = ev.get("article_id") or rec.get("article_id")
            aid = None if _is_missing(aid) else str(aid)
            eid = ev.get("event_id")
            eid = None if _is_missing(eid) else str(eid)
            if aid and eid:
                if eid in per_article_event_ids[aid]:
                    per_article_event_ids_dup[aid].add(eid)
                    score_add(
                        "dup_event_id_in_article",
                        {"article_id": aid, "event_id": eid, "event": ev},
                        score=3,
                    )
                per_article_event_ids[aid].add(eid)
                events_by_article[aid][eid] = ev

            # field completeness
            for k in ["event_text", "event_type", "city", "date_from", "date_to", "location", "article_id"]:
                if _is_missing(ev.get(k)):
                    field_missing[k] += 1

            # event_type closed set
            et = ev.get("event_type")
            if not _is_missing(et):
                et = str(et).strip()
                if et not in allowed_event_types_set:
                    field_norm_errors["event_type_out_of_set"] += 1
                    score_add(
                        "event_type_out_of_set",
                        {"article_id": aid, "event_id": eid, "event_type": et, "event": ev},
                        score=2,
                    )

            # date format + order
            d1s = ev.get("date_from")
            d2s = ev.get("date_to")
            d1 = _parse_yyyy_mm_dd(d1s)
            d2 = _parse_yyyy_mm_dd(d2s)
            if not _is_missing(d1s) and d1 is None:
                field_norm_errors["date_from_bad_format"] += 1
                score_add("date_from_bad_format", {"article_id": aid, "event_id": eid, "date_from": d1s, "event": ev}, score=2)
            if not _is_missing(d2s) and d2 is None:
                field_norm_errors["date_to_bad_format"] += 1
                score_add("date_to_bad_format", {"article_id": aid, "event_id": eid, "date_to": d2s, "event": ev}, score=2)
            if d1 and d2 and d1 > d2:
                field_norm_errors["date_order_bad"] += 1
                score_add("date_order_bad", {"article_id": aid, "event_id": eid, "date_from": d1s, "date_to": d2s, "event": ev}, score=3)
            for dn, d in [("date_from", d1), ("date_to", d2)]:
                if d and (d < min_date or d > max_date):
                    field_norm_errors[f"{dn}_out_of_range"] += 1
                    score_add(f"{dn}_out_of_range", {"article_id": aid, "event_id": eid, dn: ev.get(dn), "event": ev}, score=2)

            # location type
            loc = ev.get("location")
            if not _is_missing(loc) and not isinstance(loc, list):
                field_type_errors["location_not_list"] += 1
                score_add("location_not_list", {"article_id": aid, "event_id": eid, "location": loc, "event": ev}, score=2)
            if isinstance(loc, list) and len(loc) == 0:
                suspicious_stats["location_empty_list"] += 1
                add_example("location_empty_list", {"article_id": aid, "event_id": eid, "event": ev})

            # city null rate
            if _is_missing(ev.get("city")):
                suspicious_stats["city_missing"] += 1

            # time_uncertain proxy
            attrs = ev.get("attributes") if isinstance(ev.get("attributes"), dict) else {}
            tu = ev.get("time_uncertain", None)
            if tu is None and isinstance(attrs, dict):
                tu = attrs.get("time_uncertain", None)
            if tu is True or (isinstance(tu, str) and tu.lower() == "true"):
                suspicious_stats["time_uncertain_true"] += 1

            # severity constraint (in attributes)
            sev = None
            if isinstance(attrs, dict):
                sev = attrs.get("severity", None)
            if not _is_missing(sev):
                if str(sev).strip() not in {"mild", "moderate", "severe"}:
                    field_norm_errors["severity_out_of_set"] += 1
                    score_add(
                        "severity_out_of_set",
                        {"article_id": aid, "event_id": eid, "severity": sev, "event": ev},
                        score=2,
                    )

            # event_text length + duplicates
            etext = ev.get("event_text")
            if isinstance(etext, str):
                tlen = len(etext.strip())
                if tlen < min_event_text_len:
                    suspicious_stats["event_text_too_short"] += 1
                    add_example("event_text_too_short", {"article_id": aid, "event_id": eid, "len": tlen, "event_text": etext, "event": ev})
                if tlen > max_event_text_len:
                    suspicious_stats["event_text_too_long"] += 1
                    add_example("event_text_too_long", {"article_id": aid, "event_id": eid, "len": tlen, "event_text": etext[:300], "event": ev})

                if aid:
                    n = _norm_text(etext)
                    per_article_events_norm_text[aid][n] += 1

            # cross-level article_id consistency (if both exist)
            top_aid = rec.get("article_id")
            if aid and not _is_missing(top_aid) and str(top_aid).strip() != aid:
                field_norm_errors["event_article_id_mismatch"] += 1
                score_add(
                    "event_article_id_mismatch",
                    {"record_article_id": str(top_aid).strip(), "event_article_id": aid, "event_id": eid, "event": ev},
                    score=2,
                )

    # duplicate event_text within article
    for aid, cnt in per_article_events_norm_text.items():
        dups = sum(v for v in cnt.values() if v >= 2)
        if dups > 0:
            suspicious_stats["dup_event_text_norm_in_article"] += 1
            add_example(
                "dup_event_text_norm_in_article",
                {"article_id": aid, "dup_norm_text_count": int(dups), "top_norm_texts": cnt.most_common(5)},
            )

    # second pass: relations checks
    for rec in records:
        extraction = (rec or {}).get("extraction") or {}
        rels = extraction.get("relations") or []
        if not isinstance(rels, list):
            continue
        aid_top = rec.get("article_id")
        aid_top = None if _is_missing(aid_top) else str(aid_top).strip()
        for r in rels:
            if not isinstance(r, dict):
                continue
            relation_count += 1
            src = r.get("source_event_id")
            tgt = r.get("target_event_id")
            rel_type = r.get("relation_type")

            src = None if _is_missing(src) else str(src)
            tgt = None if _is_missing(tgt) else str(tgt)
            rel_type_s = None if _is_missing(rel_type) else str(rel_type).strip()

            if rel_type_s and rel_type_s not in allowed_relation_types_set:
                field_norm_errors["relation_type_out_of_set"] += 1
                add_example("relation_type_out_of_set", {"article_id": aid_top, "relation": r})

            if src is None or tgt is None:
                field_missing["relation_source_or_target_missing"] += 1
                add_example("relation_source_or_target_missing", {"article_id": aid_top, "relation": r})
                continue

            if src == tgt:
                suspicious_stats["relation_self_loop"] += 1
                add_example("relation_self_loop", {"article_id": aid_top, "relation": r})

            if aid_top:
                evmap = events_by_article.get(aid_top, {})
                if src not in evmap:
                    suspicious_stats["relation_src_not_found"] += 1
                    add_example("relation_src_not_found", {"article_id": aid_top, "relation": r})
                if tgt not in evmap:
                    suspicious_stats["relation_tgt_not_found"] += 1
                    add_example("relation_tgt_not_found", {"article_id": aid_top, "relation": r})

    # evidence / anchor checks (optional)
    evidence_checks_done = 0
    evidence_anchor_fail = 0
    event_text_anchor_fail = 0
    city_anchor_fail = 0
    location_anchor_fail = 0
    anchor_total = 0

    if articles_index is not None:
        # sample events to reduce cost
        all_events: List[Tuple[str, str, Dict[str, Any]]] = []
        for aid, evmap in events_by_article.items():
            for eid, ev in evmap.items():
                all_events.append((aid, eid, ev))
        rng.shuffle(all_events)
        sample = all_events[: max(0, int(sample_evidence_k))]

        for aid, eid, ev in sample:
            text = articles_index.get(aid)
            if not isinstance(text, str) or not text.strip():
                continue
            text_n = _norm_text(text)

            # event_text anchor
            etext = ev.get("event_text")
            if isinstance(etext, str) and etext.strip():
                anchor_total += 1
                if _norm_text(etext) not in text_n:
                    event_text_anchor_fail += 1
                    add_example("event_text_not_in_article", {"article_id": aid, "event_id": eid, "event_text": etext[:250]})

            # city anchor
            city = ev.get("city")
            if isinstance(city, str) and city.strip():
                anchor_total += 1
                if _norm_text(city) not in text_n:
                    city_anchor_fail += 1
                    add_example("city_not_in_article", {"article_id": aid, "event_id": eid, "city": city})

            # location anchors
            loc = ev.get("location")
            if isinstance(loc, list):
                for loc_item in loc[:10]:
                    if isinstance(loc_item, str) and loc_item.strip():
                        anchor_total += 1
                        if _norm_text(loc_item) not in text_n:
                            location_anchor_fail += 1
                            add_example("location_item_not_in_article", {"article_id": aid, "event_id": eid, "location_item": loc_item})

            # evidence sentence anchor (if exists)
            evidences = ev.get("evidence_sentences")
            if isinstance(evidences, list) and evidences:
                rng.shuffle(evidences)
                for es in evidences[:3]:
                    if not isinstance(es, str) or not es.strip():
                        continue
                    evidence_checks_done += 1
                    if _norm_text(es) not in text_n:
                        evidence_anchor_fail += 1
                        add_example(
                            "evidence_sentence_not_in_article",
                            {"article_id": aid, "event_id": eid, "evidence_sentence": es[:300]},
                        )

    # build suspicious top examples (cross-issue)
    suspicious_items: List[Dict[str, Any]] = []
    for issue, examples in issue_examples.items():
        for ex in examples:
            ex2 = dict(ex)
            ex2["_issue"] = issue
            ex2["_score"] = ex2.get("_score", 1)
            suspicious_items.append(ex2)
    suspicious_items.sort(key=lambda x: int(x.get("_score", 0)), reverse=True)

    # per-article stats
    per_article_stats = []
    for aid, evset in per_article_event_ids.items():
        n_ev = len(evset)
        n_dup_ids = len(per_article_event_ids_dup.get(aid, set()))
        n_rel = 0
        per_article_stats.append({"article_id": aid, "n_events": n_ev, "n_dup_event_ids": n_dup_ids, "n_relations": n_rel})
    per_article_stats.sort(key=lambda x: (x["n_dup_event_ids"], x["n_events"]), reverse=True)

    summary = {
        "counts": {
            "records": record_count,
            "events": event_count,
            "relations": relation_count,
            "articles_with_events": len(per_article_event_ids),
        },
        "allowed_sets": {
            "event_types": list(allowed_event_types),
            "relation_types_size": len(allowed_relation_types_set),
        },
        "missing": {
            "by_field": dict(field_missing),
            "rates": {k: (v / max(1, event_count)) for k, v in field_missing.items() if k != "relation_source_or_target_missing"}
            | {"relation_source_or_target_missing": (field_missing.get("relation_source_or_target_missing", 0) / max(1, relation_count))},
        },
        "type_errors": dict(field_type_errors),
        "norm_errors": dict(field_norm_errors),
        "suspicious_stats": dict(suspicious_stats),
        "anchor_checks": {
            "enabled": articles_index is not None,
            "sample_events": int(sample_evidence_k) if articles_index is not None else 0,
            "anchor_total": int(anchor_total),
            "event_text_not_in_article": int(event_text_anchor_fail),
            "city_not_in_article": int(city_anchor_fail),
            "location_item_not_in_article": int(location_anchor_fail),
            "evidence_checks_done": int(evidence_checks_done),
            "evidence_sentence_not_in_article": int(evidence_anchor_fail),
        },
        "top_examples": {
            "by_issue": {k: v[:max_examples_per_issue] for k, v in issue_examples.items()},
            "top_suspicious": suspicious_items[:20],
        },
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_json", required=True, help="例如 CityResilence/results/run1/results.json")
    ap.add_argument("--entities_by_type_json", default=str(Path(__file__).resolve().parents[1] / "data" / "entities_by_type.json"))
    ap.add_argument("--graph_database_export_xlsx", default=str(Path(__file__).resolve().parents[1] / "data" / "graph_database_export.xlsx"))
    ap.add_argument("--articles_csv", default=None, help="可选：articles_cleaned.csv 路径；传了才会读（文件很大）")
    ap.add_argument("--out_json", default=None, help="可选：输出 summary.json 路径；不传则打印到 stdout")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--sample_evidence_k", type=int, default=50)
    ap.add_argument("--min_event_text_len", type=int, default=5)
    ap.add_argument("--max_event_text_len", type=int, default=220)
    args = ap.parse_args()

    allowed_event_types = load_allowed_event_types(args.entities_by_type_json)
    allowed_relation_types = load_allowed_relation_types(args.graph_database_export_xlsx)
    records = load_results_json(args.results_json)

    articles_index = None
    if args.articles_csv:
        articles_index = load_articles_csv_index(args.articles_csv)

    summary = evaluate_records(
        records,
        allowed_event_types,
        allowed_relation_types,
        articles_index=articles_index,
        rng_seed=args.seed,
        sample_evidence_k=args.sample_evidence_k,
        min_event_text_len=args.min_event_text_len,
        max_event_text_len=args.max_event_text_len,
    )

    out = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(out, encoding="utf-8")
    else:
        print(out)


if __name__ == "__main__":
    main()

