import re
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


def _clean_short_text(x: Any, max_len: int = 60) -> str:
    if _is_missing(x):
        return ""
    s = " ".join(str(x).split())
    return s[:max_len]


def _norm_rel_type(x: Any) -> str:
    if _is_missing(x):
        return ""
    return str(x).strip()


def _extract_old_event_num(event_id: Any) -> Optional[int]:
    if _is_missing(event_id):
        return None
    s = str(event_id).strip()
    m = re.fullmatch(r"[Ee](\d+)", s)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def normalize_event_graph(
    raw_extraction: Dict[str, Any],
    record_article_id: str,
    allowed_relation_types: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    events = raw_extraction.get("events") or []
    relations = raw_extraction.get("relations") or []
    if not isinstance(events, list):
        events = []
    if not isinstance(relations, list):
        relations = []

    allowed_rel_set = set(allowed_relation_types or [])

    stats: Dict[str, int] = {
        "raw_events": len(events),
        "raw_relations": len(relations),
        "events_kept": 0,
        "events_dropped_non_target": 0,
        "events_dropped_invalid": 0,
        "events_reindexed": 0,
        "relations_kept": 0,
        "relations_dropped_missing_endpoints": 0,
        "relations_dropped_unmapped": 0,
        "relations_dropped_self_loop": 0,
        "relations_dropped_type": 0,
        "relations_deduped": 0,
        "relations_evidence_empty": 0,
    }

    normalized_events: List[Dict[str, Any]] = []
    old_to_new: Dict[str, str] = {}
    index_to_new: Dict[int, str] = {}

    for idx, ev in enumerate(events):
        if not isinstance(ev, dict):
            stats["events_dropped_invalid"] += 1
            continue
        if ev.get("is_target_event") is not True:
            stats["events_dropped_non_target"] += 1
            continue
        new_id = f"E{len(normalized_events) + 1}"
        old_id = ev.get("event_id")
        if not _is_missing(old_id):
            old_to_new[str(old_id).strip()] = new_id
            old_num = _extract_old_event_num(old_id)
            if old_num is not None:
                index_to_new[old_num - 1] = new_id
        normalized_ev = dict(ev)
        normalized_ev["event_id"] = new_id
        normalized_ev["article_id"] = record_article_id
        normalized_events.append(normalized_ev)

    stats["events_kept"] = len(normalized_events)
    stats["events_reindexed"] = len(normalized_events)

    normalized_relations: List[Dict[str, Any]] = []
    seen_rel = set()

    for rel in relations:
        if not isinstance(rel, dict):
            stats["relations_dropped_missing_endpoints"] += 1
            continue
        src = rel.get("source_event_id")
        tgt = rel.get("target_event_id")
        if _is_missing(src) or _is_missing(tgt):
            stats["relations_dropped_missing_endpoints"] += 1
            continue
        src_s = str(src).strip()
        tgt_s = str(tgt).strip()

        mapped_src = old_to_new.get(src_s)
        mapped_tgt = old_to_new.get(tgt_s)

        if mapped_src is None:
            src_num = _extract_old_event_num(src_s)
            if src_num is not None:
                mapped_src = index_to_new.get(src_num - 1)
        if mapped_tgt is None:
            tgt_num = _extract_old_event_num(tgt_s)
            if tgt_num is not None:
                mapped_tgt = index_to_new.get(tgt_num - 1)

        if mapped_src is None or mapped_tgt is None:
            stats["relations_dropped_unmapped"] += 1
            continue
        if mapped_src == mapped_tgt:
            stats["relations_dropped_self_loop"] += 1
            continue

        rel_type = _norm_rel_type(rel.get("relation_type"))
        if allowed_rel_set and rel_type not in allowed_rel_set:
            stats["relations_dropped_type"] += 1
            continue

        evidence = _clean_short_text(rel.get("relation_evidence_quote"), max_len=80)
        if evidence == "":
            stats["relations_evidence_empty"] += 1

        reasoning = _clean_short_text(rel.get("relation_reasoning"), max_len=80)

        rel_norm = {
            "source_event_id": mapped_src,
            "relation_type": rel_type,
            "target_event_id": mapped_tgt,
            "relation_evidence_quote": evidence,
            "relation_reasoning": reasoning,
        }
        rel_key = (rel_norm["source_event_id"], rel_norm["relation_type"], rel_norm["target_event_id"])
        if rel_key in seen_rel:
            stats["relations_deduped"] += 1
            continue
        seen_rel.add(rel_key)
        normalized_relations.append(rel_norm)

    stats["relations_kept"] = len(normalized_relations)
    normalized = {"events": normalized_events, "relations": normalized_relations}
    return normalized, stats
