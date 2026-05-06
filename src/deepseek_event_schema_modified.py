#!/usr/bin/env python
"""
阶段 B：气候事件图谱抽取（两阶段最小工程版）
- Stage 1: DeepSeek Chat 抽取事件节点：Driver / Modulator / Hazard / Impact
- Stage 2: 程序生成候选 relation pair，再用 DeepSeek Chat 判断关系
- 不做高置信规则关系抽取
- 不新增 Response 类型；人为处置/救援/清理/疏导/送医等不作为事件节点
- 输出 events、candidate_pairs、relations、invalid_relations、validation_errors，方便后续 DBSCAN 和回溯
"""
import argparse
import os
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import pandas as pd


# ====== 配置 ======
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
HTTP_PROXY = os.environ.get("HTTP_PROXY", "") or os.environ.get("http_proxy", "")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

ALLOWED_EVENT_TYPES = {"Driver", "Modulator", "Hazard", "Impact"}
ALLOWED_RELATION_TYPES = {"引发", "加剧", "削弱", "增强"}
MAX_EVIDENCE_CHARS = 120
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ====== DeepSeek 请求 ======
def _httpx_client(proxy: Optional[str]) -> httpx.Client:
    if not proxy:
        return httpx.Client()
    try:
        return httpx.Client(proxies=proxy)
    except TypeError:
        return httpx.Client(proxy=proxy)


def query_deepseek(
    prompt: str,
    api_key: str,
    *,
    model: str,
    base_url: str,
    proxy: Optional[str],
    timeout: float = 120,
    max_tokens: int = 8000,
) -> Tuple[str, Dict[str, int]]:
    """返回 (content, usage)。"""
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 为空：请先设置环境变量 DEEPSEEK_API_KEY。")

    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
    }

    with _httpx_client(proxy) as client:
        resp = client.post(
            url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()

    usage = data.get("usage") or {}
    out = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }
    return data["choices"][0]["message"]["content"], out


def add_usage(total: Dict[str, int], usage: Dict[str, int]) -> None:
    for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        total[k] = int(total.get(k, 0)) + int(usage.get(k, 0))


# ====== 文本预处理与定位 ======
def split_paragraphs(text: str) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n+", text or "") if p.strip()]
    return paragraphs if paragraphs else ([text.strip()] if text and text.strip() else [])


def split_sentences_with_spans(text: str) -> List[Dict[str, Any]]:
    """简单中文/英文切句，返回 sentence_id、paragraph_id、char_start、char_end。"""
    rows: List[Dict[str, Any]] = []
    if not text:
        return rows

    paragraph_matches = list(re.finditer(r"[^\n]+", text))
    sid = 1
    for pid, pm in enumerate(paragraph_matches, start=1):
        para = pm.group(0).strip()
        if not para:
            continue
        base = pm.start()
        # 保留标点在句子内
        for sm in re.finditer(r"[^。！？!?；;]+[。！？!?；;]?", para):
            sent = sm.group(0).strip()
            if not sent:
                continue
            start = base + sm.start() + (len(sm.group(0)) - len(sm.group(0).lstrip()))
            end = base + sm.end()
            rows.append(
                {
                    "sentence_id": sid,
                    "paragraph_id": pid,
                    "text": sent,
                    "char_start": start,
                    "char_end": end,
                }
            )
            sid += 1
    return rows


def format_numbered_text(sentences: List[Dict[str, Any]]) -> str:
    return "\n".join(
        f"[P{s['paragraph_id']}-S{s['sentence_id']}] {s['text']}" for s in sentences
    )


def find_span(text: str, evidence: str) -> Tuple[Optional[int], Optional[int]]:
    if not text or not evidence:
        return None, None
    idx = text.find(evidence)
    if idx >= 0:
        return idx, idx + len(evidence)
    # 容错：去空白后找不到时，尝试用 evidence 的前后片段定位
    compact = re.sub(r"\s+", "", evidence)
    if len(compact) >= 8:
        head = compact[:8]
        tail = compact[-8:]
        text_compact = re.sub(r"\s+", "", text)
        h = text_compact.find(head)
        t = text_compact.find(tail, h + len(head)) if h >= 0 else -1
        if h >= 0 and t >= 0:
            return None, None
    return None, None


def locate_sentence_by_evidence(
    evidence: str,
    sentences: List[Dict[str, Any]],
) -> Tuple[Optional[int], Optional[int]]:
    if not evidence:
        return None, None
    for s in sentences:
        if evidence in s["text"] or s["text"] in evidence:
            return int(s["sentence_id"]), int(s["paragraph_id"])
    # fallback：用较短公共片段粗略匹配
    ev = re.sub(r"\s+", "", evidence)
    for s in sentences:
        st = re.sub(r"\s+", "", s["text"])
        if len(ev) >= 10 and (ev[:10] in st or st[:10] in ev):
            return int(s["sentence_id"]), int(s["paragraph_id"])
    return None, None


# ====== 词表参考 ======
def build_taxonomy_reference_text(taxonomy: Optional[Dict[str, List[str]]], limit_per_type: int = 120) -> str:
    if not taxonomy:
        return ""
    parts = []
    for t in ["Driver", "Modulator", "Hazard", "Impact"]:
        items = taxonomy.get(t, [])
        if not items:
            continue
        uniq_items = []
        seen = set()
        for x in items:
            x = str(x).strip()
            if not x or x in seen:
                continue
            seen.add(x)
            uniq_items.append(x)
            if len(uniq_items) >= limit_per_type:
                break
        if uniq_items:
            parts.append(f"- {t}: {'、'.join(uniq_items)}")
    if not parts:
        return ""
    return "\n事件命名参考词表（用于语义对齐，非硬匹配）：\n" + "\n".join(parts) + "\n"


# ====== Prompt ======
def generate_event_extraction_prompt(
    news_article: str,
    numbered_article: str,
    article_id: str = "",
    publish_date: str = "",
    taxonomy_reference_text: str = "",
) -> str:
    publish_date_hint = (
        f"\n新闻发布日期：{publish_date}\n"
        "若正文中时间模糊如'近期/近日/当天'，可使用该日期作为 time_text 的参考，但不要虚构正文没有的信息。\n"
        if publish_date
        else ""
    )
    tax = ((taxonomy_reference_text or "").strip() + "\n\n") if (taxonomy_reference_text or "").strip() else ""
    aid_esc = json.dumps(article_id or "", ensure_ascii=False)[1:-1]
    static = """
你是气候灾害事件知识图谱的事件抽取系统。

任务：
从新闻文本中抽取“已经发生或正在发生”的气候灾害相关事件。

事件类型只能是：
1. Driver：直接驱动因素，如暴雨、强降水、台风、强风、冷空气、高温等。
2. Modulator：调节因素，如排水系统堵塞、地形、土壤饱和、海温异常等。
3. Hazard：灾害或危险现象，如积水、内涝、洪水、滑坡、树木倒伏、热浪、干旱等。
4. Impact：灾害造成的负面影响，如交通受阻、车辆受损、人员伤亡、房屋倒塌、农作物受灾等。

重要过滤规则：
1. 只抽取原文明确表示“已经发生 / 正在发生”的事件。
2. 不抽取未来预测、风险提示、防御建议、可能发生但尚未发生的事件。
3. 出现以下表达时，其后内容通常不抽取为事件：
   “需注意防御”“注意防范”“预计”“将出现”“可能出现”“风险较高”“需警惕”“请做好防范”“防御……及其引发的次生灾害”。
4. 如果句子只是提醒防御某类天气或次生灾害，而没有说明该事件已经发生，不要抽取。
5. 不抽取处置、救援、治理、清理、疏导、送医、巡逻、值守等人为应对行为。
6. 同类事件发生在不同地点，应拆分为不同事件。
7. event_text 使用简洁核心名词短语，如“暴雨”“积水”“交通受阻”。
8. event_name 可更具体，如“南沙区东涌镇大暴雨”“花都区梯面镇9级阵风”。
9. evidence 必须来自原文，长度不超过 120 个中文字符。
10. evidence 必须能证明该事件已经发生或正在发生。
11. 输出严格 JSON，不要输出解释，不要输出 markdown。

输出格式：
{
  "events": [
    {
      "event_id": "E01",
      "event_text": "",
      "event_name": "",
      "event_type": "Driver | Modulator | Hazard | Impact",
      "city": "",
      "date_from": "YYYY-MM-DD",
      "date_to": "YYYY-MM-DD",
      "location": [],
      "attributes": {
        "severity": "",
        "time_uncertain": false
      },
      "article_id": "{article_id}",
      "time_text": "",
      "location_text": "",
      "evidence": "",
      "sentence_id": 0,
      "paragraph_id": 0,
      "char_start": -1,
      "char_end": -1,
      "confidence": 0.0
    }
  ]
}

字段要求：
- event_type 只能是 Driver、Modulator、Hazard、Impact。
- city 填地级市，如“广州”“深圳”；无法判断填 null。
- date_from/date_to 使用 YYYY-MM-DD。
- location 按行政层级填写，如 ["广东省", "广州市", "南沙区", "东涌镇"]；无明确地点填 []。
- attributes.severity 只能是 "mild"、"moderate"、"severe" 或 ""。
- attributes.time_uncertain 表示时间是否由发布日期回填或存在不确定性。
- 如果正文时间模糊但提供了新闻发布日期，则用新闻发布日期填 date_from/date_to，并设置 time_uncertain=true。

新闻发布日期：
{publish_date}

新闻文本：
{article_text}
""".strip()
    return (
        static.replace("__AID__", aid_esc)
        + publish_date_hint
        + tax
        + "新闻文本：\n"
        + news_article
        + "\n\n新闻文本（句编号，供 sentence_id / paragraph_id）：\n"
        + numbered_article
    ).strip()


def generate_relation_extraction_prompt(
    news_article: str,
    events: List[Dict[str, Any]],
    candidate_pairs: List[Dict[str, Any]],
) -> str:
    events_json = json.dumps(events, ensure_ascii=False, indent=2)
    candidate_pairs_json = json.dumps(candidate_pairs, ensure_ascii=False, indent=2)
    static = """
你是一个气候事件知识图谱的关系抽取系统。

任务：
给定新闻原文、已抽取事件列表和候选事件对。
请只从候选事件对中选择有明确证据支持的关系。
没有明确关系的候选对不要输出。
不得输出候选对之外的关系。

关系类型只能从以下四类中选择（使用下列中文取值）：

1. 引发：A 导致 B 发生，表示从无到有的因果关系。
2. 加剧：A 使 B 更严重、更频繁、更难缓解或影响范围更大。
3. 削弱：A 减弱 B 的发生、强度、持续性或影响。
4. 增强：A 增强 B 的强度、规模、持续性或发生概率，偏机制性增强。

判断规则：
1. 只输出 relations 数组中有证据的边；无证据的候选对不要出现在输出中。
2. 不得新增事件。
3. 不得修改 event_id；边的端点必须使用 source_event_id / target_event_id（与事件列表中的 event_id 一致），并同时填写 source_event_name / target_event_name 便于核对。
4. 不要因为两个事件在文中相邻就推断因果。
5. 不要把不同地点的相似事件错误相连。
6. evidence 必须来自原文，且能够支持 source、target 和二者关系。
7. evidence 使用原文中的一句话或关键分句，长度不得超过 120 个中文字符。
8. 如果关系方向不确定，则不要输出该候选对。
9. 不输出人为处置行为相关关系，因为事件类型中不包含 Response。
10. 输出严格 JSON，不要输出解释文字，不要输出 markdown 代码块。

输出格式（仅包含有关系的边；可为空数组）：
{
  "relations": [
    {
      "source_event_id": "",
      "source_event_name": "",
      "target_event_id": "",
      "target_event_name": "",
      "relation_type": "引发 | 加剧 | 削弱 | 增强",
      "evidence": "",
      "confidence": 0.0
    }
  ]
}
""".strip()
    return (
        static
        + "\n\n新闻原文：\n"
        + news_article
        + "\n\n已抽取事件列表：\n"
        + events_json
        + "\n\n候选事件对：\n"
        + candidate_pairs_json
    ).strip()


# ====== JSON 解析 ======
def _try_load_json(text: str) -> Optional[Dict[str, Any]]:
    text = (text or "").strip()
    if not text:
        return None
    # 去掉常见 markdown 包裹
    text = re.sub(r"^```(?:json)?", "", text.strip(), flags=re.I).strip()
    text = re.sub(r"```$", "", text.strip()).strip()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


# ====== 事件标准化与校验 ======
def normalize_events(
    parsed_events: List[Dict[str, Any]],
    *,
    article_id: str,
    article_text: str,
    sentences: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    seen_ids = set()

    for idx, e in enumerate(parsed_events, start=1):
        if not isinstance(e, dict):
            continue
        event_type = str(e.get("event_type", "")).strip()
        if event_type not in ALLOWED_EVENT_TYPES:
            continue

        raw_id = str(e.get("event_id") or f"E{idx:02d}").strip()
        # 统一成 E01 风格，避免 E1/E001 混乱
        m = re.search(r"(\d+)", raw_id)
        local_id = f"E{int(m.group(1)):02d}" if m else f"E{idx:02d}"
        if local_id in seen_ids:
            local_id = f"E{idx:02d}"
        seen_ids.add(local_id)

        evidence = str(e.get("evidence", "")).strip()
        char_start, char_end = find_span(article_text, evidence)

        sent_id = e.get("sentence_id")
        para_id = e.get("paragraph_id")
        try:
            sent_id = int(sent_id) if sent_id is not None and str(sent_id).strip() != "" else None
        except Exception:
            sent_id = None
        try:
            para_id = int(para_id) if para_id is not None and str(para_id).strip() != "" else None
        except Exception:
            para_id = None
        if sent_id is None or para_id is None:
            sid2, pid2 = locate_sentence_by_evidence(evidence, sentences)
            sent_id = sent_id or sid2
            para_id = para_id or pid2

        try:
            conf = float(e.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))

        etext = str(e.get("event_text", "")).strip()
        ename = str(e.get("event_name", etext)).strip()
        if not etext and ename:
            etext = ename

        loc = e.get("location")
        if not isinstance(loc, list):
            loc = []
        else:
            loc = [str(x).strip() for x in loc if str(x).strip()]

        attrs = e.get("attributes")
        if not isinstance(attrs, dict):
            attrs = {}
        sev = attrs.get("severity", "")
        if sev is not None and not isinstance(sev, str):
            sev = str(sev)
        sev = (sev or "").strip()
        tu = attrs.get("time_uncertain", False)
        if isinstance(tu, str):
            tu = tu.strip().lower() in ("1", "true", "yes", "是")
        else:
            tu = bool(tu)

        city_raw = e.get("city")
        if city_raw is None or (isinstance(city_raw, str) and not city_raw.strip()):
            city_val = None
        else:
            city_val = str(city_raw).strip()

        df = str(e.get("date_from", "")).strip()
        dt = str(e.get("date_to", "")).strip()

        events.append(
            {
                "event_id": local_id,
                "event_text": etext,
                "event_name": ename,
                "event_type": event_type,
                "city": city_val,
                "date_from": df,
                "date_to": dt,
                "location": loc,
                "attributes": {"severity": sev, "time_uncertain": tu},
                "time_text": str(e.get("time_text", "")).strip(),
                "location_text": str(e.get("location_text", "")).strip(),
                "evidence": evidence,
                "sentence_id": sent_id,
                "paragraph_id": para_id,
                "char_start": char_start,
                "char_end": char_end,
                "confidence": conf,
                "article_id": article_id or str(e.get("article_id", "")).strip(),
            }
        )
    return events


def validate_event(event: Dict[str, Any], article_text: str) -> List[str]:
    errors = []
    if event.get("event_type") not in ALLOWED_EVENT_TYPES:
        errors.append("invalid_event_type")
    if not event.get("event_id"):
        errors.append("missing_event_id")
    if not (event.get("event_text") or "").strip():
        errors.append("missing_event_text")
    if not event.get("event_name"):
        errors.append("missing_event_name")
    loc = event.get("location")
    if not isinstance(loc, list):
        errors.append("location_not_list")
    attrs = event.get("attributes")
    if not isinstance(attrs, dict):
        errors.append("attributes_not_object")
    else:
        if "severity" not in attrs:
            errors.append("missing_severity")
        if "time_uncertain" not in attrs:
            errors.append("missing_time_uncertain")
    df = event.get("date_from")
    dt = event.get("date_to")
    dfs = str(df).strip() if df is not None else ""
    dts = str(dt).strip() if dt is not None else ""
    if not _DATE_RE.match(dfs):
        errors.append("invalid_date_from")
    if not _DATE_RE.match(dts):
        errors.append("invalid_date_to")
    try:
        cf = float(event.get("confidence", 0.0))
        if cf < 0 or cf > 1:
            errors.append("confidence_out_of_range")
    except Exception:
        errors.append("confidence_invalid")
    evidence = event.get("evidence", "")
    if not evidence or evidence not in article_text:
        errors.append("evidence_not_in_text")
    elif len(evidence) > MAX_EVIDENCE_CHARS:
        errors.append("evidence_too_long")
    return errors


# ====== 候选 pair 生成 ======
def relation_type_allowed(src_type: str, tgt_type: str) -> bool:
    return (
        (src_type == "Driver" and tgt_type in {"Driver", "Hazard", "Impact"})
        or (src_type == "Modulator" and tgt_type in {"Driver", "Hazard", "Impact"})
        or (src_type == "Hazard" and tgt_type in {"Hazard", "Impact"})
        or (src_type == "Impact" and tgt_type == "Impact")
    )


def generate_candidate_pairs(events: List[Dict[str, Any]], max_sentence_distance: int = 2) -> List[Dict[str, Any]]:
    pairs: List[Dict[str, Any]] = []
    seen = set()

    for src in events:
        for tgt in events:
            if src.get("event_id") == tgt.get("event_id"):
                continue

            src_type = src.get("event_type")
            tgt_type = tgt.get("event_type")
            if not relation_type_allowed(src_type, tgt_type):
                continue

            src_sent = src.get("sentence_id")
            tgt_sent = tgt.get("sentence_id")
            src_para = src.get("paragraph_id")
            tgt_para = tgt.get("paragraph_id")
            same_para = (
                isinstance(src_para, int)
                and isinstance(tgt_para, int)
                and src_para == tgt_para
            )
            if isinstance(src_sent, int) and isinstance(tgt_sent, int):
                if not same_para and abs(src_sent - tgt_sent) > max_sentence_distance:
                    continue

            key = (src.get("event_id"), tgt.get("event_id"))
            if key in seen:
                continue
            seen.add(key)

            pairs.append(
                {
                    "source_event_id": src.get("event_id"),
                    "source_event_name": src.get("event_name"),
                    "source_event_type": src.get("event_type"),
                    "target_event_id": tgt.get("event_id"),
                    "target_event_name": tgt.get("event_name"),
                    "target_event_type": tgt.get("event_type"),
                }
            )
    return pairs


def chunk_list(items: List[Dict[str, Any]], chunk_size: int) -> List[List[Dict[str, Any]]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


# ====== 关系标准化与校验 ======
def normalize_relations(raw_relations: List[Dict[str, Any]], events_by_id: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    relations: List[Dict[str, Any]] = []
    for r in raw_relations:
        if not isinstance(r, dict):
            continue
        src = str(r.get("source_event_id", "")).strip()
        tgt = str(r.get("target_event_id", "")).strip()
        rel_type = str(r.get("relation_type", "")).strip()
        if rel_type not in ALLOWED_RELATION_TYPES:
            continue
        try:
            conf = float(r.get("confidence", 0.0))
        except Exception:
            conf = 0.0
        conf = max(0.0, min(1.0, conf))
        relations.append(
            {
                "source_event_id": src,
                "source_event_name": events_by_id.get(src, {}).get("event_name", str(r.get("source_event_name", "")).strip()),
                "target_event_id": tgt,
                "target_event_name": events_by_id.get(tgt, {}).get("event_name", str(r.get("target_event_name", "")).strip()),
                "relation_type": rel_type,
                "evidence": str(r.get("evidence", "")).strip(),
                "confidence": conf,
            }
        )
    return relations


def validate_relation(relation: Dict[str, Any], events_by_id: Dict[str, Dict[str, Any]], article_text: str) -> List[str]:
    errors = []
    src = relation.get("source_event_id")
    tgt = relation.get("target_event_id")
    rel_type = relation.get("relation_type")

    if src not in events_by_id:
        errors.append("source_event_id_not_found")
    if tgt not in events_by_id:
        errors.append("target_event_id_not_found")
    if src == tgt:
        errors.append("self_loop")
    if rel_type not in ALLOWED_RELATION_TYPES:
        errors.append("invalid_relation_type")

    evidence = relation.get("evidence", "")
    if not evidence or evidence not in article_text:
        errors.append("evidence_not_in_text")
    elif len(evidence) > MAX_EVIDENCE_CHARS:
        errors.append("evidence_too_long")
    try:
        cf = float(relation.get("confidence", 0.0))
        if cf < 0 or cf > 1:
            errors.append("confidence_out_of_range")
    except Exception:
        errors.append("confidence_invalid")
    return errors


def deduplicate_relations(relations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    best: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for r in relations:
        key = (r.get("source_event_id", ""), r.get("relation_type", ""), r.get("target_event_id", ""))
        if key not in best or float(r.get("confidence", 0)) > float(best[key].get("confidence", 0)):
            best[key] = r
    out = []
    for idx, r in enumerate(best.values(), start=1):
        rr = dict(r)
        rr["relation_id"] = f"R{idx:02d}"
        out.append(rr)
    return out


# ====== 主抽取函数 ======
def extract_events_and_relations(
    news_article: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    proxy: Optional[str],
    article_id: str = "",
    publish_date: str = "",
    taxonomy_reference_text: str = "",
    max_sentence_distance: int = 2,
    relation_pair_chunk_size: int = 50, # chunk size影响api次数，搞大点
) -> Tuple[Dict[str, Any], Dict[str, int], List[Dict[str, Any]]]:
    """两阶段抽取，返回 (extraction, usage 汇总, 各轮 API 调用的 token 明细)。"""
    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    usage_rounds: List[Dict[str, Any]] = []
    round_index = 0
    sentences = split_sentences_with_spans(news_article)
    numbered_article = format_numbered_text(sentences)

    # Stage 1: event extraction
    event_prompt = generate_event_extraction_prompt(
        news_article=news_article,
        numbered_article=numbered_article,
        article_id=article_id,
        publish_date=publish_date,
        taxonomy_reference_text=taxonomy_reference_text,
    )
    event_text, usage = query_deepseek(event_prompt, api_key, model=model, base_url=base_url, proxy=proxy)
    add_usage(usage_total, usage)
    usage_rounds.append(
        {
            "round_index": round_index,
            "stage": "event_extraction",
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
    )
    round_index += 1
    event_parsed = _try_load_json(event_text)
    if event_parsed is None:
        raise ValueError(f"事件抽取输出不是可解析 JSON：\n{event_text[:2000]}")

    events = normalize_events(
        event_parsed.get("events", []),
        article_id=article_id,
        article_text=news_article,
        sentences=sentences,
    )
    event_validation_errors = []
    for e in events:
        errs = validate_event(e, news_article)
        if errs:
            event_validation_errors.append({"event_id": e.get("event_id"), "errors": errs})

    # 事件少于 2 个时，直接返回
    candidate_pairs = generate_candidate_pairs(events, max_sentence_distance=max_sentence_distance)
    events_by_id = {e["event_id"]: e for e in events}

    all_relation_judgments: List[Dict[str, Any]] = []
    invalid_relations: List[Dict[str, Any]] = []
    if candidate_pairs:
        # Stage 2: relation pair classification
        compact_events = [
            {
                "event_id": e["event_id"],
                "event_text": e.get("event_text", ""),
                "event_name": e["event_name"],
                "event_type": e["event_type"],
                "time_text": e.get("time_text", ""),
                "location_text": e.get("location_text", ""),
                "evidence": e.get("evidence", ""),
                "sentence_id": e.get("sentence_id"),
                "paragraph_id": e.get("paragraph_id"),
            }
            for e in events
        ]
        pair_chunks = chunk_list(candidate_pairs, relation_pair_chunk_size)
        n_rel_chunks = len(pair_chunks)
        for chunk_i, pair_chunk in enumerate(pair_chunks):
            rel_prompt = generate_relation_extraction_prompt(news_article, compact_events, pair_chunk)
            rel_text, usage = query_deepseek(rel_prompt, api_key, model=model, base_url=base_url, proxy=proxy)
            add_usage(usage_total, usage)
            usage_rounds.append(
                {
                    "round_index": round_index,
                    "stage": "relation_extraction",
                    "relation_chunk_index": chunk_i,
                    "relation_chunks_total": n_rel_chunks,
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                    "total_tokens": int(usage.get("total_tokens", 0)),
                }
            )
            round_index += 1
            rel_parsed = _try_load_json(rel_text)
            if rel_parsed is None:
                invalid_relations.append(
                    {
                        "_parse_error": "relation_json_parse_failed",
                        "_raw_snippet": rel_text[:500],
                    }
                )
                continue
            all_relation_judgments.extend(rel_parsed.get("relations", []))

    normalized_relation_judgments = normalize_relations(all_relation_judgments, events_by_id)

    valid_relation_judgments = []
    for r in normalized_relation_judgments:
        errs = validate_relation(r, events_by_id, news_article)
        if errs:
            invalid_relations.append({**r, "validation_errors": errs})
            continue
        valid_relation_judgments.append(r)

    final_relations = deduplicate_relations(valid_relation_judgments)

    extraction = {
        "events": events,
        "candidate_pairs": candidate_pairs,
        "relation_judgments": normalized_relation_judgments,
        "relations": final_relations,
        "invalid_relations": invalid_relations,
        "validation_errors": {
            "events": event_validation_errors,
        },
        "raw_stage_outputs": {
            "event_output": event_text,
        },
    }
    return extraction, usage_total, usage_rounds


def _article_usage_payload(usage: Dict[str, int], usage_rounds: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "rounds": usage_rounds,
    }


# ====== 批处理主程序 ======
def main(test: bool = False):
    root = Path(__file__).resolve().parent
    data_dir = root.parent / "data"
    if test:
        csv_path = (data_dir / "golden" / "articles.csv").resolve()
        results_dir = (data_dir / "golden").resolve()
        file_prefix = "test_"
    else:
        csv_path = (data_dir / "article" / "articles_deduped.csv").resolve()
        results_dir = (root.parent / "results").resolve()
        file_prefix = ""

    def out_name(name: str) -> str:
        return f"{file_prefix}{name}" if file_prefix else name

    taxonomy_path = (data_dir / "background" / "entities_by_type.json").resolve()

    df = pd.read_csv(csv_path)
    results_dir.mkdir(parents=True, exist_ok=True)

    taxonomy_reference_text = ""
    if taxonomy_path.exists():
        try:
            with open(taxonomy_path, "r", encoding="utf-8") as f:
                taxonomy_data = json.load(f)
            taxonomy_reference_text = build_taxonomy_reference_text(taxonomy_data, limit_per_type=120)
        except Exception as e:
            print(f"加载事件词表失败，继续无词表模式：{e}", flush=True)

    batch_size = int(os.environ.get("BATCH_SIZE", "100"))
    n_rows = len(df)

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    all_results: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    for batch_idx, start in enumerate(range(0, n_rows, batch_size), start=1):
        end = min(start + batch_size, n_rows)
        run_name = f"run{batch_idx}"
        run_dir = results_dir / out_name(run_name)
        run_dir.mkdir(parents=True, exist_ok=True)

        usage_txt_path = run_dir / out_name("usage_tokens.txt")
        run_json_path = run_dir / out_name("results.json")

        # 已完成批次：直接载入
        if usage_txt_path.exists() and run_json_path.exists():
            try:
                with open(run_json_path, "r", encoding="utf-8") as f:
                    prev_data = json.load(f)
                prev_results = prev_data.get("results", [])
                prev_stats = prev_data.get("run_stats", {})
                total_usage["prompt_tokens"] += int(prev_stats.get("total_prompt_tokens", 0))
                total_usage["completion_tokens"] += int(prev_stats.get("total_completion_tokens", 0))
                total_usage["total_tokens"] += int(prev_stats.get("total_tokens", 0))
            except Exception:
                print(f"{run_name} 已存在但结果无法加载，重新处理第 {start}~{end - 1} 条", flush=True)
                prev_results = None

            if prev_results is not None:
                failed_txt_path = run_dir / out_name("failed_indices.txt")
                fail_again_path = run_dir / out_name("failAgain.txt")

                if fail_again_path.exists():
                    print(f"{run_name} 已完成重试（failAgain.txt 已存在），直接载入", flush=True)
                    all_results.extend(prev_results)
                    continue

                if not failed_txt_path.exists():
                    print(f"{run_name} 已存在且无失败，跳过第 {start}~{end - 1} 条", flush=True)
                    all_results.extend(prev_results)
                    continue

                failed_indices = set()
                with open(failed_txt_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.isdigit():
                            failed_indices.add(int(line))

                if not failed_indices:
                    print(f"{run_name} failed_indices.txt 为空，跳过", flush=True)
                    all_results.extend(prev_results)
                    continue

                print(f"{run_name} 发现 {len(failed_indices)} 条失败，开始重试: {sorted(failed_indices)}", flush=True)
                results_by_idx = {r["index"]: r for r in prev_results if "index" in r}
                retry_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                still_failed = []

                for gidx in sorted(failed_indices):
                    row = df.iloc[gidx]
                    text = str(row["content"]) if pd.notna(row["content"]) else ""
                    raw_pt = row["publish_time"] if "publish_time" in row and pd.notna(row.get("publish_time")) else ""
                    publish_date = str(raw_pt).split()[0] if raw_pt and len(str(raw_pt).split()) > 0 else ""
                    article_id = f"A{gidx:03d}"
                    try:
                        out, usage, usage_rounds = extract_events_and_relations(
                            text,
                            api_key=DEEPSEEK_API_KEY,
                            base_url=DEEPSEEK_BASE_URL,
                            model=DEEPSEEK_MODEL,
                            proxy=HTTP_PROXY or None,
                            article_id=article_id,
                            publish_date=publish_date,
                            taxonomy_reference_text=taxonomy_reference_text,
                        )
                        add_usage(retry_usage, usage)
                        add_usage(total_usage, usage)
                        results_by_idx[gidx] = {
                            "index": int(gidx),
                            "article_id": article_id,
                            "publish_date": publish_date,
                            "content_preview": text[:100] if text else "",
                            "extraction": out,
                            "usage": _article_usage_payload(usage, usage_rounds),
                        }
                        print(f"{run_name} 重试: 第 {gidx} 条成功", flush=True)
                    except Exception as e:
                        print(f"{run_name} 重试: 第 {gidx} 条仍然失败：{e}", flush=True)
                        still_failed.append(gidx)

                updated_results = [results_by_idx[k] for k in sorted(results_by_idx)]
                prev_stats["total_prompt_tokens"] = int(prev_stats.get("total_prompt_tokens", 0)) + retry_usage["prompt_tokens"]
                prev_stats["total_completion_tokens"] = int(prev_stats.get("total_completion_tokens", 0)) + retry_usage["completion_tokens"]
                prev_stats["total_tokens"] = int(prev_stats.get("total_tokens", 0)) + retry_usage["total_tokens"]

                with open(run_json_path, "w", encoding="utf-8") as f:
                    json.dump({"run_id": run_name, "run_stats": prev_stats, "results": updated_results}, f, ensure_ascii=False, indent=2)
                with open(usage_txt_path, "w", encoding="utf-8") as f:
                    f.write(
                        f"batch_index: {prev_stats.get('batch_index', batch_idx)}\n"
                        f"start_index: {prev_stats.get('start_index', start)}\n"
                        f"end_index: {prev_stats.get('end_index', end)}\n"
                        f"total_prompt_tokens: {prev_stats['total_prompt_tokens']}\n"
                        f"total_completion_tokens: {prev_stats['total_completion_tokens']}\n"
                        f"total_tokens: {prev_stats['total_tokens']}\n"
                    )
                    f.write(
                        "# per_article: index\tarticle_id\tprompt_tokens\tcompletion_tokens\ttotal_tokens\tapi_rounds\n"
                    )
                    for r in updated_results:
                        u = r.get("usage") or {}
                        f.write(
                            f"{r.get('index', '')}\t{r.get('article_id', '')}\t"
                            f"{int(u.get('prompt_tokens', 0))}\t{int(u.get('completion_tokens', 0))}\t"
                            f"{int(u.get('total_tokens', 0))}\t{len(u.get('rounds', []))}\n"
                        )
                with open(fail_again_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(str(i) for i in sorted(still_failed)) if still_failed else "0")

                print(
                    f"{run_name} 重试完成：成功 {len(failed_indices) - len(still_failed)} 条，仍失败 {len(still_failed)} 条 → failAgain.txt",
                    flush=True,
                )
                all_results.extend(updated_results)
                continue

        batch_df = df.iloc[start:end]
        texts = batch_df["content"].fillna("").tolist()
        if "publish_time" in batch_df.columns:
            publish_dates = batch_df["publish_time"].fillna("").apply(
                lambda x: str(x).split()[0] if x and len(str(x).split()) > 0 else ""
            ).tolist()
        else:
            publish_dates = [""] * len(texts)

        batch_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        batch_results: List[Dict[str, Any]] = []
        batch_failed_indices: List[int] = []
        batch_t0 = time.perf_counter()

        for offset, text in enumerate(texts):
            global_idx = start + offset
            article_id = f"A{global_idx:03d}"
            publish_date = publish_dates[offset] if offset < len(publish_dates) else ""
            try:
                out, usage, usage_rounds = extract_events_and_relations(
                    text,
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL,
                    model=DEEPSEEK_MODEL,
                    proxy=HTTP_PROXY or None,
                    article_id=article_id,
                    publish_date=publish_date,
                    taxonomy_reference_text=taxonomy_reference_text,
                )
                add_usage(batch_usage, usage)
                add_usage(total_usage, usage)
                record = {
                    "index": int(global_idx),
                    "article_id": article_id,
                    "publish_date": publish_date,
                    "content_preview": text[:100] if text else "",
                    "extraction": out,
                    "usage": _article_usage_payload(usage, usage_rounds),
                }
                batch_results.append(record)
                all_results.append(record)
                if (offset + 1) % 10 == 0 or offset == 0:
                    print(
                        f"{run_name}: 已处理本批 {offset + 1}/{len(texts)} 条（全局 {global_idx + 1}/{n_rows}）",
                        flush=True,
                    )
            except Exception as e:
                print(f"{run_name}: 第 {global_idx} 条处理失败，跳过：{e}", flush=True)
                batch_failed_indices.append(global_idx)
                record = {
                    "index": int(global_idx),
                    "article_id": article_id,
                    "publish_date": publish_date,
                    "content_preview": text[:100] if text else "",
                    "error": str(e),
                }
                batch_results.append(record)
                all_results.append(record)

        batch_runtime_seconds = time.perf_counter() - batch_t0
        run_stats = {
            "batch_index": batch_idx,
            "start_index": start,
            "end_index": end,
            "total_runtime_seconds": round(batch_runtime_seconds, 2),
            "total_prompt_tokens": batch_usage["prompt_tokens"],
            "total_completion_tokens": batch_usage["completion_tokens"],
            "total_tokens": batch_usage["total_tokens"],
        }

        with open(run_json_path, "w", encoding="utf-8") as f:
            json.dump({"run_id": run_name, "run_stats": run_stats, "results": batch_results}, f, ensure_ascii=False, indent=2)

        with open(usage_txt_path, "w", encoding="utf-8") as f:
            f.write(
                f"batch_index: {batch_idx}\n"
                f"start_index: {start}\n"
                f"end_index: {end}\n"
                f"total_prompt_tokens: {batch_usage['prompt_tokens']}\n"
                f"total_completion_tokens: {batch_usage['completion_tokens']}\n"
                f"total_tokens: {batch_usage['total_tokens']}\n"
                f"total_runtime_seconds: {round(batch_runtime_seconds, 2)}\n"
            )
            f.write(
                "# per_article: index\tarticle_id\tprompt_tokens\tcompletion_tokens\ttotal_tokens\tapi_rounds\n"
            )
            for r in batch_results:
                u = r.get("usage") or {}
                f.write(
                    f"{r.get('index', '')}\t{r.get('article_id', '')}\t"
                    f"{int(u.get('prompt_tokens', 0))}\t{int(u.get('completion_tokens', 0))}\t"
                    f"{int(u.get('total_tokens', 0))}\t{len(u.get('rounds', []))}\n"
                )

        if batch_failed_indices:
            failed_path = run_dir / out_name("failed_indices.txt")
            with open(failed_path, "w", encoding="utf-8") as f:
                f.write("\n".join(str(i) for i in sorted(batch_failed_indices)))

        print(
            f"{run_name} 完成：第 {start}~{end - 1} 条，"
            f"tokens={batch_usage['total_tokens']} "
            f"(prompt={batch_usage['prompt_tokens']}, completion={batch_usage['completion_tokens']})，"
            f"runtime={round(batch_runtime_seconds, 2)}s",
            flush=True,
        )

    runtime_seconds = time.perf_counter() - t0
    global_run_stats = {
        "total_runtime_seconds": round(runtime_seconds, 2),
        "total_prompt_tokens": total_usage["prompt_tokens"],
        "total_completion_tokens": total_usage["completion_tokens"],
        "total_tokens": total_usage["total_tokens"],
    }

    output_path = (results_dir / out_name("deepseek_event_schema_results.json")).resolve()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"run_stats": global_run_stats, "results": all_results}, f, ensure_ascii=False, indent=2)

    print(f"\n结果汇总已保存到：{output_path}")
    print(f"成功：{sum(1 for r in all_results if 'extraction' in r)} 条，失败：{sum(1 for r in all_results if 'error' in r)} 条")
    print(
        f"run_stats: runtime={global_run_stats['total_runtime_seconds']}s, "
        f"tokens={global_run_stats['total_tokens']} "
        f"(prompt={global_run_stats['total_prompt_tokens']}, "
        f"completion={global_run_stats['total_completion_tokens']})"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="气候事件图谱抽取批处理")
    ap.add_argument(
        "--test",
        action="store_true",
        help="测试模式：读 data/golden/articles.csv，输出写入 data/golden/，文件名均带 test_ 前缀",
    )
    ns = ap.parse_args()
    main(test=ns.test)
