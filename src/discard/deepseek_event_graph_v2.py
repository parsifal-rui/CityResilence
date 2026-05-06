#!/usr/bin/env python
import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx
import pandas as pd

from normalize_event_graph import normalize_event_graph


DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
HTTP_PROXY = os.environ.get("HTTP_PROXY", "") or os.environ.get("http_proxy", "")


def _httpx_client(proxy: Optional[str]) -> httpx.Client:
    if not proxy:
        return httpx.Client()
    try:
        return httpx.Client(proxies=proxy)
    except TypeError:
        return httpx.Client(proxy=proxy)


def query_deepseek(prompt: str, api_key: str, *, model: str, base_url: str, proxy: Optional[str], timeout: float = 120) -> Tuple[str, Dict[str, int]]:
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 为空")
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 5000,
        "response_format": {"type": "json_object"},
    }
    with _httpx_client(proxy) as client:
        resp = client.post(
            url,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
    usage = data.get("usage") or {}
    usage_out = {
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
    }
    return data["choices"][0]["message"]["content"], usage_out


def _try_load_json(text: str) -> Optional[Dict[str, Any]]:
    t = text.strip()
    try:
        obj = json.loads(t)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", t)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None


def build_taxonomy_reference_text(taxonomy: Optional[Dict[str, List[str]]], limit_per_type: int = 80) -> str:
    if not taxonomy:
        return ""
    lines: List[str] = []
    for t in ["Driver", "Modulator", "Hazard", "Impact"]:
        vals = taxonomy.get(t, [])
        uniq: List[str] = []
        seen = set()
        for v in vals:
            s = str(v).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            uniq.append(s)
            if len(uniq) >= limit_per_type:
                break
        if uniq:
            lines.append(f"- {t}: {'、'.join(uniq)}")
    if not lines:
        return ""
    return "\n".join(lines)


def _load_allowed_relation_types(xlsx_path: Path) -> List[str]:
    if not xlsx_path.exists():
        return []
    df = pd.read_excel(xlsx_path)
    rel_col = None
    for c in ["relation_1", "relation", "predicate", "rel", "关系"]:
        if c in df.columns:
            rel_col = c
            break
    if rel_col is None:
        return []
    rels = []
    for v in df[rel_col].tolist():
        if isinstance(v, str) and v.strip():
            rels.append(v.strip())
    return sorted(set(rels))


def generate_semantic_only_prompt(
    news_article: str,
    publish_date: str = "",
    taxonomy_reference_text: str = "",
) -> str:
    publish_hint = f"\n新闻发布日期：{publish_date}" if publish_date else ""
    tax_hint = f"\n参考词表（语义对齐，非硬匹配）:\n{taxonomy_reference_text}" if taxonomy_reference_text else ""
    return f"""
你是城市应急事件抽取助手。请从新闻中抽取结构化事件与事件关系。

重要约束：
1) 只保留城市应急管理相关事件（高温、火灾、爆炸、污染、停水停电、食物中毒、传染病等及同类城市风险）。
2) 输出必须是 JSON 对象，且只包含两个键：events, relations。
3) 你不需要保证 article_id/event_id 最终正确，程序会后处理。
4) 每条 relation 尽量给一条短证据引文 relation_evidence_quote（<=40字）。

events 中每个元素字段：
- event_id: 临时ID，建议E1/E2...
- event_text: 简洁短语
- event_type: Driver/Modulator/Hazard/Impact
- relevance_reasoning: 简短理由+证据（<=50字）
- is_target_event: true/false
- city
- date_from (YYYY-MM-DD)
- date_to (YYYY-MM-DD)
- location (数组)
- attributes (对象)

relations 中每个元素字段：
- source_event_id
- relation_type
- target_event_id
- relation_evidence_quote
- relation_reasoning

新闻正文：
{news_article}
{publish_hint}
{tax_hint}
""".strip()


def extract_event_graph_semantic(
    news_article: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    proxy: Optional[str],
    publish_date: str = "",
    taxonomy_reference_text: str = "",
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    prompt = generate_semantic_only_prompt(
        news_article,
        publish_date=publish_date,
        taxonomy_reference_text=taxonomy_reference_text,
    )
    result_text, usage = query_deepseek(prompt, api_key, model=model, base_url=base_url, proxy=proxy)
    parsed = _try_load_json(result_text)
    if parsed is None:
        raise ValueError(f"模型输出无法解析为 JSON: {result_text[:500]}")
    parsed.setdefault("events", [])
    parsed.setdefault("relations", [])
    return parsed, usage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="仅处理前 N 条，例如 --limit 100")
    ap.add_argument("--batch_size", type=int, default=100, help="每个 run 的样本数，默认 100")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    data_dir = root.parent / "data"
    results_dir = root.parent / "results"
    csv_path = data_dir / "article" / "articles_deduped.csv"
    taxonomy_path = data_dir / "background" / "entities_by_type.json"
    rel_xlsx = data_dir / "background" / "graph_database_export.xlsx"
    results_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    if args.limit is not None and args.limit > 0:
        df = df.head(int(args.limit))
        print(f"启用 limit: 仅处理前 {len(df)} 条", flush=True)

    taxonomy_ref = ""
    if taxonomy_path.exists():
        with open(taxonomy_path, "r", encoding="utf-8") as f:
            taxonomy_ref = build_taxonomy_reference_text(json.load(f), limit_per_type=80)
    allowed_relation_types = _load_allowed_relation_types(rel_xlsx)

    batch_size = max(1, int(args.batch_size))
    print(f"开始抽取: 总计 {len(df)} 条, batch_size={batch_size}", flush=True)
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_norm: Dict[str, int] = {}
    all_results: List[Dict[str, Any]] = []
    t0 = time.perf_counter()

    for batch_idx, start in enumerate(range(0, len(df), batch_size), start=1):
        run_name = f"run{batch_idx}"
        run_dir = results_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        end = min(start + batch_size, len(df))
        print(f"{run_name} 开始：处理索引 {start}~{end - 1}", flush=True)

        batch_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        batch_norm: Dict[str, int] = {}
        batch_results: List[Dict[str, Any]] = []
        batch_t0 = time.perf_counter()

        for global_idx in range(start, end):
            row = df.iloc[global_idx]
            article_id = f"A{global_idx:03d}"
            text = str(row.get("content", "") or "")
            raw_pt = row.get("publish_time", "")
            publish_date = str(raw_pt).split()[0] if isinstance(raw_pt, str) and raw_pt.strip() else ""
            try:
                raw_extraction, usage = extract_event_graph_semantic(
                    text,
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL,
                    model=DEEPSEEK_MODEL,
                    proxy=HTTP_PROXY or None,
                    publish_date=publish_date,
                    taxonomy_reference_text=taxonomy_ref,
                )
                normalized, norm_stats = normalize_event_graph(
                    raw_extraction=raw_extraction,
                    record_article_id=article_id,
                    allowed_relation_types=allowed_relation_types,
                )
                for k in batch_usage:
                    v = int(usage.get(k, 0))
                    batch_usage[k] += v
                    total_usage[k] += v
                for k, v in norm_stats.items():
                    batch_norm[k] = batch_norm.get(k, 0) + int(v)
                    total_norm[k] = total_norm.get(k, 0) + int(v)
                rec = {
                    "index": int(global_idx),
                    "article_id": article_id,
                    "publish_date": publish_date,
                    "content_preview": text[:100],
                    "extraction": normalized,
                    "normalization_stats": norm_stats,
                }
                batch_results.append(rec)
                all_results.append(rec)
                local_done = global_idx - start + 1
                if local_done == 1 or local_done % 10 == 0 or global_idx == end - 1:
                    print(
                        f"{run_name}: 已处理本批 {local_done}/{end - start} 条（全局 {global_idx + 1}/{len(df)}）",
                        flush=True,
                    )
            except Exception as e:
                if isinstance(e, httpx.TimeoutException):
                    print(f"{run_name}: 第 {global_idx} 条请求超时: {e}", flush=True)
                elif isinstance(e, httpx.HTTPError):
                    print(f"{run_name}: 第 {global_idx} 条 HTTP 错误: {e}", flush=True)
                else:
                    print(f"{run_name}: 第 {global_idx} 条处理失败: {e}", flush=True)
                rec = {
                    "index": int(global_idx),
                    "article_id": article_id,
                    "publish_date": publish_date,
                    "content_preview": text[:100],
                    "error": str(e),
                }
                batch_results.append(rec)
                all_results.append(rec)

        run_json_path = run_dir / "results.json"
        run_stats = {
            "batch_index": batch_idx,
            "start_index": start,
            "end_index": end,
            "total_prompt_tokens": batch_usage["prompt_tokens"],
            "total_completion_tokens": batch_usage["completion_tokens"],
            "total_tokens": batch_usage["total_tokens"],
            "normalization_stats": batch_norm,
        }
        with open(run_json_path, "w", encoding="utf-8") as f:
            json.dump({"run_id": run_name, "run_stats": run_stats, "results": batch_results}, f, ensure_ascii=False, indent=2)
        batch_runtime = time.perf_counter() - batch_t0
        print(
            f"{run_name} 完成：第 {start}~{end - 1} 条, "
            f"tokens={batch_usage['total_tokens']} (prompt={batch_usage['prompt_tokens']}, completion={batch_usage['completion_tokens']}), "
            f"runtime={round(batch_runtime, 2)}s",
            flush=True,
        )

    runtime_seconds = time.perf_counter() - t0
    output_path = results_dir / "deepseek_event_graph_v2_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_stats": {
                    "total_runtime_seconds": round(runtime_seconds, 2),
                    "total_prompt_tokens": total_usage["prompt_tokens"],
                    "total_completion_tokens": total_usage["completion_tokens"],
                    "total_tokens": total_usage["total_tokens"],
                    "normalization_stats": total_norm,
                },
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"结果汇总已保存到：{output_path}", flush=True)
    print(
        f"run_stats: runtime={round(runtime_seconds, 2)}s, tokens={total_usage['total_tokens']} "
        f"(prompt={total_usage['prompt_tokens']}, completion={total_usage['completion_tokens']})",
        flush=True,
    )


if __name__ == "__main__":
    main()
