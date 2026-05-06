#!/usr/bin/env python
import os
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel

from deepseek_event_schema import generate_event_schema_prompt


# ====== 配置 ======
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
HTTP_PROXY = os.environ.get("HTTP_PROXY", "") or os.environ.get("http_proxy", "")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

MODEL_PATH = "/root/data/CityResilence/models/bge-m3"
MAX_LENGTH = 512


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
    timeout: float = 90,
) -> Tuple[str, Dict[str, int]]:
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 为空：请先设置环境变量 DEEPSEEK_API_KEY。")

    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 6000,
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


# ====== RAG 检索逻辑 ======


def _cosine_similarity(q: np.ndarray, kb: np.ndarray) -> np.ndarray:
    q = np.asarray(q).reshape(-1)
    if q.ndim != 1:
        q = q.ravel()
    nq = np.linalg.norm(q)
    if nq <= 1e-12:
        return np.zeros(kb.shape[0], dtype=np.float32)
    scores = (kb @ q) / (np.linalg.norm(kb, axis=1) * nq + 1e-12)
    return scores


def load_triples_from_jsonl(path: Path) -> List[Dict[str, Any]]:
    triples: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            triples.append(json.loads(line))
    return triples


def format_retrieved_knowledge(triples: List[Dict[str, str]]) -> str:
    if not triples:
        return ""
    lines = []
    for i, t in enumerate(triples, 1):
        line = f"{i}. [{t.get('type_1', '')}] {t.get('entity_1', '')} —{t.get('relation', '')}→ [{t.get('type_2', '')}] {t.get('entity_2', '')}"
        lines.append(line)
    return "\n".join(lines)


def generate_event_schema_prompt_with_rag(
    news_article: str,
    retrieved_knowledge: str,
    article_id: str = "",
    publish_date: str = "",
) -> str:
    base = generate_event_schema_prompt(news_article, article_id=article_id, publish_date=publish_date)
    rag_block = "\n\n**相关知识（仅供参考，不可引入新闻外信息）：**\n" + (
        retrieved_knowledge if retrieved_knowledge else "（无相关知识）"
    ) + "\n\n"
    marker = "\n\n**抽取要求（核心）：**"
    if marker in base:
        return base.replace(marker, rag_block + marker, 1)
    return base + rag_block


def _try_load_json(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()

    def clean_string_values(match):
        key = match.group(1)
        value = match.group(2)
        value_clean = " ".join(value.split())
        return f'"{key}": "{value_clean}"'

    text = re.sub(r'"([^"]+)":\s*"([^"]*(?:\n[^"]*)*)"', clean_string_values, text)

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            json_str = m.group(0)
            json_str = re.sub(r'"([^"]+)":\s*"([^"]*(?:\n[^"]*)*)"', clean_string_values, json_str)
            obj = json.loads(json_str)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def retrieve_top_k(
    query_text: str,
    *,
    model: BGEM3FlagModel,
    kb_embeddings: np.ndarray,
    triples: List[Dict[str, Any]],
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    out = model.encode([query_text], max_length=MAX_LENGTH, return_dense=True)
    query_emb = np.array(out["dense_vecs"][0], dtype=np.float32)
    scores = _cosine_similarity(query_emb, kb_embeddings)
    top_k_indices = np.argsort(scores)[::-1][:top_k]
    return [triples[i] for i in top_k_indices]


def extract_events_and_relations_with_rag(
    news_article: str,
    *,
    model: BGEM3FlagModel,
    kb_embeddings: np.ndarray,
    triples: List[Dict[str, Any]],
    api_key: str,
    base_url: str,
    llm_model: str,
    proxy: Optional[str],
    top_k: int = 20,
    article_id: str = "",
    publish_date: str = "",
) -> Dict[str, Any]:
    retrieved_triples = retrieve_top_k(
        news_article,
        model=model,
        kb_embeddings=kb_embeddings,
        triples=triples,
        top_k=top_k,
    )
    retrieved_knowledge = format_retrieved_knowledge(retrieved_triples)

    prompt = generate_event_schema_prompt_with_rag(
        news_article, retrieved_knowledge, article_id=article_id, publish_date=publish_date
    )

    result_text, usage = query_deepseek(
        prompt,
        api_key,
        model=llm_model,
        base_url=base_url,
        proxy=proxy,
    )

    parsed = _try_load_json(result_text)
    if parsed is None:
        raise ValueError(f"模型输出不是可解析 JSON：\n{result_text[:2000]}")

    parsed.setdefault("events", [])
    parsed.setdefault("relations", [])
    parsed["retrieved_knowledge_count"] = len(retrieved_triples)
    return parsed, usage


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root.parent / "data"
    results_dir = root.parent / "results"

    csv_path = (data_dir / "articles_cleaned.csv").resolve()
    kb_triples_path = (data_dir / "kb_triples.jsonl").resolve()
    kb_embeddings_path = (data_dir / "kb_embeddings.npy").resolve()

    print("加载预计算的 KB 三元组与向量...")
    triples = load_triples_from_jsonl(kb_triples_path)
    kb_embeddings = np.load(kb_embeddings_path).astype(np.float32)
    if kb_embeddings.shape[0] != len(triples):
        raise ValueError(f"kb_embeddings 行数 {kb_embeddings.shape[0]} 与 triples 数量 {len(triples)} 不一致")

    print("加载 BGE-M3 模型（用于 query 向量化）...")
    bge_model = BGEM3FlagModel(MODEL_PATH, device="cuda", use_fp16=True)

    print(f"读取新闻 CSV：{csv_path}")
    df = pd.read_csv(csv_path)
    texts = df.head(100)["content"].fillna("").tolist()
    publish_dates = df.head(100)["publish_time"].fillna("").apply(
        lambda x: x.split()[0] if x and len(x.split()) > 0 else ""
    ).tolist()

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    results: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    for i, text in enumerate(texts[:20]):
        article_id = f"A{i:03d}"
        publish_date = publish_dates[i] if i < len(publish_dates) else ""
        try:
            out, usage = extract_events_and_relations_with_rag(
                text,
                model=bge_model,
                kb_embeddings=kb_embeddings,
                triples=triples,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                llm_model=DEEPSEEK_MODEL,
                proxy=HTTP_PROXY or None,
                top_k=20,
                article_id=article_id,
                publish_date=publish_date,
            )
            for k in total_usage:
                total_usage[k] += usage.get(k, 0)
            results.append(
                {
                    "index": i,
                    "article_id": article_id,
                    "publish_date": publish_date,
                    "content_preview": text[:100] if text else "",
                    "extraction": out,
                }
            )
            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"已处理 {i+1}/{len(texts[:10])} 条（检索到 {out.get('retrieved_knowledge_count', 0)} 条相关知识）",
                    flush=True,
                )
        except Exception as e:
            print(f"第 {i} 条处理失败：{e}", flush=True)
            results.append(
                {
                    "index": i,
                    "article_id": article_id,
                    "publish_date": publish_date,
                    "content_preview": text[:100] if text else "",
                    "error": str(e),
                }
            )
    runtime_seconds = time.perf_counter() - t0
    run_stats = {
        "total_runtime_seconds": round(runtime_seconds, 2),
        "total_prompt_tokens": total_usage["prompt_tokens"],
        "total_completion_tokens": total_usage["completion_tokens"],
        "total_tokens": total_usage["total_tokens"],
    }

    output_path = results_dir / "deepseek_event_schema_rag_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"run_stats": run_stats, "results": results}, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到：{output_path}")
    print(f"成功：{sum(1 for r in results if 'extraction' in r)} 条，失败：{sum(1 for r in results if 'error' in r)} 条")
    print(f"run_stats: runtime={run_stats['total_runtime_seconds']}s, tokens={run_stats['total_tokens']} (prompt={run_stats['total_prompt_tokens']}, completion={run_stats['total_completion_tokens']})")


if __name__ == "__main__":
    main()