#!/usr/bin/env python
import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel


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
) -> str:
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 为空：请先设置环境变量 DEEPSEEK_API_KEY。")

    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 6000,#之前是3000 不太够
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
    return data["choices"][0]["message"]["content"]


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


def generate_event_schema_prompt_with_rag(news_article: str, retrieved_knowledge: str) -> str:
    rag_section = (
        f"""
**相关知识（仅供参考，不可引入新闻外信息）：**
{retrieved_knowledge if retrieved_knowledge else "（无相关知识）"}
"""
        if retrieved_knowledge
        else ""
    )

    return f"""
    你是一名信息抽取专家。请从以下新闻稿中抽取「事件图谱」：
    
    **事件类型定义：**
    - **驱动因素 (Driver)**：导致气候/灾害事件的直接驱动因子（例如：大气阻塞、重度降水、暴雨、台风）
    - **调节因素 (Modulator)**：调节灾害强度/频率的背景因素（例如：海洋表面温度、气候变化、厄尔尼诺）
    - **灾害 (Hazard)**：可能造成负面影响的现象/事件（例如：洪水、滑坡、干旱、热浪、积水）
    - **影响 (Impact)**：灾害造成的负面后果（例如：人员伤亡、财产损失、交通中断、建筑倒塌）
    
    **关系类型定义：**
    - **引发**：一个事件直接导致另一个事件发生
    - **加剧**：一个事件使另一个事件恶化
    - **削弱**：一个事件减弱另一个事件的强度
    - **增强**：一个事件增强另一个事件的强度
    - **缓解**：一个事件缓解另一个事件的负面影响
    - **抑制**：一个事件抑制另一个事件的发生
    
    **新闻稿：**
    {news_article}
    {rag_section}
    **抽取要求：**
    1. **仅抽取来源于新闻稿的事件和关系**，不得引入外部信息。相关知识仅供参考事件类型和关系类型的分类。
    2. **每个事件必须给出**：
       - `event_id`（唯一标识，如 E1, E2...）
       - `event_text`（简短描述）
       - `event_type`（从上述四类中选择）
       - `time`（时间信息，包含原文表述、标准化格式；若无时间信息，`time` 字段填 `null`）
         - `time.text` 只填时间表述本身（如"9月14日"），不要填"受...影响"等上下文
       - `location`（地点信息，包含原文表述、推断城市、行政层级；若无地点信息，`location` 字段填 `null`）
         - **城市推断规则**：若出现区/街道/县（如"龙岗区"），需补全到城市（"深圳"）；若只有省/国家或无地点，`city_inferred` 填 `null`
       - `attributes`（事件属性，如人数/金额/强度等，根据事件类型灵活填写；若无属性可填空字典 `{{}}`）
    3. 在决定 `time`、`location` 和事件/关系是否输出时，你必须在新闻稿中找到对应的证据句并据此判断，但**最终 JSON 中不要输出任何证据句字段**。
    4. **所有用于内部判断的证据句必须能在新闻稿原文中找到**，否则不要输出该事件/关系。
    5. **语义相近的事件需合并**。
    
    **输出格式（纯 JSON，不要 markdown 代码块）：**
    {{
      "events": [
        {{
          "event_id": "E1",
          "event_text": "...",
          "event_type": "Driver|Modulator|Hazard|Impact",
          "time": {{
            "text": "...",
            "normalized": "YYYY-MM-DD 或 null"
          }} 或 null,
          "location": {{
            "text": "...",
            "city_inferred": "城市名 或 null",
            "admin_hierarchy": ["省", "市", "区", "街道"]
          }} 或 null,
          "attributes": {{}}
        }}
      ],
      "relations": [
        {{
          "source_event_id": "E1",
          "relation_type": "引发|加剧|削弱|增强|缓解|抑制",
          "target_event_id": "E2"
        }}
      ]
    }}
    
    **重要：最终 JSON 中不要包含任何名为 `evidence_sentence` 或 `evidence_sentences` 的字段。**
    """.strip()


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
) -> Dict[str, Any]:
    retrieved_triples = retrieve_top_k(
        news_article,
        model=model,
        kb_embeddings=kb_embeddings,
        triples=triples,
        top_k=top_k,
    )
    retrieved_knowledge = format_retrieved_knowledge(retrieved_triples)

    prompt = generate_event_schema_prompt_with_rag(news_article, retrieved_knowledge)

    result_text = query_deepseek(
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
    return parsed


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root.parent / "data"

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

    results: List[Dict[str, Any]] = []
    for i, text in enumerate(texts[:10]):
        try:
            out = extract_events_and_relations_with_rag(
                text,
                model=bge_model,
                kb_embeddings=kb_embeddings,
                triples=triples,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                llm_model=DEEPSEEK_MODEL,
                proxy=HTTP_PROXY or None,
                top_k=20,
            )
            results.append(
                {
                    "index": i,
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
                    "content_preview": text[:100] if text else "",
                    "error": str(e),
                }
            )

    output_path = root / "deepseek_event_schema_rag_results.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到：{output_path}")
    print(f"成功处理：{sum(1 for r in results if 'extraction' in r)} 条")
    print(f"失败：{sum(1 for r in results if 'error' in r)} 条")


if __name__ == "__main__":
    main()