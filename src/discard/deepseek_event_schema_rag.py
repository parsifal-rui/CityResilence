#!/usr/bin/env python
"""
阶段 C：事件图谱 schema + RAG（本地检索）
- 从 graph_database_export.xlsx 检索相关三元组（实体匹配）
- 把检索结果拼进 prompt 作为辅助知识
- 模仿原始 module.py 的 RAG 逻辑（本地版本，无需 OpenAI embedding）
"""
import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import httpx
import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel


# ====== 配置（复用 deepseek_event_schema） ======
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
HTTP_PROXY = os.environ.get("HTTP_PROXY", "") or os.environ.get("http_proxy", "")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


def _httpx_client(proxy: Optional[str]) -> httpx.Client:
    """复用"""
    if not proxy:
        return httpx.Client()
    try:
        return httpx.Client(proxies=proxy)
    except TypeError:
        return httpx.Client(proxy=proxy)


def query_deepseek(prompt: str, api_key: str, *, model: str, base_url: str, proxy: Optional[str], timeout: float = 90) -> str:
    """复用"""
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 为空：请先设置环境变量 DEEPSEEK_API_KEY。")

    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 6000,#3000 not enough
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
    """q: (d,) 或 (1, d)，kb: (n, d)；返回 (n,) 的相似度"""
    q = np.asarray(q).reshape(-1)
    if q.ndim != 1:
        q = q.ravel()
    nq = np.linalg.norm(q)
    if nq <= 1e-12:
        return np.zeros(kb.shape[0])
    scores = (kb @ q) / (np.linalg.norm(kb, axis=1) * nq + 1e-12)
    return scores


class LocalKnowledgeRetriever:
    """
    本地知识库检索器
    - 从 graph_database_export.xlsx 加载三元组
    - 若提供 bge_model：用 BGE-M3 向量化 + 余弦相似度检索；否则回退到实体匹配
    """
    def __init__(self, kb_path: Path, bge_model: Optional[Any] = None):
        self.kb_path = kb_path
        self.bge_model = bge_model
        self.triples = []
        self.kb_embeddings: Optional[np.ndarray] = None
        self._load_knowledge_base()
        if self.bge_model is not None and self.triples:
            self._encode_kb()

    def _encode_kb(self):
        texts = [t["text"] for t in self.triples]
        out = self.bge_model.encode(texts, max_length=512, return_dense=True)
        self.kb_embeddings = np.array(out["dense_vecs"], dtype=np.float32)
        print(f"知识库向量化完成：{self.kb_embeddings.shape}")

    def _load_knowledge_base(self):
        """加载知识库"""
        df = pd.read_excel(self.kb_path)
        for _, row in df.iterrows():
            e1 = str(row.get("entity_1", ""))
            t1 = str(row.get("type_1", ""))
            rel = str(row.get("relation_1", ""))
            e2 = str(row.get("entity_2", ""))
            t2 = str(row.get("type_2", ""))
            
            # 去掉可能的列表格式标记
            t1 = t1.strip("[]'\"")
            t2 = t2.strip("[]'\"")
            
            if e1 and e2 and rel:
                self.triples.append({
                    "entity_1": e1,
                    "type_1": t1,
                    "relation": rel,
                    "entity_2": e2,
                    "type_2": t2,
                    "text": f"{e1} {rel} {e2}"  # 用于检索的文本
                })
        
        print(f"知识库加载完成：共 {len(self.triples)} 条三元组")

    def retrieve_by_embedding(self, query_text: str, top_k: int = 20) -> List[Dict[str, str]]:
        """【2】向量化查询 + 余弦相似度检索，返回 top_k 三元组"""
        if self.kb_embeddings is None or self.bge_model is None:
            return self.retrieve_by_entity_matching(query_text, top_k=top_k)
        out = self.bge_model.encode([query_text], max_length=512, return_dense=True)
        query_emb = np.array(out["dense_vecs"][0], dtype=np.float32)
        scores = _cosine_similarity(query_emb, self.kb_embeddings)
        top_k_indices = np.argsort(scores)[::-1][:top_k]
        return [self.triples[i] for i in top_k_indices]

    def retrieve_by_entity_matching(self, query_text: str, top_k: int = 20) -> List[Dict[str, str]]:
        """
        模仿原始 module.py 的检索逻辑（本地版本）：
        1. 从新闻文本中提取候选实体词（中文分词/关键词）
        2. 在三元组的 entity_1/entity_2 中查找匹配的实体
        3. 返回包含这些实体的三元组
        
        Args:
            query_text: 查询文本（新闻正文）
            top_k: 最多返回多少条三元组
        
        Returns:
            相关三元组列表
        """
        if not self.triples:
            return []
        
        # 1. 提取新闻中的候选实体词（简单分词：提取2-10字的中文词组）
        # 更激进：提取所有可能的 n-gram（2-6 字）
        candidate_entities = set()
        text_chars = [c for c in query_text if '\u4e00' <= c <= '\u9fa5']  # 只保留中文
        text_str = ''.join(text_chars)
        
        # n-gram 提取（2-6 字）
        for n in range(2, 7):
            for i in range(len(text_str) - n + 1):
                candidate_entities.add(text_str[i:i+n])
        
        if not candidate_entities:
            return []
        
        # 2. 在三元组的 entity_1/entity_2 中查找匹配
        matched_triples = []
        for triple in self.triples:
            e1 = triple["entity_1"]
            e2 = triple["entity_2"]
            
            # 检查实体是否在候选词集合中（或候选词包含该实体）
            if any(e1 in cand or cand in e1 for cand in candidate_entities):
                matched_triples.append(triple)
            elif any(e2 in cand or cand in e2 for cand in candidate_entities):
                matched_triples.append(triple)
        
        # 3. 去重并返回 Top-K
        seen = set()
        unique_results = []
        for t in matched_triples:
            key = (t["entity_1"], t["relation"], t["entity_2"])
            if key not in seen:
                seen.add(key)
                unique_results.append(t)
                if len(unique_results) >= top_k:
                    break
        
        return unique_results


def format_retrieved_knowledge(triples: List[Dict[str, str]]) -> str:
    """
    把检索到的三元组格式化成文本，用于拼接到 prompt
    """
    if not triples:
        return ""
    
    lines = []
    for i, t in enumerate(triples, 1):
        line = f"{i}. [{t['type_1']}] {t['entity_1']} —{t['relation']}→ [{t['type_2']}] {t['entity_2']}"
        lines.append(line)
    
    return "\n".join(lines)


def generate_event_schema_prompt_with_rag(news_article: str, retrieved_knowledge: str) -> str:
    """
    生成带 RAG 的事件图谱 prompt（复用 deepseek_event_schema 的 prompt，只是加入 retrieved_knowledge）
    """
    rag_section = f"""
**相关知识（仅供参考，不可引入新闻外信息）：**
{retrieved_knowledge if retrieved_knowledge else "（无相关知识）"}
""" if retrieved_knowledge else ""
    
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
    """复用 deepseek_event_schema 的 JSON 解析逻辑（含容错）"""
    text = text.strip()
    
    # 预处理：把字符串值里的换行符替换成空格
    def clean_string_values(match):
        key = match.group(1)
        value = match.group(2)
        value_clean = ' '.join(value.split())
        return f'"{key}": "{value_clean}"'
    
    text = re.sub(r'"([^"]+)":\s*"([^"]*(?:\n[^"]*)*)"', clean_string_values, text)
    
    # 1) 直接 json
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) 从文本中抠出第一个大括号 JSON
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


def extract_events_and_relations_with_rag(
    news_article: str,
    *,
    retriever: LocalKnowledgeRetriever,
    api_key: str,
    base_url: str,
    model: str,
    proxy: Optional[str],
    top_k: int = 20,
) -> Dict[str, Any]:
    """
    抽取事件图谱（events + relations）+ RAG
    
    Args:
        retriever: 知识库检索器
        top_k: 检索 Top-K 个三元组（默认 20 条）
    """
    # 1. 检索相关知识：BGE-M3 向量化 + 余弦相似度 Top-K（无 BGE 时回退实体匹配）
    retrieved_triples = retriever.retrieve_by_embedding(news_article, top_k=top_k)
    retrieved_knowledge = format_retrieved_knowledge(retrieved_triples)
    
    # 2. 生成 prompt
    prompt = generate_event_schema_prompt_with_rag(news_article, retrieved_knowledge)
    
    # 3. 调用 LLM
    result_text = query_deepseek(prompt, api_key, model=model, base_url=base_url, proxy=proxy)
    
    # 4. 解析结果
    parsed = _try_load_json(result_text)
    if parsed is None:
        raise ValueError(f"模型输出不是可解析 JSON：\n{result_text[:2000]}")
    
    parsed.setdefault("events", [])
    parsed.setdefault("relations", [])
    parsed["retrieved_knowledge_count"] = len(retrieved_triples)  # 记录检索到多少条
    return parsed


def main():
    root = Path(__file__).resolve().parent
    data_dir = root.parent / "data"
    csv_path = (data_dir / "articles_cleaned.csv").resolve()
    kb_path = (data_dir / "graph_database_export.xlsx").resolve()
    
    # 加载 BGE-M3（GPU）+ 知识库检索器
    print("正在加载 BGE-M3...")
    bge_model = BGEM3FlagModel("/root/data/CityResilence/models/bge-m3", device="cuda", use_fp16=True)
    print("正在加载知识库...")
    retriever = LocalKnowledgeRetriever(kb_path, bge_model=bge_model)
    
    # 加载新闻样本
    df = pd.read_csv(csv_path)
    texts = df.head(100)["content"].fillna("").tolist()

    results = []
    # 先跑前 1 条测试，确认可用；再改成 texts[:10] 或 texts[:100]
    for i, text in enumerate(texts[:10]):
        try:
            out = extract_events_and_relations_with_rag(
                text,
                retriever=retriever,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                model=DEEPSEEK_MODEL,
                proxy=HTTP_PROXY or None,
                top_k=20,
            )
            results.append({
                "index": i,
                "content_preview": text[:100] if text else "",
                "extraction": out
            })
            # 每处理 10 条打印一次进度
            if (i + 1) % 10 == 0 or i == 0:
                print(f"已处理 {i+1}/{len(texts[:10])} 条（检索到 {out.get('retrieved_knowledge_count', 0)} 条相关知识）", flush=True)
        except Exception as e:
            print(f"第 {i} 条处理失败：{e}", flush=True)
            results.append({
                "index": i,
                "content_preview": text[:100] if text else "",
                "error": str(e)
            })
    
    # 保存结果到 JSON 文件
    output_path = root / "deepseek_event_schema_rag_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到：{output_path}")
    print(f"成功处理：{sum(1 for r in results if 'extraction' in r)} 条")
    print(f"失败：{sum(1 for r in results if 'error' in r)} 条")


if __name__ == "__main__":
    main()
