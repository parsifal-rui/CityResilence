#!/usr/bin/env python
"""
阶段 B：事件图谱 schema 抽取
- 输出从「实体/关系」升级为「events + relations」
- 事件包含：time, location (城市级推断), attributes
- 关系类型扩展
"""
import os
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd


# ====== 配置（复用 deepseek_only_chat 的配置） ======
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
HTTP_PROXY = os.environ.get("HTTP_PROXY", "") or os.environ.get("http_proxy", "")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


def _httpx_client(proxy: Optional[str]) -> httpx.Client:
    """复用 deepseek_only_chat 的 httpx 客户端逻辑"""
    if not proxy:
        return httpx.Client()
    try:
        return httpx.Client(proxies=proxy)
    except TypeError:
        return httpx.Client(proxy=proxy)


def query_deepseek(prompt: str, api_key: str, *, model: str, base_url: str, proxy: Optional[str], timeout: float = 90) -> str:
    """复用 deepseek_only_chat 的请求逻辑"""
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 为空：请先设置环境变量 DEEPSEEK_API_KEY。")

    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 6000,  # 事件 schema 更复杂，token 上限提高到 6000
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


def generate_event_schema_prompt(news_article: str, article_id: str = "", publish_date: str = "") -> str:
    """
    生成事件图谱 schema 的 prompt
    
    Args:
        news_article: 新闻正文
        article_id: 文章 ID（可选）
        publish_date: 新闻发布日期 YYYY-MM-DD（可选，用于时间回填）
    
    输出结构示例：
    {
      "events": [
        {
          "event_id": "E1",
          "event_text": "积水",
          "event_type": "Hazard",
          "city": "深圳",
          "date_from": "2024-09-01",
          "date_to": "2024-09-01",
          "location": ["广东省", "深圳市", "龙岗区", "金碧街"],
          "attributes": {
            "severity": "severe",
            "time_uncertain": false
          },
          "article_id": "001",
          "evidence_sentences": ["..."]
        }
      ],
      "relations": [
        {
          "source_event_id": "E1",
          "relation_type": "引发",
          "target_event_id": "E2",
          "evidence_sentence": "..."
        }
      ]
    }
    """
    publish_date_hint = f"\n**新闻发布日期：**\n{publish_date}\n（若正文中时间模糊如'近期/近日'，使用该日期，并在 attributes 里标注 time_uncertain=true）" if publish_date else ""
    
    return f"""
你是一名信息抽取专家。请从以下新闻稿中抽取「事件图谱」：

**事件类型定义：**
- **驱动因素 (Driver)**：导致气候/灾害事件的直接驱动因子（例如：大气阻塞、降水、暴雨、台风）
- **调节因素 (Modulator)**：调节灾害强度/频率的背景因素（例如：海温、气候变化、厄尔尼诺）
- **灾害 (Hazard)**：可能造成负面影响的现象/事件（例如：洪水、滑坡、干旱、热浪、积水）
- **影响 (Impact)**：灾害造成的负面后果（例如：伤亡、损失、中断、倒塌）

**关系类型定义：**
- **引发**：一个事件直接导致另一个事件发生
- **加剧**：一个事件使另一个事件恶化
- **削弱**：一个事件减弱另一个事件的强度
- **增强**：一个事件增强另一个事件的强度
- **缓解**：一个事件缓解另一个事件的负面影响
- **抑制**：一个事件抑制另一个事件的发生

**新闻稿：**
{news_article}
{publish_date_hint}

**抽取要求（核心）：**
1. **event_text / event_type 保持简洁**：
   - event_text 只保留核心名词短语，如"暴雨"而不是"突降暴雨"，"积水"而不是"多处出现积水"
   - 形容词（如"严重"）、程度（如"特大"）、影响范围（如"多处"）放入 attributes（例如 severity="severe", scope="multiple"）

2. **时间字段 (date_from / date_to)**：
   - 必须至少到"天"级别（格式 YYYY-MM-DD）
   - 若新闻中无法解析到具体日期（如"近期/近日/近年"），使用新闻发布日期，并在 attributes 里加 time_uncertain=true
   - 若事件是单日，date_from = date_to
   - 若事件跨度多日，填写起止日期

3. **地点字段 (city / location)**：
   - **city（必填）**：至少推断到地级市级别（如"深圳市"可简写为"深圳"）；若无法推断城市则填 null
   - **location（数组）**：自顶向下行政层级，如 ["广东省", "深圳市", "龙岗区", "金碧街"]；具体街道/路名可放在最后一级
   - 若只有省/国家或无地点信息，location 填空数组 []，city 填 null

4. **article_id**：
   - 若提供了文章 ID，每个事件必须包含 article_id 字段

5. **attributes（灵活扩展）**：
   - 根据事件类型灵活添加属性，如：
     - severity: "mild" / "moderate" / "severe"
     - time_uncertain: true / false
     - number / unit（Impact 类：伤亡人数/损失金额等）
     - intensity（Driver/Hazard 类：强度描述）
     - scope（影响范围）
   - 若无额外属性可填空字典 {{}}

6. **evidence_sentences**：
   - 每个事件必须提供支持该事件的原文句子列表
   - 所有证据句子必须能在新闻稿原文中找到
   - 证据句子必须是单行文本，不能包含换行符；若原文有换行，用空格替代

7. **关系抽取**：
   - source_event_id / target_event_id 关联已抽取的事件 ID
   - relation_type 从上述关系类型中选择
   - evidence_sentence 支持该关系的原文句子

8. **事件拆分规则（重要）**：
   - 同类事件但发生在**不同地点**（区/街道级别）时，必须拆分为独立事件
   - 同类事件但**严重程度不同**（如红色/橙色/黄色预警）时，必须拆分为独立事件
   - 每个事件的 location 只能包含**一个具体地点**（一条街道/一个区域），不能合并多个地点

9. **语义去重**：
   - 语义相近的事件需合并（例如"暴雨"和"强降雨"应合并为同一事件）
   - 但若地点或严重程度不同，则不要合并（参见第8条）

**输出格式（纯 JSON，不要 markdown 代码块）：**
{{
  "events": [
    {{
      "event_id": "E1",
      "event_text": "积水",
      "event_type": "Hazard",
      "city": "深圳",
      "date_from": "2024-09-01",
      "date_to": "2024-09-01",
      "location": ["广东省", "深圳市", "龙岗区", "金碧街"],
      "attributes": {{
        "severity": "severe",
        "time_uncertain": false
      }},
      "article_id": "{article_id}",
      "evidence_sentences": ["..."]
    }}
  ],
  "relations": [
    {{
      "source_event_id": "E1",
      "relation_type": "引发",
      "target_event_id": "E2",
      "evidence_sentence": "..."
    }}
  ]
}}

**最小必填字段示例：**
{{
  "event_text": "积水",
  "event_type": "Hazard",
  "city": "深圳",
  "date_from": "2024-09-01",
  "date_to": "2024-09-01",
  "location": ["广东省", "深圳市", "龙岗区", "金碧街"],
  "attributes": {{
    "severity": "severe",
    "time_uncertain": false
  }},
  "article_id": "{article_id}"
}}

**事件拆分示例：**
原文："南沙黄阁、番禺石楼发布暴雨红色预警；荔湾发布橙色预警；越秀、天河发布黄色预警"
应拆分为 4 个独立事件（不同地点 + 不同严重程度）：
- E1: event_text="暴雨", city="广州", location=["广东省","广州市","南沙区","黄阁镇"], severity="severe"（红色预警）
- E2: event_text="暴雨", city="广州", location=["广东省","广州市","番禺区","石楼镇"], severity="severe"（红色预警）
- E3: event_text="暴雨", city="广州", location=["广东省","广州市","荔湾区"], severity="moderate"（橙色预警）
- E4: event_text="暴雨", city="广州", location=["广东省","广州市","越秀区"], severity="mild"（黄色预警）
- E5: event_text="暴雨", city="广州", location=["广东省","广州市","天河区"], severity="mild"（黄色预警）

原文："坪山大道路边树木倒地...荔景路遇到树木倾倒"
应拆分为 2 个独立事件（不同地点）：
- E1: event_text="树木倒地", location=["广东省","深圳市","坪山区","坪山大道"]
- E2: event_text="树木倒地", location=["广东省","深圳市","坪山区","荔景路"]
""".strip()


def _try_load_json(text: str) -> Optional[Dict[str, Any]]:
    """
    复用 deepseek_only_chat 的 JSON 解析逻辑，并增强容错：
    - 清理字符串字段里的异常换行
    """
    text = text.strip()
    
    # 预处理：把字符串值里的换行符替换成空格（避免 JSON 解析失败）
    # 匹配 "key": "value 里有换行的情况
    def clean_string_values(match):
        key = match.group(1)
        value = match.group(2)
        # 把换行和多余空格规范化
        value_clean = ' '.join(value.split())
        return f'"{key}": "{value_clean}"'
    
    # 正则：匹配 "key": "value" 格式，value 可能跨行
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
            # 再次应用清理
            json_str = re.sub(r'"([^"]+)":\s*"([^"]*(?:\n[^"]*)*)"', clean_string_values, json_str)
            obj = json.loads(json_str)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return None


def extract_events_and_relations(
    news_article: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    proxy: Optional[str],
    article_id: str = "",
    publish_date: str = "",
    revise_rounds: int = 0,
) -> Dict[str, Any]:
    """
    抽取事件图谱（events + relations）
    
    Args:
        news_article: 新闻正文
        article_id: 文章 ID（可选）
        publish_date: 新闻发布日期 YYYY-MM-DD（可选，用于时间回填）
        revise_rounds: 自检修订轮数（0=不做修订）
    
    Returns:
        包含 events 和 relations 的字典
    """
    prompt = generate_event_schema_prompt(news_article, article_id=article_id, publish_date=publish_date)
    result_text = query_deepseek(prompt, api_key, model=model, base_url=base_url, proxy=proxy)

    # 可选：多轮自检修订（暂时不实现，先确保单次输出稳定）
    # for _ in range(max(0, revise_rounds)):
    #     follow = generate_follow_up_prompt(result_text, news_article)
    #     result_text = query_deepseek(follow, api_key, model=model, base_url=base_url, proxy=proxy)

    parsed = _try_load_json(result_text)
    if parsed is None:
        raise ValueError(f"模型输出不是可解析 JSON：\n{result_text[:2000]}")
    
    parsed.setdefault("events", [])
    parsed.setdefault("relations", [])
    return parsed


def main():
    root = Path(__file__).resolve().parent
    csv_path = Path("./kg-llm-new/data/articles_cleaned.csv").resolve()
    df = pd.read_csv(csv_path)

    texts = df.head(100)["content"].fillna("").tolist()
    publish_dates = df.head(100)["publish_time"].fillna("").apply(lambda x: x.split()[0] if x and len(x.split()) > 0 else "").tolist()

    results = []
    # 先跑前 1 条测试，确认可用；再改成 texts[:10] 或 texts[:100]
    for i, text in enumerate(texts[:20]):
        article_id = f"A{i:03d}"
        publish_date = publish_dates[i] if i < len(publish_dates) else ""
        try:
            out = extract_events_and_relations(
                text,
                api_key=DEEPSEEK_API_KEY,
                base_url=DEEPSEEK_BASE_URL,
                model=DEEPSEEK_MODEL,
                proxy=HTTP_PROXY or None,
                article_id=article_id,
                publish_date=publish_date,
                revise_rounds=0,
            )
            results.append({
                "index": i,
                "article_id": article_id,
                "publish_date": publish_date,
                "content_preview": text[:100] if text else "",
                "extraction": out
            })
            # 每处理 10 条打印一次进度
            if (i + 1) % 10 == 0 or i == 0:
                print(f"已处理 {i+1}/{len(texts[:10])} 条", flush=True)
        except Exception as e:
            print(f"第 {i} 条处理失败：{e}", flush=True)
            results.append({
                "index": i,
                "article_id": article_id,
                "publish_date": publish_date,
                "content_preview": text[:100] if text else "",
                "error": str(e)
            })
    
    # 保存结果到 JSON 文件
    output_path = Path("./kg-llm-new/deepseek_event_schema_results.json").resolve()
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存到：{output_path}")
    print(f"成功处理：{sum(1 for r in results if 'extraction' in r)} 条")
    print(f"失败：{sum(1 for r in results if 'error' in r)} 条")


if __name__ == "__main__":
    main()
