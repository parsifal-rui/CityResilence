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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def query_deepseek(prompt: str, api_key: str, *, model: str, base_url: str, proxy: Optional[str], timeout: float = 90) -> Tuple[str, Dict[str, int]]:
    """复用 deepseek_only_chat 的请求逻辑。返回 (content, usage)。"""
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY 为空：请先设置环境变量 DEEPSEEK_API_KEY。")

    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 8000,  # 防止截断，提高到 8000
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
    return "\n**事件命名参考词表（用于语义对齐，非硬匹配）：**\n" + "\n".join(parts) + "\n"


def generate_event_schema_prompt(
    news_article: str,
    article_id: str = "",
    publish_date: str = "",
    taxonomy_reference_text: str = "",
) -> str:
    """
    生成事件图谱 schema 的 prompt
    """
    publish_date_hint = f"\n**新闻发布日期：**\n{publish_date}\n（若正文中时间模糊如'近期/近日'，使用该日期，并在 attributes 里标注 time_uncertain=true）" if publish_date else ""
    
    return f"""
你是一名信息抽取专家。请从以下新闻稿中抽取「事件图谱」：

**事件类型定义：**
- **驱动因素 (Driver)**：导致气候/灾害事件的直接驱动因子（例如：大气阻塞、降水、暴雨、台风）
- **调节因素 (Modulator)**：调节灾害强度/频率的背景因素（例如：海温、气候变化、厄尔尼诺）
- **灾害 (Hazard)**：可能造成负面影响的现象/事件（例如：洪水、滑坡、干旱、热浪、积水）
- **影响 (Impact)**：灾害造成的负面后果（例如：伤亡、损失、中断、倒塌）

**城市应急相关性过滤（极其重要）：**
- 本次抽取仅保留确属 urban / weather / climate hazards, modulators, drivers 的事件
- 对城市事件重点关注：噪音、高温、火灾、爆炸、污染、停水停电、食物中毒、传染病，以及其他直接造成社会经济影响、需要城市应急管理关注的事件
- 明显与城市应急无关的事件不保留
- 每个候选事件都要先给出判断理由，再给出是否保留

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
{taxonomy_reference_text}

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

6. **关系抽取**：
   - source_event_id / target_event_id 关联已抽取的事件 ID
   - relation_type 从上述关系类型中选择

7. **事件拆分规则（重要）**：
   - 同类事件但发生在**不同地点**（区/街道级别）时，必须拆分为独立事件
   - 同类事件但**严重程度不同**（如红色/橙色/黄色预警）时，必须拆分为独立事件
   - 每个事件的 location 只能包含**一个具体地点**（一条街道/一个区域），不能合并多个地点

8. **语义去重**：
   - 语义相近的事件需合并（例如"暴雨"和"强降雨"应合并为同一事件）
   - 但若地点或严重程度不同，则不要合并（参见第7条）

9. **相关性判定字段（必须输出）**：
   - relevance_reasoning：简短说明该事件为何属于（或不属于）城市/天气/气候风险事件。**必须在此处简要引用原文核心词句作为证据**（限50字以内，禁止大段摘抄）。
   - is_target_event：布尔值。true 表示应保留；false 表示不属于目标范围
   - 最终 events 中只保留 is_target_event=true 的事件

**输出格式（纯 JSON，不要 markdown 代码块）：**
{{
  "events": [
    {{
      "event_id": "E1",
      "event_text": "积水",
      "event_type": "Hazard",
      "relevance_reasoning": "原文称'金碧街出现严重积水'，直接影响交通，属城市应急管理关注范围。",
      "is_target_event": true,
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
  ],
  "relations": [
    {{
      "source_event_id": "E1",
      "relation_type": "引发",
      "target_event_id": "E2"
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
    taxonomy_reference_text: str = "",
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
    prompt = generate_event_schema_prompt(
        news_article,
        article_id=article_id,
        publish_date=publish_date,
        taxonomy_reference_text=taxonomy_reference_text,
    )
    result_text, usage = query_deepseek(prompt, api_key, model=model, base_url=base_url, proxy=proxy)

    parsed = _try_load_json(result_text)
    if parsed is None:
        raise ValueError(f"模型输出不是可解析 JSON：\n{result_text[:2000]}")
    
    parsed.setdefault("events", [])
    parsed.setdefault("relations", [])
    parsed["events"] = [
        e for e in parsed["events"]
        if isinstance(e, dict) and e.get("is_target_event") is True
    ]
    return parsed, usage


def main():
    root = Path(__file__).resolve().parent
    data_dir = root.parent / "data"
    results_dir = root.parent / "results"
    csv_path = (data_dir / "article" /"articles_deduped.csv").resolve()
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

    batch_size = 100
    n_rows = len(df)

    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    all_results = []
    t0 = time.perf_counter()

    for batch_idx, start in enumerate(range(0, n_rows, batch_size), start=1):
        end = min(start + batch_size, n_rows)
        run_name = f"run{batch_idx}"
        run_dir = results_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        usage_txt_path = run_dir / "usage_tokens.txt"
        run_json_path = run_dir / "results.json"

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
                print(f"{run_name} 已存在但结果无法加载，重新处理第 {start}~{end-1} 条", flush=True)
                prev_results = None

            if prev_results is not None:
                failed_txt_path = run_dir / "failed_indices.txt"
                fail_again_path = run_dir / "failAgain.txt"

                if fail_again_path.exists():
                    print(f"{run_name} 已完成重试（failAgain.txt 已存在），直接载入", flush=True)
                    all_results.extend(prev_results)
                    continue

                if not failed_txt_path.exists():
                    print(f"{run_name} 已存在且无失败，跳过第 {start}~{end-1} 条", flush=True)
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
                results_by_idx = {r["index"]: r for r in prev_results}
                retry_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                still_failed = []

                for gidx in sorted(failed_indices):
                    offset = gidx - start
                    row = df.iloc[gidx]
                    text = str(row["content"]) if pd.notna(row["content"]) else ""
                    raw_pt = row["publish_time"] if pd.notna(row.get("publish_time")) else ""
                    publish_date = str(raw_pt).split()[0] if raw_pt and len(str(raw_pt).split()) > 0 else ""
                    article_id = f"A{gidx:03d}"
                    try:
                        out, usage = extract_events_and_relations(
                            text,
                            api_key=DEEPSEEK_API_KEY,
                            base_url=DEEPSEEK_BASE_URL,
                            model=DEEPSEEK_MODEL,
                            proxy=HTTP_PROXY or None,
                            article_id=article_id,
                            publish_date=publish_date,
                            revise_rounds=0,
                            taxonomy_reference_text=taxonomy_reference_text,
                        )
                        for k in retry_usage:
                            v = int(usage.get(k, 0))
                            retry_usage[k] += v
                            total_usage[k] += v
                        results_by_idx[gidx] = {
                            "index": int(gidx),
                            "article_id": article_id,
                            "publish_date": publish_date,
                            "content_preview": text[:100] if text else "",
                            "extraction": out,
                        }
                        print(f"{run_name} 重试: 第 {gidx} 条成功", flush=True)
                    except Exception as e:
                        print(f"{run_name} 重试: 第 {gidx} 条仍然失败：{e}", flush=True)
                        still_failed.append(gidx)

                updated_results = [results_by_idx[r["index"]] for r in prev_results if r["index"] in results_by_idx]
                prev_stats["total_prompt_tokens"] = int(prev_stats.get("total_prompt_tokens", 0)) + retry_usage["prompt_tokens"]
                prev_stats["total_completion_tokens"] = int(prev_stats.get("total_completion_tokens", 0)) + retry_usage["completion_tokens"]
                prev_stats["total_tokens"] = int(prev_stats.get("total_tokens", 0)) + retry_usage["total_tokens"]
                with open(run_json_path, "w", encoding="utf-8") as f:
                    json.dump(
                        {"run_id": run_name, "run_stats": prev_stats, "results": updated_results},
                        f, ensure_ascii=False, indent=2,
                    )
                with open(usage_txt_path, "w", encoding="utf-8") as f:
                    f.write(
                        f"batch_index: {prev_stats.get('batch_index', batch_idx)}\n"
                        f"start_index: {prev_stats.get('start_index', start)}\n"
                        f"end_index: {prev_stats.get('end_index', end)}\n"
                        f"total_prompt_tokens: {prev_stats['total_prompt_tokens']}\n"
                        f"total_completion_tokens: {prev_stats['total_completion_tokens']}\n"
                        f"total_tokens: {prev_stats['total_tokens']}\n"
                    )

                with open(fail_again_path, "w", encoding="utf-8") as f:
                    if still_failed:
                        f.write("\n".join(str(i) for i in sorted(still_failed)))
                    else:
                        f.write("0")

                retry_ok = len(failed_indices) - len(still_failed)
                print(
                    f"{run_name} 重试完成：成功 {retry_ok} 条，仍失败 {len(still_failed)} 条 → failAgain.txt",
                    flush=True,
                )
                all_results.extend(updated_results)
                continue

        batch_df = df.iloc[start:end]
        texts = batch_df["content"].fillna("").tolist()
        publish_dates = batch_df["publish_time"].fillna("").apply(
            lambda x: x.split()[0] if x and isinstance(x, str) and len(x.split()) > 0 else ""
        ).tolist()

        batch_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        batch_results = []
        batch_failed_indices = []
        batch_t0 = time.perf_counter()

        for offset, text in enumerate(texts):
            global_idx = start + offset
            article_id = f"A{global_idx:03d}"
            publish_date = publish_dates[offset] if offset < len(publish_dates) else ""
            try:
                out, usage = extract_events_and_relations(
                    text,
                    api_key=DEEPSEEK_API_KEY,
                    base_url=DEEPSEEK_BASE_URL,
                    model=DEEPSEEK_MODEL,
                    proxy=HTTP_PROXY or None,
                    article_id=article_id,
                    publish_date=publish_date,
                    revise_rounds=0,
                    taxonomy_reference_text=taxonomy_reference_text,
                )
                for k in batch_usage:
                    v = int(usage.get(k, 0))
                    batch_usage[k] += v
                    total_usage[k] += v
                record = {
                    "index": int(global_idx),
                    "article_id": article_id,
                    "publish_date": publish_date,
                    "content_preview": text[:100] if text else "",
                    "extraction": out,
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
            json.dump(
                {
                    "run_id": run_name,
                    "run_stats": run_stats,
                    "results": batch_results,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

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

        if batch_failed_indices:
            failed_path = run_dir / "failed_indices.txt"
            with open(failed_path, "w", encoding="utf-8") as f:
                f.write("\n".join(str(i) for i in sorted(batch_failed_indices)))

        print(
            f"{run_name} 完成：第 {start}~{end-1} 条，"
            f"tokens={batch_usage['total_tokens']} "
            f"(prompt={batch_usage['prompt_tokens']}, completion={batch_usage['completion_tokens']})，"
            f"runtime={round(batch_runtime_seconds, 2)}s",
            flush=True,
        )

    runtime_seconds = time.perf_counter() - t0
    global_run_stats = {
        "total_runtime_seconds": round(runtime_seconds, 2),
        "total_prompt_tokens": total_usage["prompt_tokens"],
        "total_completion_tokens": total_usage["total_completion_tokens"],
        "total_tokens": total_usage["total_tokens"],
    }

    output_path = (results_dir / "deepseek_event_schema_results.json").resolve()
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
    main()