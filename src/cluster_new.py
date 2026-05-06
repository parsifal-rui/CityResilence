"""
时空语义三维联合聚类（CPU版，DBSCAN）

对应教授要求：
1. 文章去重：假设你已在上游完成，这里不再做同文转载去重。
2. 基本单元：event 记录；要求 city 存在且能命中经纬度。
3. 聚类对象：Driver / Hazard / Modulator，统一聚类；不使用 Impact。
4. 语义维度：默认同一篇文章 semantic distance = 0，不同文章 = 1。
   span/paragraph/sentence 细化逻辑已保留为注释，后续可打开。
5. 时间/空间硬阈值：超过 T_max 天或 S_max km，不能互为邻居。
6. 输出：event_texts 按频次排序，不输出 event_type；debug 模式可输出 members。

用法：
  python cluster_events_cpu.py results/run1/results.json out/clustered_events.json
  python cluster_events_cpu.py results/run*/results.json out/clustered_events.json
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any

import numpy as np
from sklearn.cluster import DBSCAN


# ─── 超参数配置 ────────────────────────────────────────────────────────────────

@dataclass
class Cfg:
    # 硬阈值：超过该时间/空间范围，两个 event 不能成为 DBSCAN 邻居
    T_max: int = 2                 # days
    S_max: float = 200.0           # km

    # 三维距离权重。注意：语义距离现在是 0/1，因此 w_s 不宜太大
    w_s: float = 0.30              # semantic/article relation weight
    w_t: float = 0.35              # time weight
    w_p: float = 0.35              # place/space weight

    eps: float = 0.55 
    min_samples: int = 3

    lating_path: str = "data/lating.json"
    discard_path: str = "data/discard.txt"

    # 输出每个 cluster 的成员明细，便于人工 debug；正式版可关掉
    include_members: bool = True


CLUSTER_EVENT_TYPES = {"driver", "hazard", "modulator"}


# ─── discard.txt 解析 ─────────────────────────────────────────────────────────

def load_discard_ids(path: str | Path) -> Set[str]:
    ids: Set[str] = set()
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                m = re.search(r'"article_id":\s*"([^"]+)"', line)
                if m:
                    ids.add(m.group(1))
    except FileNotFoundError:
        pass
    return ids


# ─── 城市经纬度加载 ────────────────────────────────────────────────────────────

def load_city_coords(path: str | Path) -> Dict[str, Tuple[float, float]]:
    """
    递归展平 lating.json → {city_name: (lat_deg, lon_deg)}

    注意：沿用你原始代码里的修正逻辑：
    JSON 中 "lat" 字段存储经度，"lng" 字段存储纬度，因此读取时交换为 (lat, lon)。
    """
    def _walk(node: dict, out: dict):
        lon_str, lat_str = node.get("lat"), node.get("lng")
        name = node.get("name")
        if name and lon_str and lat_str:
            out[name] = (float(lat_str), float(lon_str))
        for child in node.get("children", []):
            _walk(child, out)

    with open(path, encoding="utf-8") as f:
        root = json.load(f)

    out: Dict[str, Tuple[float, float]] = {}
    if isinstance(root, list):
        for node in root:
            _walk(node, out)
    else:
        _walk(root, out)
    return out


# ─── 日期解析 ──────────────────────────────────────────────────────────────────

_EPOCH = datetime(2000, 1, 1)
_DATE_FMTS = ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y年%m月%d日")


def date_to_days(s: str) -> int | None:
    if not s:
        return None
    s = str(s).strip()
    # 兼容 "2024-08-29 18:01" 这种形式
    s_short = s[:10]
    for x in (s, s_short):
        for fmt in _DATE_FMTS:
            try:
                return (datetime.strptime(x, fmt) - _EPOCH).days
            except ValueError:
                pass
    return None


def days_to_str(d: int) -> str:
    return (_EPOCH + timedelta(days=int(d))).strftime("%Y-%m-%d")


# ─── 基础工具 ──────────────────────────────────────────────────────────────────

def norm_event_type(t: Any) -> str:
    return str(t or "").strip().lower()


def keep_for_cluster(ev: dict) -> bool:
    return norm_event_type(ev.get("event_type")) in CLUSTER_EVENT_TYPES


def is_valid_location(loc: Any) -> bool:
    return isinstance(loc, list) and len(loc) > 0


# ─── Haversine 向量化（N×N） ───────────────────────────────────────────────────

def haversine_matrix_np(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """输入纬度/经度数组，返回 N×N 空间距离矩阵，单位 km。"""
    R = 6371.0
    phi = np.deg2rad(lats.astype(np.float32))
    lam = np.deg2rad(lons.astype(np.float32))

    dphi = phi[:, None] - phi[None, :]
    dlam = lam[:, None] - lam[None, :]

    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(phi)[:, None] * np.cos(phi)[None, :] * np.sin(dlam / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    return (2.0 * R * np.arcsin(np.sqrt(a))).astype(np.float32)


# ─── 语义距离：默认同文=0，异文=1 ───────────────────────────────────────────────

def build_semantic_distance(records: List[dict]) -> np.ndarray:
    """
    默认版本：
      - 同一篇文章内 event 语义距离 = 0
      - 不同文章 event 语义距离 = 1

    span/paragraph/sentence 细化版先注释掉，后续如果需要更严格控制长文内误聚，
    可替换为下方注释逻辑。
    """
    n = len(records)
    aids = [r["article_id"] for r in records]
    d_sem = np.ones((n, n), dtype=np.float32)

    for i in range(n):
        d_sem[i, i] = 0.0
        for j in range(i + 1, n):
            val = 0.0 if aids[i] == aids[j] else 1.0
            d_sem[i, j] = d_sem[j, i] = val

    return d_sem

    # ── span/paragraph/sentence 细化版：暂时注释 ────────────────────────────────
    # def _safe_int(x):
    #     try:
    #         if x is None:
    #             return None
    #         return int(x)
    #     except Exception:
    #         return None
    #
    # def _char_gap(a: dict, b: dict) -> int | None:
    #     a_start, a_end = _safe_int(a.get("char_start")), _safe_int(a.get("char_end"))
    #     b_start, b_end = _safe_int(b.get("char_start")), _safe_int(b.get("char_end"))
    #     vals = [a_start, a_end, b_start, b_end]
    #     if any(v is None or v < 0 for v in vals):
    #         return None
    #     return max(0, max(a_start, b_start) - min(a_end, b_end))
    #
    # n = len(records)
    # d_sem = np.ones((n, n), dtype=np.float32)
    #
    # for i in range(n):
    #     d_sem[i, i] = 0.0
    #     for j in range(i + 1, n):
    #         a, b = records[i], records[j]
    #         if a["article_id"] != b["article_id"]:
    #             val = 1.0
    #         else:
    #             para_a, para_b = _safe_int(a.get("paragraph_id")), _safe_int(b.get("paragraph_id"))
    #             sent_a, sent_b = _safe_int(a.get("sentence_id")), _safe_int(b.get("sentence_id"))
    #             gap = _char_gap(a, b)
    #
    #             if para_a is not None and para_b is not None and para_a == para_b:
    #                 val = 0.0
    #             elif sent_a is not None and sent_b is not None and abs(sent_a - sent_b) <= 1:
    #                 val = 0.0
    #             elif gap is not None and gap <= 300:
    #                 val = 0.0
    #             else:
    #                 # 同文但距离较远：弱相关，而不是完全无关
    #                 val = 0.4
    #
    #         d_sem[i, j] = d_sem[j, i] = val
    # return d_sem


# ─── 三维距离矩阵 ───────────────────────────────────────────────────────────────

def build_dist_matrix(records: List[dict], cfg: Cfg) -> np.ndarray:
    n = len(records)
    days = np.array([r["date_days"] for r in records], dtype=np.float32)
    lats = np.array([r["lat"] for r in records], dtype=np.float32)
    lons = np.array([r["lon"] for r in records], dtype=np.float32)

    d_sem = build_semantic_distance(records)

    dt = np.abs(days[:, None] - days[None, :]).astype(np.float32)
    d_time = np.clip(dt / float(cfg.T_max), 0.0, 1.0).astype(np.float32)

    geo_km = haversine_matrix_np(lats, lons)
    d_space = np.clip(geo_km / float(cfg.S_max), 0.0, 1.0).astype(np.float32)

    D = (cfg.w_s * d_sem + cfg.w_t * d_time + cfg.w_p * d_space).astype(np.float32)

    # 硬约束：超过 T_max 或 S_max，不能成为邻居
    hard_mask = (dt > cfg.T_max) | (geo_km > cfg.S_max)
    D[hard_mask] = 1e9
    np.fill_diagonal(D, 0.0)

    return D


# ─── 预处理：results.json → event records ─────────────────────────────────────

def flatten_records(data: dict, discard_ids: Set[str] = frozenset()) -> List[dict]:
    """
    单 run：从 test_results/results.json 展平为 cluster records。

    注意：
    - 只保留 Driver/Hazard/Modulator。
    - Impact 仍然可以在上游 events/relations 入库，但不进入本阶段 cluster。
    - city 为空的 event 不进入 cluster。
    - location 为空或不是 list 的 event 不进入 cluster。
    """
    records: List[dict] = []

    for art in data.get("results", []):
        art_id = art.get("article_id")
        if not art_id or art_id in discard_ids:
            continue

        publish_date = art.get("publish_date", "")
        extraction = art.get("extraction", {}) or {}

        for ev in extraction.get("events", []):
            if not keep_for_cluster(ev):
                continue

            city = str(ev.get("city") or "").strip()
            if not city:
                continue

            location = ev.get("location")
            if not is_valid_location(location):
                continue

            date = ev.get("date_from") or publish_date
            date_days = date_to_days(str(date))
            if date_days is None:
                continue

            records.append({
                "article_id": art_id,
                "run_id": data.get("run_id", ""),
                "event_id": ev.get("event_id", ""),
                "event_text": ev.get("event_text", ""),
                "event_name": ev.get("event_name", ""),
                "event_type": ev.get("event_type", ""),
                "city": city,
                "location": location,
                "date": str(date)[:10],
                "date_days": date_days,
                "evidence": ev.get("evidence", ""),
                "sentence_id": ev.get("sentence_id"),
                "paragraph_id": ev.get("paragraph_id"),
                "char_start": ev.get("char_start"),
                "char_end": ev.get("char_end"),
                "confidence": ev.get("confidence"),
            })

    return records


def flatten_records_multi(data_list: List[dict], discard_ids: Set[str] = frozenset()) -> List[dict]:
    """
    多 run 合并：按 article_id 去重，只保留第一次出现的文章。
    如果你上游已有文章去重，这里仍然可以保留这个保护。
    """
    seen: Set[str] = set()
    all_records: List[dict] = []

    for data in data_list:
        filtered_results = []
        for art in data.get("results", []):
            art_id = art.get("article_id")
            if not art_id or art_id in discard_ids or art_id in seen:
                continue
            seen.add(art_id)
            filtered_results.append(art)

        data2 = dict(data)
        data2["results"] = filtered_results
        all_records.extend(flatten_records(data2, discard_ids=frozenset()))

    return all_records


# ─── 经纬度过滤和补充 ─────────────────────────────────────────────────────────

def attach_coords(records: List[dict], city_coords: Dict[str, Tuple[float, float]]) -> List[dict]:
    """
    只保留 city 能命中经纬度表的 records。
    不再使用广东中心点兜底，避免错误聚类。
    """
    valid: List[dict] = []
    missing = Counter()

    for r in records:
        city = r.get("city", "")
        if city in city_coords:
            lat, lon = city_coords[city]
            r2 = dict(r)
            r2["lat"] = float(lat)
            r2["lon"] = float(lon)
            valid.append(r2)
        else:
            missing[city] += 1

    if missing:
        print(f"[warn] missing cities in lating.json: {missing.most_common(20)}")
        print(f"[warn] dropped records due to missing city coords: {sum(missing.values())}")

    return valid


# ─── 主聚类函数 ────────────────────────────────────────────────────────────────

def cluster_events(records: List[dict], cfg: Cfg | None = None, base_dir: str | Path = ".") -> List[dict]:
    if cfg is None:
        cfg = Cfg()

    base = Path(base_dir)
    city_coords = load_city_coords(base / cfg.lating_path)
    records = attach_coords(records, city_coords)

    n = len(records)
    if n < cfg.min_samples:
        print(f"[cluster] not enough records: n={n}, min_samples={cfg.min_samples}")
        return []

    D = build_dist_matrix(records, cfg)

    labels = DBSCAN(
        eps=cfg.eps,
        min_samples=cfg.min_samples,
        metric="precomputed",
    ).fit_predict(D)

    valid_mask = labels >= 0
    n_clusters = int(labels[valid_mask].max() + 1) if valid_mask.any() else 0
    noise = int((~valid_mask).sum())
    print(f"[cluster] n={n}, clusters={n_clusters}, noise={noise}, noise_ratio={noise / n:.3f}")

    cluster_map: Dict[int, List[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        if lbl >= 0:
            cluster_map[int(lbl)].append(i)

    clustered_events: List[dict] = []
    for cid, members in sorted(cluster_map.items()):
        m_records = [records[i] for i in members]
        m_texts = [r["event_text"] for r in m_records]
        m_days = [r["date_days"] for r in m_records]
        m_cities = [r["city"] for r in m_records]
        m_aids = [r["article_id"] for r in m_records]

        text_counts = Counter(m_texts)

        item = {
            "cluster_id": cid,
            "event_texts": [text for text, _ in text_counts.most_common()],
            "event_text_counts": dict(text_counts.most_common()),
            "cities": sorted(set(m_cities)),
            "date_start": days_to_str(min(m_days)),
            "date_end": days_to_str(max(m_days)),
            "article_ids": sorted(set(m_aids)),
            "n_records": len(members),
            "n_articles": len(set(m_aids)),
        }

        if cfg.include_members:
            item["members"] = [
                {
                    "article_id": r.get("article_id"),
                    "event_id": r.get("event_id"),
                    "event_text": r.get("event_text"),
                    "event_name": r.get("event_name"),
                    "event_type": r.get("event_type"),
                    "city": r.get("city"),
                    "date": r.get("date"),
                    "location": r.get("location"),
                    "sentence_id": r.get("sentence_id"),
                    "paragraph_id": r.get("paragraph_id"),
                    "char_start": r.get("char_start"),
                    "char_end": r.get("char_end"),
                    "evidence": r.get("evidence"),
                    "confidence": r.get("confidence"),
                }
                for r in m_records
            ]

        clustered_events.append(item)

    return clustered_events


# ─── 结果摘要 ──────────────────────────────────────────────────────────────────

def summarize_clusters(clusters: List[dict], n_input_records: int) -> dict:
    if not clusters:
        return {
            "n_input_records": n_input_records,
            "n_clusters": 0,
            "n_clustered_records": 0,
            "noise_records_est": n_input_records,
            "noise_ratio_est": 1.0 if n_input_records else 0.0,
            "avg_cluster_size": 0,
            "median_cluster_size": 0,
            "max_cluster_size": 0,
        }

    sizes = np.array([c["n_records"] for c in clusters], dtype=np.int32)
    n_clustered = int(sizes.sum())
    return {
        "n_input_records": int(n_input_records),
        "n_clusters": int(len(clusters)),
        "n_clustered_records": n_clustered,
        "noise_records_est": int(n_input_records - n_clustered),
        "noise_ratio_est": float((n_input_records - n_clustered) / n_input_records) if n_input_records else 0.0,
        "avg_cluster_size": float(np.mean(sizes)),
        "median_cluster_size": float(np.median(sizes)),
        "max_cluster_size": int(np.max(sizes)),
        "avg_articles_per_cluster": float(np.mean([c["n_articles"] for c in clusters])),
        "max_articles_per_cluster": int(max(c["n_articles"] for c in clusters)),
    }


# ─── CLI 入口 ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python cluster_events_cpu.py <input1.json> [input2.json ...] <output.json>")
        sys.exit(1)

    *input_paths, output_path = sys.argv[1:]

    base_dir = Path(__file__).parent.parent
    cfg = Cfg()

    discard_ids = load_discard_ids(base_dir / cfg.discard_path)
    print(f"Discard list: {len(discard_ids)} article_ids")

    data_list = []
    for p in input_paths:
        with open(p, encoding="utf-8") as f:
            data_list.append(json.load(f))

    if len(data_list) == 1:
        records = flatten_records(data_list[0], discard_ids)
    else:
        records = flatten_records_multi(data_list, discard_ids)

    print(f"Loaded {len(records)} candidate cluster records from {len(data_list)} file(s)")

    clusters = cluster_events(records, cfg=cfg, base_dir=base_dir)
    summary = summarize_clusters(clusters, n_input_records=len(records))
    print("Summary:", json.dumps(summary, ensure_ascii=False, indent=2))

    out = {
        "config": cfg.__dict__,
        "summary": summary,
        "clusters": clusters,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Saved → {output_path}")
