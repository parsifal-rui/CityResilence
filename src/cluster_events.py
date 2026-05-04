"""
时空语义三维联合聚类（DBSCAN）
支持：多 run 合并全局聚类（dedup by article_id）、discard.txt 过滤、
      簇代表文本距质心最近（centroid）或高频词（most_common）
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Set

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN


# ─── 超参数配置 ────────────────────────────────────────────────────────────────

@dataclass
class Cfg:
    T_max: int         = 2          # 时间阈值（天）
    S_max: float       = 200.0      # 空间阈值（km）
    w_s: float         = 0.65       # 语义权重
    w_t: float         = 0.20       # 时间权重
    w_p: float         = 0.15       # 空间权重
    eps: float         = 0.28       # DBSCAN epsilon
    min_samples: int   = 3          # DBSCAN min_samples
    repr_strategy: str = "centroid" # "centroid" | "most_common"
    model_path: str    = "models/bge-m3"
    lating_path: str   = "data/lating.json"
    discard_path: str  = "data/discard.txt"
    device: str        = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


# ─── discard.txt 解析 ─────────────────────────────────────────────────────────

def load_discard_ids(path: str) -> Set[str]:
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

def load_city_coords(path: str) -> Dict[str, tuple]:
    """
    递归展平 lating.json → {city_name: (lat_deg, lon_deg)}
    注：JSON 中 "lat" 字段存储经度，"lng" 字段存储纬度，此处做交换修正。
    """
    def _walk(node: dict, out: dict):
        lon_str, lat_str = node.get("lat"), node.get("lng")
        if lon_str and lat_str:
            out[node["name"]] = (float(lat_str), float(lon_str))   # (lat, lon)
        for child in node.get("children", []):
            _walk(child, out)

    with open(path, encoding="utf-8") as f:
        root = json.load(f)
    out: Dict[str, tuple] = {}
    _walk(root, out)
    return out


# ─── 日期解析 ──────────────────────────────────────────────────────────────────

_EPOCH = datetime(2000, 1, 1)
_DATE_FMTS = ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d", "%Y年%m月%d日")

def date_to_days(s: str) -> int:
    for fmt in _DATE_FMTS:
        try:
            return (datetime.strptime(str(s).strip(), fmt) - _EPOCH).days
        except ValueError:
            pass
    return 0

def days_to_str(d: int) -> str:
    return (_EPOCH + timedelta(days=int(d))).strftime("%Y-%m-%d")


# ─── 语义编码 ──────────────────────────────────────────────────────────────────

def encode_texts(texts: List[str], model_path: str, device: str) -> torch.Tensor:
    """对去重文本列表提取 BGE-M3 Dense Vector（L2 归一化）。"""
    model = SentenceTransformer(model_path, device=device)
    vecs = model.encode(
        texts,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=False,
    )
    if isinstance(vecs, torch.Tensor):
        return vecs.cpu().float()
    if isinstance(vecs, list):
        vecs = torch.stack([v.cpu() if isinstance(v, torch.Tensor) else torch.tensor(v) for v in vecs])
        return vecs.float()
    return torch.from_numpy(np.array(vecs, dtype=np.float32))   # (N_unique, D)


# ─── Haversine 向量化（N×N） ───────────────────────────────────────────────────

def haversine_matrix(lats: torch.Tensor, lons: torch.Tensor) -> torch.Tensor:
    """
    输入：lats/lons 均为度数的一维 Tensor（N,）
    输出：N×N km 距离矩阵
    """
    R = 6371.0
    φ = torch.deg2rad(lats)    # (N,)
    λ = torch.deg2rad(lons)    # (N,)
    dφ = φ.unsqueeze(1) - φ.unsqueeze(0)   # (N, N)
    dλ = λ.unsqueeze(1) - λ.unsqueeze(0)   # (N, N)
    a = (torch.sin(dφ / 2) ** 2
         + torch.cos(φ).unsqueeze(1) * torch.cos(φ).unsqueeze(0) * torch.sin(dλ / 2) ** 2)
    return 2 * R * torch.asin(torch.clamp(torch.sqrt(a), 0.0, 1.0))   # (N, N)


# ─── 子集距离矩阵构建 ──────────────────────────────────────────────────────────

def build_dist_matrix(
    embs:  torch.Tensor,    # (N, D) 已归一化
    days:  torch.Tensor,    # (N,)  整型天数
    lats:  torch.Tensor,    # (N,)  纬度（度）
    lons:  torch.Tensor,    # (N,)  经度（度）
    aids:  List[str],       # (N,)  article_ids
    cfg:   Cfg,
) -> np.ndarray:
    dev = cfg.device
    embs, days, lats, lons = embs.to(dev), days.to(dev), lats.to(dev), lons.to(dev)

    # 语义距离：同一篇文章距离为0，否则为1（根据教授要求）
    N = len(aids)
    d_sem = torch.ones((N, N), device=dev)
    for i in range(N):
        for j in range(N):
            if aids[i] == aids[j]:
                d_sem[i, j] = 0.0

    # 时间距离
    dt      = (days.unsqueeze(1) - days.unsqueeze(0)).abs().float()        # (N, N) 天数差
    d_time  = torch.clamp(dt / cfg.T_max, 0.0, 1.0)

    # 空间距离
    geo_km  = haversine_matrix(lats, lons)                                 # (N, N)
    d_space = torch.clamp(geo_km / cfg.S_max, 0.0, 1.0)

    # 加权融合
    D = cfg.w_s * d_sem + cfg.w_t * d_time + cfg.w_p * d_space            # (N, N)

    # 硬约束 Mask：超出时间或空间阈值直接置 inf
    hard_mask = (dt > cfg.T_max) | (geo_km > cfg.S_max)
    D[hard_mask] = float("inf")

    return D.cpu().numpy()


# ─── 预处理：展平 results.json → 事件记录列表 ──────────────────────────────────

def flatten_records(data: dict, discard_ids: Set[str] = frozenset()) -> List[dict]:
    """单 run：从 results.json 展平为事件记录列表，过滤 discard_ids 和空 location。"""
    records = []
    for art in data.get("results", []):
        art_id = art["article_id"]
        if art_id in discard_ids:
            continue
        publish_date = art.get("publish_date", "")
        for ev in art.get("extraction", {}).get("events", []):
            if not ev.get("location"):
                continue
            if ev.get("event_type") == "impact":
                continue
            date = ev.get("date_from") or publish_date
            if not date:
                continue
            records.append({
                "event_text": ev["event_text"],
                "event_type": ev["event_type"],
                "city":       ev.get("city", ""),
                "date":       date,
                "article_id": art_id,
                "run_id":     data.get("run_id", ""),
            })
    return records


def flatten_records_multi(data_list: List[dict], discard_ids: Set[str] = frozenset()) -> List[dict]:
    """
    多 run 合并展平：按 article_id 去重（同一篇文章只保留首次出现的抽取结果），
    过滤 discard_ids 和空 location。
    """
    seen: Set[str] = set()
    records = []
    for data in data_list:
        for art in data.get("results", []):
            art_id = art["article_id"]
            if art_id in discard_ids or art_id in seen:
                continue
            seen.add(art_id)
            publish_date = art.get("publish_date", "")
            for ev in art.get("extraction", {}).get("events", []):
                if not ev.get("location"):
                    continue
                if ev.get("event_type") == "impact":
                    continue
                date = ev.get("date_from") or publish_date
                if not date:
                    continue
                records.append({
                    "event_text": ev["event_text"],
                    "event_type": ev["event_type"],
                    "city":       ev.get("city", ""),
                    "date":       date,
                    "article_id": art_id,
                    "run_id":     data.get("run_id", ""),
                })
    return records


# ─── 主函数 ────────────────────────────────────────────────────────────────────

def cluster_events(records: List[dict], cfg: Cfg | None = None, base_dir: str = ".") -> List[dict]:
    if cfg is None:
        cfg = Cfg()
    base = Path(base_dir)

    city_coords = load_city_coords(base / cfg.lating_path)
    # 广东中心作为未命中城市的兜底坐标
    _DEFAULT_COORD = (23.13, 113.27)

    # ── 特征预计算 ──
    texts    = [r["event_text"]  for r in records]
    types    = [r["event_type"]  for r in records]
    cities   = [r["city"]        for r in records]
    days_raw = [date_to_days(str(r["date"])) for r in records]
    art_ids  = [r["article_id"]  for r in records]

    # 对去重文本编码
    unique_texts = list(dict.fromkeys(texts))
    print(f"[encode] {len(unique_texts)} unique texts → device={cfg.device}")
    all_embs = encode_texts(unique_texts, str(base / cfg.model_path), cfg.device)
    text2row = {t: i for i, t in enumerate(unique_texts)}

    emb_rows = [text2row[t] for t in texts]     # 每条记录对应的 embedding 行号
    lats_all = torch.tensor([city_coords.get(c, _DEFAULT_COORD)[0] for c in cities], dtype=torch.float32)
    lons_all = torch.tensor([city_coords.get(c, _DEFAULT_COORD)[1] for c in cities], dtype=torch.float32)
    days_all = torch.tensor(days_raw, dtype=torch.float32)

    # ── 不按 event_type 分组，统一聚类 ──
    n = len(records)
    if n < cfg.min_samples:
        return []

    sub_embs  = all_embs[emb_rows]
    sub_days  = days_all
    sub_lats  = lats_all
    sub_lons  = lons_all
    sub_aids  = art_ids

    D = build_dist_matrix(sub_embs, sub_days, sub_lats, sub_lons, sub_aids, cfg)
    D_finite = np.where(np.isinf(D), 1e9, D)   # DBSCAN 需要有限值

    local_labels = DBSCAN(
        eps=cfg.eps, min_samples=cfg.min_samples, metric="precomputed"
    ).fit_predict(D_finite)

    global_labels = local_labels

    valid = local_labels >= 0
    n_clusters = int(valid.any()) and (int(local_labels[valid].max()) + 1)
    print(f"[all_types] n={n}, clusters={n_clusters}, noise={int((~valid).sum())}")

    # ── 汇总输出 ──
    cluster_map: Dict[int, List[int]] = defaultdict(list)
    for i, lbl in enumerate(global_labels):
        if lbl >= 0:
            cluster_map[int(lbl)].append(i)

    clustered_events = []
    for cid, members in sorted(cluster_map.items()):
        m_texts  = [texts[i]    for i in members]
        m_types  = [types[i]    for i in members]
        m_days   = [days_raw[i] for i in members]
        m_cities = [cities[i]   for i in members]
        m_aids   = [art_ids[i]  for i in members]

        clustered_events.append({
            "cluster_id":               cid,
            "event_texts":              [text for text, count in Counter(m_texts).most_common()],
            "cities":                   sorted({c for c in m_cities if c}),
            "date_start":               days_to_str(min(m_days)),
            "date_end":                 days_to_str(max(m_days)),
            "article_ids":              sorted(set(m_aids)),
            "n_records":                len(members),
            "n_articles":               len(set(m_aids)),
        })

    return clustered_events


# ─── 入口 ──────────────────────────────────────────────────────────────────────
# 用法：
#   单 run:  python cluster_events.py results/run1/results.json out.json
#   全局:    python cluster_events.py results/run*/results.json out.json
#            （shell glob 展开多个路径，最后一个参数为输出路径）

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: cluster_events.py <input1.json> [input2.json ...] <output.json>")
        sys.exit(1)

    *input_paths, output_path = sys.argv[1:]
    if not input_paths:
        input_paths = ["results/run1/results.json"]
        output_path = "results/run1/clustered_events.json"

    base_dir = str(Path(__file__).parent.parent)
    cfg = Cfg()

    discard_ids = load_discard_ids(str(Path(base_dir) / cfg.discard_path))
    print(f"Discard list: {len(discard_ids)} article_ids")

    data_list = []
    for p in input_paths:
        with open(p, encoding="utf-8") as f:
            data_list.append(json.load(f))

    if len(data_list) == 1:
        records = flatten_records(data_list[0], discard_ids)
    else:
        records = flatten_records_multi(data_list, discard_ids)

    print(f"Loaded {len(records)} records from {len(data_list)} run(s), device={cfg.device}")

    clustered = cluster_events(records, cfg=cfg, base_dir=base_dir)
    print(f"Clusters: {len(clustered)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clustered, f, ensure_ascii=False, indent=2)
    print(f"Saved → {output_path}")
