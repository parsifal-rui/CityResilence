"""
时空语义三维联合聚类（DBSCAN）
输入：JSON 中展平的事件记录列表，每条记录 = {event_text, event_type, city, date, article_id}
输出：clustered_events 列表
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN


# ─── 超参数配置 ────────────────────────────────────────────────────────────────

@dataclass
class Cfg:
    T_max: int        = 7          # 时间阈值（天）
    S_max: float      = 150.0      # 空间阈值（km）
    w_s: float        = 0.5        # 语义权重
    w_t: float        = 0.25       # 时间权重
    w_p: float        = 0.25       # 空间权重
    eps: float        = 0.35       # DBSCAN epsilon
    min_samples: int  = 2          # DBSCAN min_samples
    model_path: str   = "models/bge-m3"
    lating_path: str  = "data/lating.json"
    device: str       = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


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
    cfg:   Cfg,
) -> np.ndarray:
    dev = cfg.device
    embs, days, lats, lons = embs.to(dev), days.to(dev), lats.to(dev), lons.to(dev)

    # 语义距离：1 - cosine（embs 已归一化，点积 = cosine；clamp 消除浮点负值）
    d_sem   = torch.clamp(1.0 - embs @ embs.T, 0.0, 2.0)                  # (N, N)

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

def flatten_records(data: dict) -> List[dict]:
    """
    从 results.json 的顶层 dict 展平为事件记录列表。
    过滤条件：location 为空列表的事件。
    date 优先取 date_from，缺失时回退到文章 publish_date。
    article_id 取父级文章的 article_id（事件内字段固定为 null）。
    """
    records = []
    for art in data.get("results", []):
        art_id       = art["article_id"]
        publish_date = art.get("publish_date", "")
        for ev in art.get("extraction", {}).get("events", []):
            if not ev.get("location"):          # 过滤 location 为空
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

    # ── 按 event_type 分组聚类 ──
    type_groups: Dict[str, List[int]] = defaultdict(list)
    for i, t in enumerate(types):
        type_groups[t].append(i)

    global_labels = np.full(len(records), -1, dtype=np.int32)
    global_offset = 0

    for etype, idxs_list in type_groups.items():
        idxs = np.array(idxs_list)
        n = len(idxs)
        if n < cfg.min_samples:
            continue

        sub_embs  = all_embs[[emb_rows[i] for i in idxs]]
        sub_days  = days_all[idxs]
        sub_lats  = lats_all[idxs]
        sub_lons  = lons_all[idxs]

        D = build_dist_matrix(sub_embs, sub_days, sub_lats, sub_lons, cfg)
        D_finite = np.where(np.isinf(D), 1e9, D)   # DBSCAN 需要有限值

        local_labels = DBSCAN(
            eps=cfg.eps, min_samples=cfg.min_samples, metric="precomputed"
        ).fit_predict(D_finite)

        valid = local_labels >= 0
        if valid.any():
            global_labels[idxs[valid]] = local_labels[valid] + global_offset
            global_offset += int(local_labels[valid].max()) + 1

        n_clusters = int(valid.any()) and (int(local_labels[valid].max()) + 1)
        print(f"[{etype}] n={n}, clusters={n_clusters}, noise={int((~valid).sum())}")

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
            "representative_event_text": Counter(m_texts).most_common(1)[0][0],
            "event_type":               Counter(m_types).most_common(1)[0][0],
            "cities":                   sorted({c for c in m_cities if c}),
            "date_start":               days_to_str(min(m_days)),
            "date_end":                 days_to_str(max(m_days)),
            "article_ids":              sorted(set(m_aids)),
            "n_records":                len(members),
            "n_articles":               len(set(m_aids)),
        })

    return clustered_events


# ─── 入口 ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    input_path  = sys.argv[1] if len(sys.argv) > 1 else "results/run1/results.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "results/run1/clustered_events.json"
    base_dir    = str(Path(__file__).parent.parent)

    with open(input_path, encoding="utf-8") as f:
        raw = json.load(f)
    records = flatten_records(raw)
    print(f"Loaded {len(records)} records (after filtering empty-location events), device={Cfg().device}")

    clustered = cluster_events(records, base_dir=base_dir)
    print(f"Clusters: {len(clustered)}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clustered, f, ensure_ascii=False, indent=2)
    print(f"Saved → {output_path}")
