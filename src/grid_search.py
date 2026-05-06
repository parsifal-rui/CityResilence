"""
DBSCAN 参数网格搜索脚本

用途：
- 不重新调用 DeepSeek。
- 直接复用已经保存的 test_results/results.json。
- 对 T_max / S_max / eps / w_s 做多组全局实验。
- 每组输出一个 cluster json，并汇总 summary.csv/json。

用法：
  python grid_search_cluster.py results/run1/results.json cluster_grid_runs
  python grid_search_cluster.py results/run*/results.json cluster_grid_runs

注意：
- 默认导入同目录下的 cluster_events_cpu.py。
- 如果 cluster_events_cpu.py 放在 src/ 下，请相应调整 import 或 PYTHONPATH。
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import List

import numpy as np

from cluster_events_cpu import (
    Cfg,
    cluster_events,
    flatten_records,
    flatten_records_multi,
    load_discard_ids,
    summarize_clusters,
)


# ─── 参数网格 ──────────────────────────────────────────────────────────────────

T_LIST = [1, 2, 3]
S_LIST = [100.0, 200.0, 300.0]
EPS_LIST = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]
WS_LIST = [0.20, 0.30, 0.40]

# min_samples 按教授建议固定为 3；如需测试可改成 [2, 3, 4]
MIN_SAMPLES_LIST = [3]


# ─── 额外 summary 指标 ─────────────────────────────────────────────────────────

def enrich_summary(summary: dict, clusters: List[dict]) -> dict:
    if not clusters:
        summary.update({
            "avg_city_count": 0.0,
            "max_city_count": 0,
            "avg_date_span_days": 0.0,
            "max_date_span_days": 0,
            "clusters_single_article_ratio": 0.0,
        })
        return summary

    city_counts = [len(c.get("cities", [])) for c in clusters]
    article_counts = [c.get("n_articles", 0) for c in clusters]

    date_spans = []
    for c in clusters:
        try:
            d0 = np.datetime64(c["date_start"])
            d1 = np.datetime64(c["date_end"])
            date_spans.append(int((d1 - d0).astype(int)))
        except Exception:
            date_spans.append(0)

    summary.update({
        "avg_city_count": float(np.mean(city_counts)),
        "max_city_count": int(max(city_counts)),
        "avg_date_span_days": float(np.mean(date_spans)),
        "max_date_span_days": int(max(date_spans)),
        "clusters_single_article_ratio": float(sum(a == 1 for a in article_counts) / len(article_counts)),
    })
    return summary


# ─── 数据加载 ──────────────────────────────────────────────────────────────────

def load_records(input_paths: List[str], base_dir: Path) -> list[dict]:
    cfg0 = Cfg()
    discard_ids = load_discard_ids(base_dir / cfg0.discard_path)
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
    return records


# ─── 主程序 ────────────────────────────────────────────────────────────────────

def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: python grid_search_cluster.py <input1.json> [input2.json ...] <out_dir>")
        sys.exit(1)

    *input_paths, out_dir = sys.argv[1:]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 如果你的脚本位于 src/cluster_events_cpu.py，base_dir 通常是项目根目录
    base_dir = Path(__file__).parent.parent

    records = load_records(input_paths, base_dir=base_dir)

    summaries = []
    run_idx = 0

    for T_max, S_max, eps, w_s, min_samples in product(
        T_LIST, S_LIST, EPS_LIST, WS_LIST, MIN_SAMPLES_LIST
    ):
        run_idx += 1

        # 让时间和空间平分剩余权重
        rest = 1.0 - w_s
        w_t = rest / 2.0
        w_p = rest / 2.0

        cfg = Cfg(
            T_max=T_max,
            S_max=S_max,
            w_s=w_s,
            w_t=w_t,
            w_p=w_p,
            eps=eps,
            min_samples=min_samples,
            include_members=False,  # grid search 阶段先关掉，节省输出体积
        )

        print(
            f"\n[run {run_idx}] "
            f"T={T_max}, S={S_max}, eps={eps}, "
            f"w_s={w_s}, w_t={w_t:.2f}, w_p={w_p:.2f}, min_samples={min_samples}"
        )

        clusters = cluster_events(records, cfg=cfg, base_dir=base_dir)
        summary = summarize_clusters(clusters, n_input_records=len(records))
        summary = enrich_summary(summary, clusters)

        run_name = (
            f"run{run_idx:03d}"
            f"_T{T_max}"
            f"_S{int(S_max)}"
            f"_eps{eps:.2f}"
            f"_ws{w_s:.2f}"
            f"_ms{min_samples}"
        )

        summary.update({
            "run_name": run_name,
            "T_max": T_max,
            "S_max": S_max,
            "eps": eps,
            "w_s": w_s,
            "w_t": w_t,
            "w_p": w_p,
            "min_samples": min_samples,
            "output_file": f"{run_name}.json",
        })

        summaries.append(summary)

        # 每组保存一个较轻的 cluster 输出
        out = {
            "config": asdict(cfg),
            "summary": summary,
            "clusters": clusters,
        }
        with open(out_dir / f"{run_name}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        print("summary:", json.dumps(summary, ensure_ascii=False, indent=2))

    # 保存 summary.json
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)

    # 保存 summary.csv
    if summaries:
        fieldnames = list(summaries[0].keys())
        with open(out_dir / "summary.csv", "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)

    print(f"\nSaved grid search results → {out_dir}")
    print(f"Total runs: {len(summaries)}")


if __name__ == "__main__":
    main()
