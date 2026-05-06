#!/usr/bin/env python
from pathlib import Path
from typing import List, Dict, Any
import os
import json
import hashlib

import numpy as np
import pandas as pd
from FlagEmbedding import BGEM3FlagModel


MODEL_PATH = "/root/data/CityResilence/models/bge-m3"  # 本地已下载的 bge-m3 目录
MAX_LENGTH = 512


def load_triples(excel_path: Path) -> List[Dict[str, Any]]:
    df = pd.read_excel(excel_path)
    triples: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        e1 = str(row.get("entity_1", ""))
        t1 = str(row.get("type_1", ""))
        rel = str(row.get("relation_1", ""))
        e2 = str(row.get("entity_2", ""))
        t2 = str(row.get("type_2", ""))

        t1 = t1.strip("[]'\"")
        t2 = t2.strip("[]'\"")

        if e1 and e2 and rel:
            triples.append(
                {
                    "entity_1": e1,
                    "type_1": t1,
                    "relation": rel,
                    "entity_2": e2,
                    "type_2": t2,
                    "text": f"{e1} {rel} {e2}",
                }
            )

    return triples


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parent
    data_dir = root.parent / "data"

    kb_excel = (data_dir / "graph_database_export.xlsx").resolve()
    kb_triples_path = (data_dir / "kb_triples.jsonl").resolve()
    kb_embeddings_path = (data_dir / "kb_embeddings.npy").resolve()
    kb_meta_path = (data_dir / "kb_meta.json").resolve()

    print(f"读取知识库 Excel：{kb_excel}")
    triples = load_triples(kb_excel)
    print(f"共加载三元组：{len(triples)} 条")

    texts = [t["text"] for t in triples]

    print("加载 BGE-M3 模型...")
    model = BGEM3FlagModel(MODEL_PATH, device="cuda", use_fp16=True)

    print("开始向量化知识库三元组...")
    out = model.encode(texts, max_length=MAX_LENGTH, return_dense=True)
    dense_vecs = np.array(out["dense_vecs"], dtype=np.float32)
    print(f"向量化完成，形状：{dense_vecs.shape}")

    print(f"保存三元组到：{kb_triples_path}")
    with kb_triples_path.open("w", encoding="utf-8") as f:
        for t in triples:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    print(f"保存向量到：{kb_embeddings_path}")
    np.save(kb_embeddings_path, dense_vecs)

    excel_hash = sha256_file(kb_excel)
    meta = {
        "kb_excel_path": str(kb_excel),
        "kb_excel_sha256": excel_hash,
        "kb_excel_mtime": os.path.getmtime(kb_excel),
        "model_path": MODEL_PATH,
        "max_length": MAX_LENGTH,
        "num_triples": len(triples),
        "embedding_dim": int(dense_vecs.shape[1]) if dense_vecs.size > 0 else 0,
    }

    print(f"保存元信息到：{kb_meta_path}")
    with kb_meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("KB 向量构建完成。")


if __name__ == "__main__":
    main()