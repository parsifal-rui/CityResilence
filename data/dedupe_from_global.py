import json
import re
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
GLOBAL_PATH = BASE / "results" / "global_clustered.json"
DISCARD_PATH = BASE / "data" / "discard.txt"
OUT_PATH = BASE / "data" / "dedupe_candidates.txt"

# 仅在小簇里做“同报道”比对
MAX_CLUSTER_ARTICLES = 4  # 即 n_articles < 5
CONTENT_SIM_THRESHOLD = 0.92


def aid_num(aid: str) -> int:
    m = re.search(r"\d+", aid or "")
    return int(m.group()) if m else 10**9


def load_discard_ids(path: Path) -> set[str]:
    ids = set()
    if not path.exists():
        return ids
    pat = re.compile(r'"article_id":\s*"([^"]+)"')
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pat.search(line)
        if m:
            ids.add(m.group(1))
    return ids


def normalize_text(s: str) -> str:
    s = (s or "").strip().lower()
    # 仅保留中英文和数字，去掉空白/标点，提升同文改写匹配稳定性
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", s)


def text_similarity(a: str, b: str) -> float:
    na, nb = normalize_text(a), normalize_text(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def load_article_content_index(results_dir: Path) -> dict[str, dict]:
    """
    建立 article_id -> {"run_id": runX, "content": content_preview} 索引。
    若同一 article_id 在多个 run 出现，保留首次出现。
    """
    index: dict[str, dict] = {}
    run_files = sorted(results_dir.glob("run*/results.json"))
    for run_file in run_files:
        run_id = run_file.parent.name
        raw = json.loads(run_file.read_text(encoding="utf-8"))
        for item in raw.get("results", []):
            aid = item.get("article_id")
            if not aid or aid in index:
                continue
            index[aid] = {
                "run_id": run_id,
                "content": item.get("content_preview", "") or "",
            }
    return index


def main():
    clusters = json.loads(GLOBAL_PATH.read_text(encoding="utf-8"))
    discard_ids = load_discard_ids(DISCARD_PATH)
    article_index = load_article_content_index(BASE / "results")

    suggested_drop: set[str] = set()
    reasons: dict[str, list[str]] = {}
    checked_pairs = 0
    matched_pairs = 0

    for c in clusters:
        n_articles = int(c.get("n_articles", 0) or 0)
        if n_articles == 0 or n_articles > MAX_CLUSTER_ARTICLES:
            continue

        cid = c.get("cluster_id")
        aids = sorted(set(c.get("article_ids", [])), key=aid_num)
        if len(aids) < 2:
            continue

        for a, b in combinations(aids, 2):
            a_info = article_index.get(a)
            b_info = article_index.get(b)
            if not a_info or not b_info:
                continue
            checked_pairs += 1
            sim = text_similarity(a_info["content"], b_info["content"])
            if sim < CONTENT_SIM_THRESHOLD:
                continue

            matched_pairs += 1
            keep, drop = (a, b) if aid_num(a) < aid_num(b) else (b, a)
            suggested_drop.add(drop)
            reason = (
                f"cluster={cid} sim={sim:.3f} "
                f"{keep}({article_index[keep]['run_id']}) vs {drop}({article_index[drop]['run_id']})"
            )
            reasons.setdefault(drop, []).append(reason)

    new_candidates = sorted([x for x in suggested_drop if x not in discard_ids], key=aid_num)
    already_in_discard = sorted([x for x in suggested_drop if x in discard_ids], key=aid_num)

    lines = []
    lines.append(f"# checked_pairs={checked_pairs}")
    lines.append(f"# matched_pairs(sim>={CONTENT_SIM_THRESHOLD})={matched_pairs}")
    lines.append(f"# total_suggested_drop={len(suggested_drop)}")
    lines.append(f"# already_in_discard={len(already_in_discard)}")
    lines.append(f"# new_candidates={len(new_candidates)}")
    lines.append(f"# rule: cluster_n_articles<= {MAX_CLUSTER_ARTICLES}")
    lines.append("")

    for aid in new_candidates:
        hint = " | ".join(reasons.get(aid, [])[:5])
        lines.append(f'"article_id": "{aid}"  # {hint}')

    OUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"saved: {OUT_PATH}")
    print(f"new_candidates: {len(new_candidates)}")


if __name__ == "__main__":
    main()