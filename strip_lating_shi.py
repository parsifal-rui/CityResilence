"""去掉 lating.json 里各节点 name 末尾的「市」，便于与抽取结果里的「广州」等对表。"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def strip_trailing_shi(node: object) -> None:
    if isinstance(node, dict):
        n = node.get("name")
        if isinstance(n, str) and len(n) > 1 and n.endswith("市"):
            node["name"] = n[:-1]
        for ch in node.get("children", []):
            strip_trailing_shi(ch)
    elif isinstance(node, list):
        for x in node:
            strip_trailing_shi(x)


def main() -> None:
    base = Path(__file__).resolve().parent
    inp = Path(sys.argv[1]) if len(sys.argv) > 1 else base / "data" / "lating.json"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else inp

    data = json.loads(inp.read_text(encoding="utf-8"))
    strip_trailing_shi(data)
    out.write_text(json.dumps(data, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    print(f"written -> {out}")


if __name__ == "__main__":
    main()
