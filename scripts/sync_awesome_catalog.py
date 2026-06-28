#!/usr/bin/env python3
"""Sincroniza o catálogo de datasets a partir do awesome-health-datasets.

Busca o README do repositório, extrai os datasets agrupados por categoria
(cabeçalhos ## e itens de lista `- [nome](url) - descrição`) e grava um JSON
em core/data/benchmarks/awesome_catalog.json, exibido como referência na aba
BENCHMARKS.

Uso:
    python scripts/sync_awesome_catalog.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from urllib.request import urlopen

_REPO = "fabianofilho/awesome-health-datasets"
RAW_URLS = [
    f"https://raw.githubusercontent.com/{_REPO}/master/README.md",
    f"https://raw.githubusercontent.com/{_REPO}/main/README.md",
]
OUT = Path(__file__).resolve().parent.parent / "core" / "data" / "benchmarks" / "awesome_catalog.json"

# itens: - [nome](url) - descrição   (o separador pode ser " - " ou " — ")
_ITEM = re.compile(r"^\s*-\s*\[([^\]]+)\]\(([^)]+)\)\s*[-—]?\s*(.*)$")
_HEADER = re.compile(r"^(#{2,3})\s+(.*)$")


def _clean_category(text: str) -> str:
    # remove emojis/ícones e espaços das bordas
    return re.sub(r"[^\w\s&/À-ÿ()\-]", "", text).strip()


def parse(md: str) -> dict:
    catalog: dict[str, list[dict]] = {}
    current = None
    for line in md.splitlines():
        h = _HEADER.match(line)
        if h:
            title = _clean_category(h.group(2))
            # ignora seções utilitárias
            if any(k in title.lower() for k in ("license", "licença", "contribu", "platforms", "repositor")):
                current = None
            else:
                current = title
                catalog.setdefault(current, [])
            continue
        if current is None:
            continue
        m = _ITEM.match(line)
        if m:
            name, url, desc = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            if name and url.startswith("http"):
                catalog[current].append({"name": name, "url": url, "desc": desc})
    # remove categorias vazias
    return {k: v for k, v in catalog.items() if v}


def main() -> int:
    md = None
    for url in RAW_URLS:
        try:
            with urlopen(url, timeout=30) as r:
                md = r.read().decode("utf-8")
            break
        except Exception:
            continue
    if md is None:
        print(f"ERRO ao buscar README em {RAW_URLS}", file=sys.stderr)
        return 1

    catalog = parse(md)
    total = sum(len(v) for v in catalog.values())
    OUT.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Catálogo gravado em {OUT.relative_to(OUT.parents[3])}")
    print(f"{len(catalog)} categorias, {total} datasets")
    for cat, items in catalog.items():
        print(f"  {cat}: {len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
