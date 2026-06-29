"""Datasets brasileiros (DATASUS) materializados para o BenchLab.

Samples pré-construídos a partir de dados públicos do DATASUS (SINASC, SIH,
SINAN) usando o próprio pipeline de desfechos do app (core.outcomes +
CohortBuilder). Cada arquivo `data/br/<key>.parquet` contém as features
sugeridas do desfecho + a coluna alvo binária `target`. UF/ano e contagens
ficam em `data/br/manifest.json`.

Por que vendorizar: baixar do DATASUS é lento, é por UF/ano e pode cair no
upload manual se o mirror falhar. Para o BenchLab (comparação rápida e offline)
os samples são materializados uma vez. Para rodar o desfecho com dados frescos
e na escala completa, use a página de Análise (fluxo DATASUS ao vivo).
"""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import pandas as pd

_DIR = Path(__file__).parent / "data" / "br"

MANIFEST: dict = json.loads((_DIR / "manifest.json").read_text(encoding="utf-8"))


def _load(key: str) -> pd.DataFrame:
    """Lê o sample parquet do desfecho e devolve com target binário."""
    df = pd.read_parquet(_DIR / f"{key}.parquet")
    df["target"] = df["target"].astype(int)
    return df


def loader_for(key: str):
    """Devolve um loader sem argumentos para o desfecho `key` (contrato Benchmark)."""
    return partial(_load, key)
