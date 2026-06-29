"""Parkinsons (Oxford) — UCI. CSV público com cabeçalho, sem credencial."""
from __future__ import annotations

import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"

DICT_META = {
    "target": {"label": "Doença de Parkinson", "desc": "status (1=Parkinson, 0=saudável)", "type": "Derivada"},
}


def load() -> pd.DataFrame:
    """Baixa Parkinsons e devolve com target binário (status → target).

    195 gravações de voz, 22 features acústicas (jitter, shimmer, HNR, etc).
    A coluna `name` (identificador da gravação) é REMOVIDA. `status`: 1 = doente.
    """
    df = pd.read_csv(URL)
    df = df.drop(columns=["name"], errors="ignore")
    df = df.rename(columns={"status": "target"})
    df["target"] = df["target"].astype(int)
    return df
