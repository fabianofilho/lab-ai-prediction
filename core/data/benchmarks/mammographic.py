"""Mammographic Mass — UCI. CSV público, sem credencial."""
from __future__ import annotations

import pandas as pd

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "mammographic-masses/mammographic_masses.data"
)

_COLUMNS = ["birads", "age", "shape", "margin", "density", "severity"]

DICT_META = {
    "age":     {"label": "Idade",              "desc": "Anos", "type": "Numérica"},
    "shape":   {"label": "Forma da massa",     "desc": "1=redonda, 2=oval, 3=lobular, 4=irregular", "type": "Categórica"},
    "margin":  {"label": "Margem da massa",    "desc": "1=circunscrita ... 5=espiculada", "type": "Categórica"},
    "density": {"label": "Densidade da massa", "desc": "1=alta ... 4=contendo gordura", "type": "Ordinal"},
    "target":  {"label": "Massa maligna",      "desc": "1=maligna, 0=benigna", "type": "Derivada"},
}


def load() -> pd.DataFrame:
    """Baixa Mammographic Mass e devolve com target binário (severity → target).

    A coluna `birads` (avaliação BI-RADS do radiologista) é REMOVIDA: é a própria
    estimativa de malignidade do laudo e a documentação UCI a marca como não
    preditiva, então vaza o desfecho. Missing marcados como '?' viram NaN.
    """
    df = pd.read_csv(URL, header=None, names=_COLUMNS, na_values="?")
    df = df.drop(columns=["birads"])
    df = df.rename(columns={"severity": "target"})
    df["target"] = df["target"].astype(int)
    return df
