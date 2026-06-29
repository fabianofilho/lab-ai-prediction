"""Indian Liver Patient Dataset (ILPD) — UCI. CSV público, sem credencial."""
from __future__ import annotations

import pandas as pd

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv"
)

_COLUMNS = [
    "age", "gender", "tb", "db", "alkphos", "sgpt", "sgot",
    "tp", "alb", "ag_ratio", "selector",
]

DICT_META = {
    "age":      {"label": "Idade",                    "desc": "Anos", "type": "Numérica"},
    "gender":   {"label": "Sexo",                     "desc": "Male/Female", "type": "Categórica"},
    "tb":       {"label": "Bilirrubina total",        "desc": "mg/dL", "type": "Numérica"},
    "db":       {"label": "Bilirrubina direta",       "desc": "mg/dL", "type": "Numérica"},
    "alkphos":  {"label": "Fosfatase alcalina",       "desc": "UI/L", "type": "Numérica"},
    "sgpt":     {"label": "ALT (SGPT)",               "desc": "Alanina aminotransferase (UI/L)", "type": "Numérica"},
    "sgot":     {"label": "AST (SGOT)",               "desc": "Aspartato aminotransferase (UI/L)", "type": "Numérica"},
    "tp":       {"label": "Proteínas totais",         "desc": "g/dL", "type": "Numérica"},
    "alb":      {"label": "Albumina",                 "desc": "g/dL", "type": "Numérica"},
    "ag_ratio": {"label": "Razão albumina/globulina", "desc": "A/G ratio", "type": "Numérica"},
    "target":   {"label": "Doença hepática",          "desc": "1=paciente hepático, 0=saudável", "type": "Derivada"},
}


def load() -> pd.DataFrame:
    """Baixa ILPD e devolve com target binário (selector → target).

    Original: `selector` 1 = paciente hepático, 2 = saudável. Recodificado para
    1 = doença, 0 = ausência. Poucos valores ausentes em `ag_ratio` (NaN).
    """
    df = pd.read_csv(URL, header=None, names=_COLUMNS)
    df["target"] = (df["selector"] == 1).astype(int)
    df = df.drop(columns=["selector"])
    return df
