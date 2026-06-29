"""Chronic Kidney Disease — UCI 2015. CSV vendorizado (sem rede em runtime).

A UCI distribui este dataset apenas como ARFF dentro de um .rar, que exige
extrator externo e parsing especial. Para manter o loader sem dependências e
reproduzível, o ARFF oficial foi parseado uma vez (missing '?' → NaN, classe
ckd/notckd → target 1/0) e salvo em `data/ckd.csv`. Provenance: UCI dataset 336.

Observação: 1 das 400 linhas do ARFF original está corrompida (campo extra) e
foi descartada; o CSV tem 399 registros.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

_CSV = Path(__file__).parent / "data" / "ckd.csv"

DICT_META = {
    "age":    {"label": "Idade",                "desc": "Anos", "type": "Numérica"},
    "bp":     {"label": "Pressão arterial",     "desc": "mm/Hg", "type": "Numérica"},
    "sg":     {"label": "Densidade urinária",   "desc": "Specific gravity", "type": "Ordinal"},
    "al":     {"label": "Albumina (urina)",     "desc": "0-5", "type": "Ordinal"},
    "su":     {"label": "Açúcar (urina)",       "desc": "0-5", "type": "Ordinal"},
    "rbc":    {"label": "Hemácias (urina)",     "desc": "normal/abnormal", "type": "Categórica"},
    "pc":     {"label": "Piócitos",             "desc": "normal/abnormal", "type": "Categórica"},
    "pcc":    {"label": "Cilindros piócitos",   "desc": "present/notpresent", "type": "Categórica"},
    "ba":     {"label": "Bactérias",            "desc": "present/notpresent", "type": "Categórica"},
    "bgr":    {"label": "Glicemia aleatória",   "desc": "mg/dL", "type": "Numérica"},
    "bu":     {"label": "Ureia sanguínea",      "desc": "mg/dL", "type": "Numérica"},
    "sc":     {"label": "Creatinina sérica",    "desc": "mg/dL", "type": "Numérica"},
    "sod":    {"label": "Sódio",                "desc": "mEq/L", "type": "Numérica"},
    "pot":    {"label": "Potássio",             "desc": "mEq/L", "type": "Numérica"},
    "hemo":   {"label": "Hemoglobina",          "desc": "g/dL", "type": "Numérica"},
    "pcv":    {"label": "Hematócrito",          "desc": "Packed cell volume", "type": "Numérica"},
    "wbcc":   {"label": "Leucócitos",           "desc": "células/mm³", "type": "Numérica"},
    "rbcc":   {"label": "Hemácias (sangue)",    "desc": "milhões/mm³", "type": "Numérica"},
    "htn":    {"label": "Hipertensão",          "desc": "yes/no", "type": "Categórica"},
    "dm":     {"label": "Diabetes mellitus",    "desc": "yes/no", "type": "Categórica"},
    "cad":    {"label": "Doença coronariana",   "desc": "yes/no", "type": "Categórica"},
    "appet":  {"label": "Apetite",              "desc": "good/poor", "type": "Categórica"},
    "pe":     {"label": "Edema de membros",     "desc": "yes/no", "type": "Categórica"},
    "ane":    {"label": "Anemia",               "desc": "yes/no", "type": "Categórica"},
    "target": {"label": "Doença renal crônica", "desc": "1=DRC, 0=sem DRC", "type": "Derivada"},
}


def load() -> pd.DataFrame:
    """Lê o CSV vendorizado e devolve com target binário (ckd=1, notckd=0)."""
    df = pd.read_csv(_CSV)
    df["target"] = df["target"].astype(int)
    return df
