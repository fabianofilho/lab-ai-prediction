"""Heart Failure Clinical Records — UCI 2020. CSV público, sem credencial."""
from __future__ import annotations

import pandas as pd

URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "00519/heart_failure_clinical_records_dataset.csv"
)

DICT_META = {
    "age":                       {"label": "Idade",                    "desc": "Anos", "type": "Numérica"},
    "anaemia":                   {"label": "Anemia",                   "desc": "Queda de hemácias/hemoglobina (0/1)", "type": "Categórica"},
    "creatinine_phosphokinase":  {"label": "CPK",                      "desc": "Creatina fosfoquinase no sangue (mcg/L)", "type": "Numérica"},
    "diabetes":                  {"label": "Diabetes",                 "desc": "0/1", "type": "Categórica"},
    "ejection_fraction":         {"label": "Fração de ejeção",         "desc": "% de sangue ejetado por batimento", "type": "Numérica"},
    "high_blood_pressure":       {"label": "Hipertensão",              "desc": "0/1", "type": "Categórica"},
    "platelets":                 {"label": "Plaquetas",                "desc": "kiloplaquetas/mL", "type": "Numérica"},
    "serum_creatinine":          {"label": "Creatinina sérica",        "desc": "mg/dL", "type": "Numérica"},
    "serum_sodium":              {"label": "Sódio sérico",             "desc": "mEq/L", "type": "Numérica"},
    "sex":                       {"label": "Sexo",                     "desc": "1=homem, 0=mulher", "type": "Categórica"},
    "smoking":                   {"label": "Tabagismo",                "desc": "0/1", "type": "Categórica"},
    "target":                    {"label": "Óbito no seguimento",      "desc": "DEATH_EVENT (0/1)", "type": "Derivada"},
}


def load() -> pd.DataFrame:
    """Baixa e devolve o dataset com target binário (DEATH_EVENT → target).

    A coluna `time` (tempo de seguimento até óbito/censura) é REMOVIDA: ela
    vaza o desfecho (sobreviventes têm seguimento longo), erro clássico neste
    dataset.
    """
    df = pd.read_csv(URL)
    df = df.drop(columns=["time"], errors="ignore")
    df = df.rename(columns={"DEATH_EVENT": "target"})
    df["target"] = df["target"].astype(int)
    return df
