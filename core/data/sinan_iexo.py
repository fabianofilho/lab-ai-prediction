"""SINAN-IEXO (intoxicação exógena) preprocessor."""
from __future__ import annotations
import pandas as pd

from core.data import rotulo

KEEP_COLS = [
    "NU_NOTIFIC", "DT_NOTIFIC", "DT_SIN_PRI", "DT_ENCERRA",
    "SG_UF_NOT", "ID_MUNICIP", "ID_MN_RESI",
    "NU_IDADE_N", "CS_SEXO", "CS_GESTANT", "CS_RACA", "CS_ESCOL_N",
    "ID_OCUPA_N",
    # Desfecho
    "CLASSI_FIN",       # 1=intoxicação confirmada, 2=só exposição, 3=reação adversa, 4=diagnóstico diferencial, 5=síndrome de abstinência, 9=ignorado
    "EVOLUCAO",         # evolução do caso: ver EVOLUCAO_MAPA abaixo
    "DT_OBITO",
    # Agente tóxico
    "AGENTE_TOX",       # categoria do agente
    "AGENTE_1", "AGENTE_2", "AGENTE_3",
    # Circunstância / intenção
    "CIRCUNSTAN",       # 1=acidental, 2=tentativa suicídio, 3=homicídio, 4=abuso, etc.
    "UTILIZACAO",
    # Via de exposição
    "VIA_1", "VIA_2", "VIA_3",
    # Atenção/local
    "TPATENDE",         # tipo de atendimento
    "HOSPITAL",         # hospitalizado
    # Trabalho
    "SIT_TRAB", "LOC_EXPO", "DOENCA_TRA",
]

# EVOLUCAO (campo 68, evolução do caso) e CLASSI_FIN (campo 65) no
# dicionário do SINAN-Intoxicação Exógena: PySUS 0.15.0,
# pysus/metadata/SINAN/IEXO.csv, conferido em 2026-09-26. Não conferido
# contra um IEXOBR bruto (a rede do DataSUS estava bloqueada).
EVOLUCAO_MAPA = {
    "1": "cura_sem_sequela",
    "2": "cura_com_sequela",
    "3": "obito_por_intoxicacao",
    "4": "obito_outra_causa",
    "5": "perda_de_seguimento",
    "9": "ignorado",
}
EVOLUCAO_CURA_SEM_SEQUELA = {"1"}
EVOLUCAO_ADVERSO = {"2", "3"}  # cura com sequela e óbito por intoxicação exógena
# Censura do desfecho adverso: óbito por outra causa (risco competitivo) e
# perda de seguimento (desfecho desconhecido). Ignorado (9) e em branco
# também ficam sem rótulo. Nenhum deles vira 0.
EVOLUCAO_CENSURA = {"4", "5"}
CIRCUNSTAN_SUICIDIO = "2"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[cols].copy()

    for col in ["DT_NOTIFIC", "DT_SIN_PRI", "DT_ENCERRA", "DT_OBITO"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "NU_IDADE_N" in df.columns:
        df["idade_anos"] = _decode_idade(df["NU_IDADE_N"])

    if "HOSPITAL" in df.columns:
        df["hospitalizado"] = (df["HOSPITAL"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True) == "1").astype(int)

    # Alvo com censura: 1 adverso, 0 cura sem sequela e NaN para censura,
    # ignorado ou em branco (ver drop_censored)
    if "EVOLUCAO" in df.columns:
        evolucao = rotulo.codigo(df["EVOLUCAO"])
        df["desfecho_adverso"] = rotulo.alvo_com_censura(
            evolucao, EVOLUCAO_ADVERSO, EVOLUCAO_CURA_SEM_SEQUELA,
        )
        df["obito"] = (evolucao == "3").astype(int)    # óbito por intoxicação exógena
        df["sequela"] = (evolucao == "2").astype(int)  # cura com sequela

    if "CIRCUNSTAN" in df.columns:
        df["tentativa_suicidio"] = (
            df["CIRCUNSTAN"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True) == CIRCUNSTAN_SUICIDIO
        ).astype(int)

    for col in ["CS_SEXO", "CS_RACA", "CS_ESCOL_N"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

    return df


def filter_confirmed(df: pd.DataFrame) -> pd.DataFrame:
    if "CLASSI_FIN" in df.columns:
        return df[df["CLASSI_FIN"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True) == "1"].copy()
    return df


def drop_censored(df: pd.DataFrame, target_col: str = "desfecho_adverso") -> pd.DataFrame:
    """Tira da coorte a censura (EVOLUCAO 4 e 5), o ignorado (9) e o em
    branco, com a contagem em ``df.attrs``."""
    return rotulo.excluir_sem_rotulo(df, target_col, "EVOLUCAO", EVOLUCAO_CENSURA)


def _decode_idade(serie: pd.Series) -> pd.Series:
    s = pd.to_numeric(serie, errors="coerce")
    unit = (s // 1000).astype("Int64")
    value = (s % 1000).astype(float)
    age = pd.Series(index=serie.index, dtype=float)
    age[unit == 4] = value[unit == 4]
    age[unit == 3] = value[unit == 3] / 12
    age[unit == 2] = value[unit == 2] / 365
    age[unit == 1] = value[unit == 1] / 8760
    return age
