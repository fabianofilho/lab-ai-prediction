"""SINAN-TB (tuberculosis notifications) preprocessor."""

from __future__ import annotations

import logging

import pandas as pd

from core.data import rotulo

logger = logging.getLogger(__name__)


KEEP_COLS = [
    "NU_NOTIFIC",          # notification number
    "DT_NOTIFIC",          # notification date
    "DT_DIAG",             # diagnosis date
    "DT_ENCERRA",          # case closure date
    "SEM_NOT",             # epidemiological week
    # Patient
    "NU_IDADE_N",          # age
    "CS_SEXO",
    "CS_RACA",
    "CS_ESCOL_N",          # education
    "ID_MN_RESI",          # municipality of residence
    "SG_UF_NOT",           # state of notification
    # Clinical
    "FORMA",               # clinical form (pulmonary/extrapulmonary/both)
    "BACILOSC_E",          # initial sputum smear
    "CULTURA_ES",          # sputum culture
    "HIV",                 # HIV co-infection status
    "AGRAVAIDS",           # AIDS complication
    "TRAT_SUPER",          # directly observed treatment (DOT)
    "SITUA_ENCE",          # situação de encerramento: ver SITUA_ENCE_MAPA abaixo
    "TRATAMENTO",          # entry type (1=new case, 2=relapse, 3=re-entry after abandonment, ...)
    "RAIOX_TORA",          # chest X-ray result
    # Identifiers
    "NM_PACIENT",
    "NM_MAE_PAC",
    "DT_NASC",
    "CNS_1",
    "TP_NOT",
]

# SITUA_ENCE (situação de encerramento) no dicionário vigente do SINAN-TB.
# Mesmo mapa de sinan-continual-learning (core/diseases/tuberculose.py,
# ENCERRAMENTO), do data_dict deste app e do PySUS (5 = transferência,
# 7 = TB-DR). Não conferido contra um TUBEBR bruto: a rede estava bloqueada
# quando o mapa foi corrigido. Os valores brutos podem vir com espaços.
SITUA_ENCE_MAPA = {
    "1": "cura",
    "2": "abandono",
    "3": "obito_por_tb",
    "4": "obito_outras_causas",
    "5": "transferencia",
    "6": "mudanca_de_diagnostico",
    "7": "tb_drogarresistente",
    "8": "mudanca_de_esquema",
    "9": "falencia",
    "10": "abandono_primario",
}
SITUA_CURA = {"1"}
SITUA_ABANDONO = {"2", "10"}   # abandono e abandono primário
SITUA_OBITO = {"3", "4"}       # óbito por TB e óbito por outras causas
SITUA_FALENCIA = {"9"}
# Censura: o caso saiu de vista (transferência) ou deixou de ser o tratamento
# acompanhado (mudança de diagnóstico, TB-DR, mudança de esquema). Esses casos
# saem da coorte do desfecho e nunca viram 0.
SITUA_CENSURA = {"5", "6", "7", "8"}

# No abandono, o óbito (3 e 4) também é censura: risco competitivo, por
# decisão do lab. Quem morreu antes de encerrar o tratamento não teve como
# abandoná-lo, e contá-lo como 0 ensinaria ao modelo que ele não abandonaria.
SITUA_CENSURA_ABANDONO = SITUA_CENSURA | SITUA_OBITO

# Negativos de cada alvo. O que não é positivo, negativo nem censura (em
# branco, código fora do dicionário) também fica sem rótulo.
SITUA_NEGATIVO_ABANDONO = SITUA_CURA | SITUA_FALENCIA
SITUA_NEGATIVO_OBITO = SITUA_CURA | SITUA_FALENCIA

# No óbito, o abandono (2 e 10) também é censura, pela mesma decisão do lab:
# quem abandonou saiu de vista antes do encerramento, e contá-lo como 0
# afirmaria que ele não morreu.
SITUA_CENSURA_OBITO = SITUA_CENSURA | SITUA_ABANDONO
CENSURA_POR_ALVO = {
    "abandono": SITUA_CENSURA_ABANDONO,
    "obito_tb": SITUA_CENSURA_OBITO,
}


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize a raw SINAN-TB DataFrame."""
    cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[cols].copy()

    # Dates — dbfread already returns date objects; coerce gracefully
    for col in ["DT_NOTIFIC", "DT_DIAG", "DT_ENCERRA", "DT_NASC"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Age in years
    if "NU_IDADE_N" in df.columns:
        df["idade_anos"] = _decode_idade_sinan(df["NU_IDADE_N"])

    # Alvos binários com censura: 1 positivo, 0 negativo e NaN para censura
    # ou código fora do dicionário (ver drop_censored).
    if "SITUA_ENCE" in df.columns:
        situacao = rotulo.codigo(df["SITUA_ENCE"])
        df["abandono"] = rotulo.alvo_com_censura(situacao, SITUA_ABANDONO, SITUA_NEGATIVO_ABANDONO)
        df["cura"] = situacao.isin(SITUA_CURA).astype(int)
        df["obito_tb"] = rotulo.alvo_com_censura(situacao, SITUA_OBITO, SITUA_NEGATIVO_OBITO)

    # DOT (tratamento supervisionado)
    if "TRAT_SUPER" in df.columns:
        df["dot"] = (df["TRAT_SUPER"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True) == "1").astype(int)

    # HIV positive flag
    if "HIV" in df.columns:
        df["hiv_pos"] = (df["HIV"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True) == "1").astype(int)

    # Identifiers
    for col in ["NM_PACIENT", "NM_MAE_PAC", "CNS_1"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True).str.upper().replace("NAN", "")

    return df


def filter_closed_cases(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only cases with a definitive outcome (closure date present)."""
    if "DT_ENCERRA" in df.columns:
        return df[df["DT_ENCERRA"].notna()].copy()
    return df


def drop_censored(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Tira da coorte os casos sem rótulo do desfecho `target_col`.

    Sai quem tem SITUA_ENCE de censura do alvo (``CENSURA_POR_ALVO``) ou
    código fora do dicionário (em branco, ignorado). As contagens ficam em
    ``df.attrs["exclusoes_rotulo"]`` e no log (``rotulo.excluir_sem_rotulo``).
    """
    return rotulo.excluir_sem_rotulo(
        df, target_col, "SITUA_ENCE", CENSURA_POR_ALVO.get(target_col, SITUA_CENSURA), log=logger,
    )


def _decode_idade_sinan(serie: pd.Series) -> pd.Series:
    """Decode SINAN NU_IDADE_N to years (similar logic to SIM IDADE)."""
    s = pd.to_numeric(serie, errors="coerce")
    unit = (s // 1000).astype("Int64")
    value = (s % 1000).astype(float)
    age = pd.Series(index=serie.index, dtype=float)
    age[unit == 4] = value[unit == 4]         # years
    age[unit == 3] = value[unit == 3] / 12    # months
    age[unit == 2] = value[unit == 2] / 365   # days
    age[unit == 1] = value[unit == 1] / 8760  # hours
    return age
