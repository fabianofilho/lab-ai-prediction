"""SINAN-TB (tuberculosis notifications) preprocessor."""

from __future__ import annotations

import logging

import pandas as pd

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
# Censura: o caso saiu de vista (transferência) ou deixou de ser o tratamento
# acompanhado (mudança de diagnóstico, TB-DR, mudança de esquema). Esses casos
# saem da coorte do desfecho e nunca viram 0.
SITUA_CENSURA = {"5", "6", "7", "8"}


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
        situacao = _codigo(df["SITUA_ENCE"])
        df["abandono"] = _alvo_com_censura(situacao, SITUA_ABANDONO)
        df["cura"] = situacao.isin(SITUA_CURA).astype(int)
        df["obito_tb"] = _alvo_com_censura(situacao, SITUA_OBITO)

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

    Sai quem tem SITUA_ENCE de censura (5 a 8) ou código fora do dicionário
    (em branco, ignorado). As contagens ficam em
    ``df.attrs["exclusoes_rotulo"]`` e no log, para que a exclusão seja
    visível em vez de silenciosa.
    """
    if target_col not in df.columns:
        return df
    sem_rotulo = df[target_col].isna()
    if "SITUA_ENCE" in df.columns:
        censura = sem_rotulo & _codigo(df["SITUA_ENCE"]).isin(SITUA_CENSURA)
    else:
        censura = pd.Series(False, index=df.index)
    n_censura = int(censura.sum())
    n_sem_codigo = int(sem_rotulo.sum()) - n_censura

    out = df.loc[~sem_rotulo].copy()
    out[target_col] = out[target_col].astype(int)
    out.attrs["exclusoes_rotulo"] = {
        "censura": n_censura,
        "sem_codigo": n_sem_codigo,
        "mantidos": len(out),
    }
    if n_censura or n_sem_codigo:
        logger.warning(
            "%s: %d casos censurados (SITUA_ENCE 5 a 8) e %d sem código válido "
            "excluídos da coorte; %d mantidos.",
            target_col, n_censura, n_sem_codigo, len(out),
        )
    return out


def _codigo(serie: pd.Series) -> pd.Series:
    """Normaliza o código bruto do SINAN ("2 ", "2.0", 2) para "2"."""
    return serie.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)


def _alvo_com_censura(situacao: pd.Series, positivos: set[str]) -> pd.Series:
    """1.0 para positivos, 0.0 para os demais encerramentos e NaN para censura
    ou código fora de SITUA_ENCE_MAPA."""
    alvo = situacao.isin(positivos).astype(float)
    rotulado = situacao.isin(SITUA_ENCE_MAPA.keys()) & ~situacao.isin(SITUA_CENSURA)
    return alvo.where(rotulado)


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
