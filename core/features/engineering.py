"""Shared feature engineering utilities."""

from __future__ import annotations

import pandas as pd
import numpy as np


def as_category(series: pd.Series) -> pd.Series:
    """Variável nominal como texto, com o ausente preservado como NaN.

    O código bruto passa pela mesma normalização dos rótulos
    (``core.data.rotulo.codigo``): " 2", "2.0", 2 e "02" viram "2". Em branco,
    "nan" e "None" viram NaN, e não uma categoria a mais.

    Antes, cada desfecho convertia a categoria em código numérico com
    ``pd.Categorical(...).codes``. Isso inventava ordem numa variável nominal
    (o modelo linear e o discretizador liam 1 < 2 < 3) e apagava o ausente: no
    pandas 2 ele virava a categoria "nan", no pandas 3 virava o código -1. A
    coluna agora chega ao pipeline como categórica, e quem decide a codificação
    é o tratamento escolhido na tela (one-hot por padrão).
    """
    from core.data.rotulo import codigo

    texto = codigo(series)
    vazio = texto.eq("").to_numpy()
    return pd.Series(
        np.where(vazio, np.nan, texto.astype(object).to_numpy()),
        index=series.index,
        dtype=object,
    )


def encode_categoricals(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Marca as colunas como categóricas nominais (ver ``as_category``)."""
    df = df.copy()
    for col in cat_cols:
        if col in df.columns:
            df[col] = as_category(df[col])
    return df


def icd10_chapter(series: pd.Series) -> pd.Series:
    """Capítulo da CID-10 (primeira letra), como categoria nominal."""
    return as_category(series).str[0].str.upper()


def icd10_block(series: pd.Series) -> pd.Series:
    """Bloco de 3 caracteres da CID-10, como categoria nominal."""
    return as_category(series).str[:3].str.upper()


def age_group(age_series: pd.Series, bins=None, labels=None) -> pd.Series:
    """Convert continuous age to age-group categories (numeric codes)."""
    if bins is None:
        bins = [0, 1, 5, 18, 40, 60, 80, 120]
        labels = [0, 1, 2, 3, 4, 5, 6]
    return pd.cut(age_series, bins=bins, labels=labels, right=False).astype(float)


def flag_missing(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Add binary indicator columns for missingness."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def clip_outliers(df: pd.DataFrame, col: str, lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
    """Clip a column at quantile bounds."""
    df = df.copy()
    if col in df.columns:
        lo = df[col].quantile(lower_q)
        hi = df[col].quantile(upper_q)
        df[col] = df[col].clip(lo, hi)
    return df
