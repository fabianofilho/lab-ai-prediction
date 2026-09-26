"""Rótulo binário com censura a partir de um campo de código do SINAN.

É o mecanismo que o abandono de TB passou a usar na Onda 0, generalizado
para os outros desfechos do SINAN:

- o preprocessador marca 1.0 nos códigos positivos, 0.0 nos negativos e NaN
  em todo o resto: censura, ignorado, em branco e código fora do dicionário;
- o desfecho tira esses casos da coorte com ``excluir_sem_rotulo``, que
  registra a contagem em ``df.attrs["exclusoes_rotulo"]`` e no log;
- ``exigir_rotulo`` impede que um NaN que escapou vire 0 no ``get_target``.

Censura nunca vira 0, e ignorado também não.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_AUSENTES = {"", "nan", "none", "<na>", "nat"}


def codigo(serie: pd.Series) -> pd.Series:
    """Normaliza o código bruto do SINAN para texto.

    " 2", "2.0", 2 e "02" viram "2"; "10" continua "10" e "0" continua "0".
    Ausente vira "" (fora de qualquer dicionário, então sem rótulo).
    """
    # astype("string") + fillna: no pandas 3, astype(str) mantém o NaN como NaN
    texto = serie.astype("string").fillna("").str.strip().str.replace(r"\.0$", "", regex=True)
    texto = texto.mask(texto.str.lower().isin(_AUSENTES), "")
    so_digitos = texto.str.fullmatch(r"\d+").fillna(False).astype(bool)
    sem_zeros = texto.str.lstrip("0").replace("", "0")
    return texto.mask(so_digitos, sem_zeros)


def alvo_com_censura(
    cod: pd.Series, positivos: Iterable[str], negativos: Iterable[str]
) -> pd.Series:
    """1.0 nos positivos, 0.0 nos negativos e NaN no resto.

    O resto é censura, ignorado, em branco ou código fora do dicionário: nada
    disso vira 0. ``cod`` já vem normalizado por ``codigo``.
    """
    positivos, negativos = set(positivos), set(negativos)
    if positivos & negativos:
        raise ValueError(f"códigos positivos e negativos ao mesmo tempo: {sorted(positivos & negativos)}")
    alvo = pd.Series(np.nan, index=cod.index, dtype=float)
    alvo[cod.isin(negativos)] = 0.0
    alvo[cod.isin(positivos)] = 1.0
    return alvo


def excluir_sem_rotulo(
    df: pd.DataFrame,
    target_col: str,
    campo: str,
    censura: Iterable[str],
    log: logging.Logger | None = None,
) -> pd.DataFrame:
    """Tira da coorte os casos sem rótulo em ``target_col``.

    Conta à parte a censura (código de ``campo`` em ``censura``) e o que não
    tem código válido (em branco, ignorado, fora do dicionário). As contagens
    ficam em ``df.attrs["exclusoes_rotulo"]`` e no log, para que a exclusão
    seja visível em vez de silenciosa.
    """
    if target_col not in df.columns:
        return df
    log = log or logger
    censura = set(censura)
    sem_rotulo = df[target_col].isna()
    if campo in df.columns:
        cod = codigo(df[campo])
        eh_censura = sem_rotulo & cod.isin(censura)
        presentes = sorted(set(cod[eh_censura]), key=lambda c: (len(c), c))
    else:
        eh_censura = pd.Series(False, index=df.index)
        presentes = []
    n_censura = int(eh_censura.sum())
    n_sem_codigo = int(sem_rotulo.sum()) - n_censura

    out = df.loc[~sem_rotulo].copy()
    out[target_col] = out[target_col].astype(int)
    out.attrs["exclusoes_rotulo"] = {
        "censura": n_censura,
        "sem_codigo": n_sem_codigo,
        "mantidos": len(out),
    }
    if n_censura or n_sem_codigo:
        log.warning(
            "%s: %d casos censurados (%s %s) e %d sem código válido excluídos "
            "da coorte; %d mantidos.",
            target_col, n_censura, campo, ", ".join(presentes) or "-",
            n_sem_codigo, len(out),
        )
    return out


def exigir_rotulo(y: pd.Series, chave: str) -> pd.Series:
    """Devolve o alvo como int, ou falha se ainda houver caso sem rótulo.

    Substitui o ``fillna(0)``: censura não é negativo.
    """
    if y.isna().any():
        raise ValueError(
            f"{chave}: {int(y.isna().sum())} casos sem rótulo (censura ou sem "
            "código); exclua-os com excluir_sem_rotulo antes de treinar."
        )
    return y.astype(int)
