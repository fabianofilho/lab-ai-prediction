"""SINAN-HANS (hanseniase / leprosy notifications) preprocessor."""

from __future__ import annotations

import pandas as pd

from core.data import rotulo


KEEP_COLS = [
    "TP_NOT",
    "DT_NOTIFIC", "DT_DIAG",
    "SG_UF_NOT", "ID_MUNICIP", "ID_MN_RESI",
    # Patient
    "NU_IDADE_N", "CS_SEXO", "CS_RACA", "CS_ESCOL_N", "CS_GESTANT",
    # Clinical
    "FORMACLINI",      # forma clínica: 1=indeterminada, 2=tuberculoide, 3=dimorfa, 4=virchowiana
    "CLASSOPERA",      # classificação operacional: 1=paucibacilar (PB), 2=multibacilar (MB)
    "MODOENTR",        # modo de entrada: 1=caso novo, 2 a 5=transferência, 6=recidiva, 7=outros reingressos, 9=ignorado
    "MODODETECT",      # modo de detecção do caso novo: 1=encaminhamento, 2=demanda espontânea, 3=exame de coletividade, 4=exame de contatos, 5=outros, 9=ignorado
    "BACILOSCOP",      # baciloscopy: 0=neg, 1=pos, 2=not done
    "ESQ_INI_N",       # initial treatment scheme: 1=PB 6 doses, 2=MB 12 doses
    "AVALIA_N",        # grau de incapacidade no diagnóstico: ver AVALIA_N_MAPA
    # Outcome
    "TPALTA_N",        # tipo de saída: ver TPALTA_N_MAPA abaixo
    "DTALTA_N",        # discharge date
    # Treatment tracking
    "DOSE_RECEB",      # doses received
    "DTULTCOMP",       # last completion date
]

# TPALTA_N (campo 19, tipo de saída) no dicionário do SINAN-Hanseníase:
# PySUS 0.15.0, pysus/metadata/SINAN/HANS.csv, conferido em 2026-09-26.
# 9 não é digitável: aparece em caso migrado ou notificado até a versão 1.3,
# quando a saída administrativa era transferência. Não conferido contra um
# HANSBR bruto (a rede do DataSUS estava bloqueada).
TPALTA_N_MAPA = {
    "1": "cura",
    "2": "transferencia_mesmo_municipio",
    "3": "transferencia_outro_municipio",
    "4": "transferencia_outro_estado",
    "5": "transferencia_outro_pais",
    "6": "obito",
    "7": "abandono",
    "8": "erro_diagnostico",
    "9": "transferencia_nao_especificada",
}
TPALTA_CURA = {"1"}
TPALTA_ABANDONO = {"7"}
TPALTA_OBITO = {"6"}
TPALTA_TRANSFERENCIA = {"2", "3", "4", "5", "9"}
TPALTA_ERRO_DIAGNOSTICO = {"8"}
# Censura do abandono: transferência (o caso saiu de vista), erro
# diagnóstico (não era hanseníase) e óbito (risco competitivo, mesma decisão
# do abandono de TB). Esses casos saem da coorte e nunca viram 0.
TPALTA_CENSURA_ABANDONO = TPALTA_TRANSFERENCIA | TPALTA_ERRO_DIAGNOSTICO | TPALTA_OBITO

# AVALIA_N (campo 37, avaliação do grau de incapacidade física no
# diagnóstico), mesmo dicionário: 3 é "não avaliado", não um grau acima de 2.
AVALIA_N_MAPA = {"0": "grau zero", "1": "grau I", "2": "grau II", "3": "não avaliado"}
AVALIA_GRAUS = {"0", "1", "2"}
AVALIA_NAO_AVALIADO = {"3"}

# CLASSOPERA: 1 paucibacilar (PB), 2 multibacilar (MB), conforme o PySUS
CLASSOPERA_MB = "2"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize a raw SINAN-HANS DataFrame."""
    cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[cols].copy()

    # Dates
    for col in ["DT_NOTIFIC", "DT_DIAG", "DTALTA_N", "DTULTCOMP"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Age
    if "NU_IDADE_N" in df.columns:
        df["idade_anos"] = _decode_idade(df["NU_IDADE_N"])

    # Alvo com censura: 1 abandono, 0 cura e NaN para censura ou código fora
    # do dicionário (ver drop_censored)
    if "TPALTA_N" in df.columns:
        alta = rotulo.codigo(df["TPALTA_N"])
        df["abandono"] = rotulo.alvo_com_censura(alta, TPALTA_ABANDONO, TPALTA_CURA)
        df["cura"] = alta.isin(TPALTA_CURA).astype(int)

    # Flag multibacilar: vem da classificação operacional (CLASSOPERA 2 = MB).
    # FORMACLINI 2 é a forma tuberculoide, que é paucibacilar.
    if "CLASSOPERA" in df.columns:
        df["mb"] = (df["CLASSOPERA"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True) == CLASSOPERA_MB).astype(int)

    # Grau de incapacidade no diagnóstico (0, 1 ou 2). Não avaliado (3),
    # em branco e código fora do dicionário viram NaN, nunca um grau.
    if "AVALIA_N" in df.columns:
        avalia = rotulo.codigo(df["AVALIA_N"])
        grau = pd.to_numeric(avalia.where(avalia.isin(AVALIA_GRAUS)), errors="coerce")
        df["grau_incapacidade"] = grau.astype(float)

    # Standardize categoricals
    for col in ["CS_SEXO", "CS_RACA", "CS_ESCOL_N"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

    return df


def drop_censored(df: pd.DataFrame, target_col: str = "abandono") -> pd.DataFrame:
    """Tira da coorte do abandono a censura (TPALTA_N 2 a 6, 8 e 9) e o que
    não tem código válido (sem tipo de saída, em branco, fora do dicionário),
    com as duas contagens em ``df.attrs["exclusoes_rotulo"]`` e no log.

    Não filtre os casos com saída registrada antes: o filtro tiraria o em
    branco e o código fora do dicionário sem contá-los. O filtro antigo de
    encerrados (1 a 6) ainda descartava o abandono (7).
    """
    return rotulo.excluir_sem_rotulo(df, target_col, "TPALTA_N", TPALTA_CENSURA_ABANDONO)


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
