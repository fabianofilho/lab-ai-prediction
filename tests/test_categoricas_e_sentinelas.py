"""Os três defeitos que o benchmark de pré-processamento expôs (LAB-162).

1. ``build_features`` convertia categoria em código numérico e apagava o
   ausente. Agora a categoria chega como texto e o ausente como NaN.
2. ``SentinelReplacer`` só comparava com número e não alcançava o código de
   ignorado gravado como texto, que é como ele chega no SINAN.
3. O SINASC descartava a história obstétrica, e ``UF_SIGLA`` entrava como
   atributo constante numa coorte de um estado só.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from core.data import sinasc as sinasc_prep
from core.features import engineering as eng
from core.features.cohort import CohortBuilder
from core.models.pipeline import OHE_MAX_CATEGORIAS, SentinelReplacer, build_pipeline
from core.outcomes import OUTCOMES

# ── 1. categoria como categoria ───────────────────────────────────────────────

def test_as_category_normaliza_codigo_e_preserva_ausente():
    serie = pd.Series([" 2", "2.0", 2, "02", None, np.nan, "", "nan", "10", "M"])
    got = eng.as_category(serie)
    assert got.dtype == object
    assert got.iloc[:4].tolist() == ["2", "2", "2", "2"]
    assert got.iloc[4:8].isna().all(), "ausente virou categoria"
    assert got.iloc[8:].tolist() == ["10", "M"]


def test_as_category_nao_inventa_ordem():
    got = eng.as_category(pd.Series(["3", "1", "2"]))
    assert not pd.api.types.is_numeric_dtype(got)


def test_cid_vira_categoria_nominal():
    serie = pd.Series(["I21", "j18", None])
    assert eng.icd10_chapter(serie).iloc[:2].tolist() == ["I", "J"]
    assert eng.icd10_block(serie).iloc[:2].tolist() == ["I21", "J18"]
    assert pd.isna(eng.icd10_chapter(serie).iloc[2])


def _tb_raw():
    n = 6
    return pd.DataFrame({
        "SITUA_ENCE": ["1", "2", "1", "2", "1", "10"],
        "DT_ENCERRA": pd.to_datetime(["2023-06-30"] * n),
        "NU_IDADE_N": [4035] * n,
        "CS_SEXO": ["M", "F", "M", "F", None, "M"],
        "FORMA": ["1", "2", " 1", "1.0", "9", None],
        "HIV": ["2"] * n,
        "TRAT_SUPER": ["1"] * n,
    })


def test_tb_build_features_entrega_categoria_com_ausente():
    oc = OUTCOMES["abandono_tb"]
    df = oc.build_features(oc.build_cohort({"SINAN_TB": _tb_raw()}))
    for col in ("CS_SEXO", "FORMA"):
        assert not pd.api.types.is_numeric_dtype(df[col]), f"{col} virou número"
    assert df["CS_SEXO"].isna().sum() == 1, "ausente de CS_SEXO sumiu"
    # " 1" e "1.0" são o mesmo código
    assert set(df["FORMA"].dropna()) == {"1", "2", "9"}


# ── 2. ignorado gravado como texto ────────────────────────────────────────────

def test_sentinel_alcanca_codigo_em_texto():
    X = pd.DataFrame({
        "cat": ["9", " 99", "9.0", "1", "19", "0801010012"],
        "num": [9.0, 1.0, 2.0, 99.0, 3.0, 4.0],
    })
    out = SentinelReplacer([9, 99]).fit_transform(X)
    assert out["cat"].isna().tolist() == [True, True, True, False, False, False]
    assert out["num"].isna().tolist() == [True, False, False, True, False, False]


def test_sentinel_nao_troca_parte_da_celula():
    X = pd.DataFrame({"proc": ["0801010012", "0409", "99a"]})
    out = SentinelReplacer([9, 99]).fit_transform(X)
    assert out["proc"].notna().all()


# ── one-hot com teto ──────────────────────────────────────────────────────────

def test_ohe_tem_teto_de_colunas():
    rng = np.random.default_rng(0)
    n = 2000
    X = pd.DataFrame({
        "proc": [f"P{i}" for i in rng.integers(0, 400, n)],
        "idade": rng.normal(50, 10, n),
    })
    y = pd.Series(rng.integers(0, 2, n))
    pipe = build_pipeline(X, algorithm="logreg").fit(X, y)
    largura = pipe.named_steps["prep"].transform(X).shape[1]
    assert largura <= OHE_MAX_CATEGORIAS + 1


def test_pipeline_treina_com_categoria_ausente_e_ignorado():
    rng = np.random.default_rng(1)
    n = 300
    X = pd.DataFrame({
        "sexo": rng.choice(["M", "F", None], n),
        "forma": rng.choice(["1", "2", "9"], n),
        "idade": rng.normal(40, 12, n),
    })
    y = pd.Series(rng.integers(0, 2, n))
    pipe = build_pipeline(X, algorithm="rf", treatment={"null_sentinels": [9]})
    proba = pipe.fit(X, y).predict_proba(X)[:, 1]
    assert np.isfinite(proba).all()


# ── 3. história obstétrica e atributo constante ───────────────────────────────

def _sinasc_raw(n=40, uf="32"):
    rng = np.random.default_rng(2)
    return pd.DataFrame({
        "DTNASC": ["01062023"] * n,
        "SEXO": rng.choice(["1", "2"], n),
        "PESO": rng.integers(1500, 4000, n),
        "GESTACAO": rng.choice(["4", "5"], n),
        "GRAVIDEZ": ["1"] * n,
        "PARTO": rng.choice(["1", "2"], n),
        "CONSULTAS": rng.choice(["2", "3", "4"], n),
        "IDADEMAE": rng.integers(16, 42, n),
        "QTDGESTANT": rng.choice(["0", "1", "2", "99"], n),
        "QTDPARTNOR": rng.choice(["0", "1", "99"], n),
        "QTDPARTCES": rng.choice(["0", "1"], n),
        "QTDFILVIVO": rng.choice(["0", "1", "2", "99"], n),
        "QTDFILMORT": rng.choice(["0", "99"], n),
        "CODMUNRES": [f"{uf}0530"] * n,
    })


def test_sinasc_guarda_historia_obstetrica_e_ignorado_vira_ausente():
    df = sinasc_prep.preprocess(_sinasc_raw())
    for col in sinasc_prep.QTD_COLS:
        assert col in df.columns, f"{col} descartada no preprocess"
        assert pd.api.types.is_numeric_dtype(df[col])
        assert not (df[col] == sinasc_prep.QTD_IGNORADO).any(), f"99 sobrou em {col}"
    assert df["QTDGESTANT"].isna().any()


def test_prematuridade_usa_historia_obstetrica_e_tira_uf_constante():
    oc = OUTCOMES["prematuridade"]
    cohort = oc.build_features(oc.build_cohort({"SINASC": _sinasc_raw()}))
    X, _ = CohortBuilder(oc).get_Xy(cohort)
    for col in ("QTDGESTANT", "QTDPARTNOR", "QTDPARTCES"):
        assert col in X.columns
    if "UF_SIGLA" in cohort.columns:
        assert "UF_SIGLA" not in X.columns, "UF constante entrou como atributo"


def test_uf_que_varia_continua():
    oc = OUTCOMES["prematuridade"]
    raw = pd.concat([_sinasc_raw(uf="32"), _sinasc_raw(uf="35")], ignore_index=True)
    cohort = oc.build_features(oc.build_cohort({"SINASC": raw}))
    if "UF_SIGLA" not in cohort.columns:
        return
    X, _ = CohortBuilder(oc).get_Xy(cohort)
    assert "UF_SIGLA" in X.columns
