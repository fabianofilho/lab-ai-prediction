"""Escala das numéricas sem escalonador escondido.

Até aqui o build_pipeline acrescentava um StandardScaler depois do
pré-processamento para logística e MLP. Ele escalonava também as colunas
one-hot, escalonava de novo quem tinha escolhido robust ou binning, e trocava
"nenhuma" por Z-score sem aparecer na tela. Agora a escala é só a do
tratamento: escolhida na tela, ou Z-score nas numéricas quando não há
tratamento e o algoritmo depende de escala.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler

from core.models.pipeline import SCALE_SENSITIVE, build_pipeline


def _dados(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "idade": rng.normal(60, 15, n),
        "custo": rng.lognormal(8, 1, n),
        "sexo": rng.choice(["M", "F"], n),
    })
    y = pd.Series(rng.integers(0, 2, n))
    return X, y


def _numericas(pipe, X):
    """Colunas numéricas como o modelo as recebe, na ordem de X.

    Passa por todos os passos antes do modelo, não só pelo pré-processador:
    o escalonador antigo vinha depois dele.
    """
    saida = pipe[:-1].transform(X)
    nomes = list(pipe.named_steps["prep"].get_feature_names_out())
    idx = [i for i, n in enumerate(nomes) if n.split("__")[-1] in ("idade", "custo")]
    return np.asarray(saida)[:, idx], np.asarray(saida), nomes


@pytest.mark.parametrize("algoritmo", sorted(SCALE_SENSITIVE))
def test_sem_escalonador_escondido(algoritmo):
    X, y = _dados()
    pipe = build_pipeline(X, algoritmo, treatment={"num_default": "none"}).fit(X, y)
    assert "scaler" not in pipe.named_steps
    num, _, _ = _numericas(pipe, X)
    # "nenhuma" escolhida na tela continua sendo nenhuma
    np.testing.assert_allclose(num, X[["idade", "custo"]].to_numpy())


@pytest.mark.parametrize("algoritmo", sorted(SCALE_SENSITIVE))
def test_sem_tratamento_aplica_zscore_so_nas_numericas(algoritmo):
    X, y = _dados()
    pipe = build_pipeline(X, algoritmo).fit(X, y)
    num, saida, nomes = _numericas(pipe, X)
    np.testing.assert_allclose(num.mean(axis=0), 0, atol=1e-8)
    np.testing.assert_allclose(num.std(axis=0), 1, atol=1e-8)
    one_hot = [i for i, n in enumerate(nomes) if "sexo" in n]
    assert set(np.unique(saida[:, one_hot])) <= {0.0, 1.0}, "one-hot foi escalonado"


def test_robust_nao_e_escalonado_duas_vezes():
    X, y = _dados()
    pipe = build_pipeline(X, "logreg", treatment={"num_default": "robust"}).fit(X, y)
    num, _, _ = _numericas(pipe, X)
    esperado = RobustScaler().fit_transform(X[["idade", "custo"]])
    np.testing.assert_allclose(num, esperado)


def test_arvore_sem_tratamento_continua_sem_escala():
    X, y = _dados()
    pipe = build_pipeline(X, "rf").fit(X, y)
    num, _, _ = _numericas(pipe, X)
    np.testing.assert_allclose(num, X[["idade", "custo"]].to_numpy())
