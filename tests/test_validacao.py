"""Validação: HPO só no treino, Optuna semeado e balanceamento padrão.

Os testes de pipeline precisam de scikit-learn e optuna e são pulados se o
ambiente não tiver o stack de ML. Os testes sobre pages/analise.py leem o
código-fonte (AST), sem importar streamlit.
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ANALISE = Path(__file__).resolve().parent.parent / "pages" / "analise.py"


def _dados(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "c": rng.integers(0, 3, size=n).astype(float),
    })
    logit = 1.5 * X["a"] - X["b"]
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int), name="y")
    return X, y


# ── split_train_test ─────────────────────────────────────────────────────────

def test_split_holdout_disjunto_e_estratificado():
    pytest.importorskip("sklearn")
    from core.models.pipeline import split_train_test

    X, y = _dados()
    X_tr, X_te, y_tr, _ = split_train_test(X, y, "holdout", holdout_size=0.25)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert len(X_tr) + len(X_te) == len(X)
    assert len(X_te) == 50
    assert list(y_tr.index) == list(X_tr.index)


def test_split_temporal_treino_so_antes_do_corte():
    pytest.importorskip("sklearn")
    from core.models.pipeline import split_train_test

    X, y = _dados(n=100)
    dates = pd.Series(pd.date_range("2022-01-01", periods=100, freq="D"))
    cutoff = "2022-03-01"
    X_tr, X_te, _, _ = split_train_test(X, y, "temporal", dates=dates, cutoff=cutoff)
    assert (dates[X_tr.index] < pd.Timestamp(cutoff)).all()
    assert (dates[X_te.index] >= pd.Timestamp(cutoff)).all()
    assert len(X_tr) == 59 and len(X_te) == 41


def test_split_temporal_insuficiente_levanta_erro():
    pytest.importorskip("sklearn")
    from core.models.pipeline import split_train_test

    X, y = _dados(n=30)
    dates = pd.Series(pd.date_range("2022-01-01", periods=30, freq="D"))
    with pytest.raises(ValueError, match="insuficiente"):
        split_train_test(X, y, "temporal", dates=dates, cutoff="2022-01-05")


# ── Optuna semeado ────────────────────────────────────────────────────────────

def test_optuna_mesma_semente_mesmos_hiperparametros():
    pytest.importorskip("sklearn")
    pytest.importorskip("optuna")
    from core.models.pipeline import optimize_hyperparams

    X, y = _dados()
    kw = {"algorithm": "logreg", "n_trials": 6, "n_folds": 3}
    p1 = optimize_hyperparams(X, y, seed=123, **kw)
    p2 = optimize_hyperparams(X, y, seed=123, **kw)
    p3 = optimize_hyperparams(X, y, seed=7, **kw)
    assert p1 == p2
    assert p1 != p3


# ── pages/analise.py (sem importar streamlit) ────────────────────────────────

def _calls(tree, nome):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if (isinstance(f, ast.Name) and f.id == nome) or (
                isinstance(f, ast.Attribute) and f.attr == nome
            ):
                yield node


@pytest.fixture(scope="module")
def analise_ast():
    return ast.parse(ANALISE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("fn", ["optimize_hyperparams", "random_search", "grid_search"])
def test_hpo_recebe_so_particao_de_treino(analise_ast, fn):
    """A busca de hiperparâmetros recebe X_hpo (treino em holdout/temporal)."""
    chamadas = list(_calls(analise_ast, fn))
    assert chamadas, f"{fn} não é chamada em pages/analise.py"
    for c in chamadas:
        primeiro = c.args[0]
        assert isinstance(primeiro, ast.Name) and primeiro.id == "X_hpo", (
            f"{fn} na linha {c.lineno} recebe {ast.unparse(primeiro)} em vez de X_hpo"
        )


def test_optuna_recebe_semente_da_ui(analise_ast):
    for c in _calls(analise_ast, "optimize_hyperparams"):
        kws = {k.arg for k in c.keywords}
        assert "seed" in kws, f"optimize_hyperparams na linha {c.lineno} sem seed"


def test_balanceamento_padrao_e_nenhum(analise_ast):
    radios = [
        c for c in _calls(analise_ast, "radio")
        if c.args and isinstance(c.args[0], ast.Constant) and c.args[0].value == "Balanceamento"
    ]
    assert len(radios) == 1
    radio = radios[0]
    opcoes = ast.literal_eval(radio.args[1])
    index = next(k.value for k in radio.keywords if k.arg == "index")
    assert opcoes[ast.literal_eval(index)] == "Nenhum"
