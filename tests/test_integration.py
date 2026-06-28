"""Testes de integração que baixam dados reais do DATASUS.

Pulados por padrão (rede lenta/instável). Rode localmente com:
    RUN_NETWORK_TESTS=1 pytest tests/test_integration.py -v
"""
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS") != "1",
    reason="teste de rede; defina RUN_NETWORK_TESTS=1 para rodar",
)


@pytest.mark.network
def test_sinasc_outcome_end_to_end():
    """Baixa SINASC, constrói coorte e treina: alvo binário e AUC plausível."""
    from core.data.downloader import fetch
    from core.outcomes import OUTCOMES
    from core.features.cohort import CohortBuilder
    from core.models.pipeline import train_cv

    oc = OUTCOMES["baixo_peso_nascer"]
    raw = fetch("SINASC", "AC", 2022, max_rows=8000)
    cohort = oc.build_features(oc.build_cohort({"SINASC": raw}))
    X, y = CohortBuilder(oc).get_Xy(cohort)

    assert y.nunique() == 2, "alvo não é binário"
    assert 0.0 < y.mean() < 1.0, f"prevalência degenerada: {y.mean()}"
    assert "PESO" not in X.columns, "PESO (fonte do alvo) vazou nas features"

    res = train_cv(X, y, algorithm="lgbm", n_folds=3)
    auc = res["mean_metrics"].get("auc", res["mean_metrics"].get("roc_auc", 0))
    assert 0.5 <= auc <= 1.0, f"AUC fora do intervalo: {auc}"


@pytest.mark.network
def test_national_sinan_uf_filter():
    """SINAN nacional no cap realista retorna linhas do UF pedido (não vazio)."""
    from core.data.downloader import fetch

    df = fetch("SINAN_VIOL", "SP", 2023, max_rows=1000)
    assert len(df) > 0, "filtro de UF retornou vazio (regressão do national-filter)"
    if "SG_UF_NOT" in df.columns:
        ufs = (
            df["SG_UF_NOT"].astype(str).str.replace(r"\.0$", "", regex=True).unique()
        )
        assert list(ufs) == ["35"], f"UF inesperada após filtro: {ufs}"


@pytest.mark.network
def test_benchmark_loaders_run():
    """Os benchmarks 'ok' baixam e devolvem alvo binário."""
    from core.data.benchmarks import BENCHMARKS

    for key, b in BENCHMARKS.items():
        if b.status != "ok":
            continue
        df = b.loader()
        assert b.target_col in df.columns, f"{key}: sem coluna alvo"
        assert df[b.target_col].nunique() == 2, f"{key}: alvo não binário"
