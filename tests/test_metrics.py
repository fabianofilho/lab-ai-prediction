"""Métricas de desempenho clínico (core/models/metrics.py).

Valores conhecidos: modelo perfeitamente calibrado dá CITL perto de 0 e slope
perto de 1; probabilidades dobradas dão O:E perto de 0,5; não tratar ninguém
tem benefício líquido 0; o IC contém o valor pontual; a mesma semente
reproduz o IC. Casos degenerados devolvem NaN com MetricWarning, sem exceção.
Só precisa de numpy, pandas e scikit-learn (o CI instala os três).
"""
import math
import warnings

import numpy as np
import pandas as pd
import pytest

from core.models import metrics as M

sklearn_metrics = pytest.importorskip("sklearn.metrics")


def _calibrado(n=20000, seed=0, media=-1.0, dp=1.0):
    """y sorteado da própria probabilidade: calibração perfeita por construção."""
    rng = np.random.default_rng(seed)
    lp = rng.normal(media, dp, n)
    p = 1 / (1 + np.exp(-lp))
    y = (rng.random(n) < p).astype(int)
    return y, p


# ── Valores conhecidos ───────────────────────────────────────────────────────


def test_modelo_calibrado_tem_citl_perto_de_zero_e_slope_perto_de_um():
    y, p = _calibrado()
    assert abs(M.calibration_in_the_large(y, p)) < 0.05
    assert abs(M.calibration_slope(y, p) - 1) < 0.05
    assert abs(M.observed_expected_ratio(y, p) - 1) < 0.05
    assert M.expected_calibration_error(y, p) < 0.02


def test_probabilidade_dobrada_da_oe_perto_de_meio_e_citl_negativo():
    y, p = _calibrado(media=-3.0, dp=0.5)  # p pequeno: 2p ainda cabe em [0, 1]
    p2 = 2 * p
    assert p2.max() < 1
    assert M.observed_expected_ratio(y, p2) == pytest.approx(0.5, abs=0.03)
    # superestima o risco: CITL negativo, perto de logit do fator 1/2
    assert M.calibration_in_the_large(y, p2) < -0.5


def test_risco_espalhado_demais_da_slope_abaixo_de_um():
    y, p = _calibrado()
    lp = np.log(p / (1 - p))
    p_esp = 1 / (1 + np.exp(-2 * lp))  # logit dobrado: slope verdadeiro 0,5
    assert M.calibration_slope(y, p_esp) == pytest.approx(0.5, abs=0.05)


def test_citl_por_offset_difere_do_intercepto_com_slope_livre():
    y, p = _calibrado()
    lp = np.log(p / (1 - p))
    p_esp = 1 / (1 + np.exp(-(2 * lp + 0.3)))
    b0, _b1 = M.logistic_recalibration(y, p_esp)
    citl = M.calibration_in_the_large(y, p_esp)
    assert not math.isclose(b0, citl, abs_tol=0.05)


def test_recalibracao_bate_com_regressao_logistica_do_sklearn():
    from sklearn.linear_model import LogisticRegression

    y, p = _calibrado(n=3000, seed=3)
    lp = np.log(p / (1 - p))
    lr = LogisticRegression(C=np.inf, tol=1e-10, max_iter=1000).fit(lp.reshape(-1, 1), y)
    b0, b1 = M.logistic_recalibration(y, p)
    assert b0 == pytest.approx(lr.intercept_[0], abs=1e-5)
    assert b1 == pytest.approx(lr.coef_[0, 0], abs=1e-5)


def test_auroc_ap_e_brier_batem_com_o_sklearn_inclusive_com_empate():
    y, p = _calibrado(n=3000, seed=1)
    p = np.round(p, 2)  # força empates
    assert M.roc_auc(y, p) == pytest.approx(sklearn_metrics.roc_auc_score(y, p), abs=1e-12)
    assert M.average_precision(y, p) == pytest.approx(
        sklearn_metrics.average_precision_score(y, p), abs=1e-12)
    assert M.brier_score(y, p) == pytest.approx(
        sklearn_metrics.brier_score_loss(y, p), abs=1e-12)


def test_ece_com_bins_conhecidos():
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.1, 0.9, 1.0])
    # bin 1: p média 0,1, observado 0,5 -> 0,4; bin 9: p média 0,95, observado 0,5 -> 0,45
    assert M.expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.425)
    # um bin só: |0,5 - 0,525|
    assert M.expected_calibration_error(y, p, n_bins=1) == pytest.approx(0.025)


# ── Decision curve ───────────────────────────────────────────────────────────


def test_tratar_ninguem_tem_beneficio_liquido_zero_e_tratar_todos_tem_a_formula():
    y, p = _calibrado(n=5000)
    dc = M.decision_curve(y, p)
    assert (dc["net_benefit_none"] == 0).all()
    prev = y.mean()
    t = dc["threshold"].to_numpy()
    np.testing.assert_allclose(dc["net_benefit_all"], prev - (1 - prev) * t / (1 - t))
    np.testing.assert_allclose(dc["best_trivial"], np.maximum(dc["net_benefit_all"], 0))
    np.testing.assert_allclose(dc["gain"], dc["net_benefit_model"] - dc["best_trivial"])


def test_beneficio_liquido_do_modelo_na_mao():
    y = np.array([1, 1, 0, 0, 0])
    p = np.array([0.9, 0.4, 0.6, 0.2, 0.1])
    dc = M.decision_curve(y, p, thresholds=[0.5])
    # em 0,5 trata 0,9 (VP) e 0,6 (FP): 1/5 - 1/5 * 1 = 0
    assert dc["net_benefit_model"].iloc[0] == pytest.approx(0.0)
    # p >= t: quem tem exatamente o limiar é tratado
    dc2 = M.decision_curve(y, p, thresholds=[0.4])
    assert dc2["net_benefit_model"].iloc[0] == pytest.approx(2 / 5 - 1 / 5 * (0.4 / 0.6))


def test_modelo_perfeito_supera_as_estrategias_triviais_em_toda_a_grade():
    y = np.array([0] * 80 + [1] * 20)
    p = np.where(y == 1, 0.995, 0.005)  # fora da grade: ninguém empata no limiar
    faixas = M.net_benefit_ranges(M.decision_curve(y, p), margin=0.01)
    assert faixas["any_gain"] == [(0.01, 0.99)]
    assert faixas["any_gain_contiguous"]


def test_faixa_relevante_exige_margem_e_fica_dentro_da_ingenua():
    y, p = _calibrado(n=20000, seed=2)
    dc = M.decision_curve(y, p)
    faixas = M.net_benefit_ranges(dc, margin=0.01)
    ing = {t for lo, hi in faixas["any_gain"] for t in dc["threshold"] if lo <= t <= hi}
    rel = {t for lo, hi in faixas["relevant_gain"] for t in dc["threshold"] if lo <= t <= hi}
    assert rel and rel <= ing and rel != ing
    ganho = dc.set_index("threshold")["gain"]
    assert all(ganho[t] >= 0.01 - 1e-12 for t in rel)


def test_faixa_com_buraco_nao_e_contigua_e_o_texto_mostra_os_trechos():
    curva = pd.DataFrame({
        "threshold": [0.1, 0.2, 0.3, 0.4, 0.5],
        "gain": [0.02, 0.0, 0.03, 0.015, -0.01],
    })
    faixas = M.net_benefit_ranges(curva, margin=0.01)
    assert faixas["any_gain"] == [(0.1, 0.1), (0.3, 0.4)]
    assert not faixas["any_gain_contiguous"]
    assert M.format_ranges(faixas["any_gain"]) == "10%; 30% a 40%"
    assert M.format_ranges([]) == "nenhuma"
    assert faixas["threshold_max_gain"] == pytest.approx(0.3)


def test_limiar_fora_de_zero_e_um_da_erro_claro():
    y, p = _calibrado(n=100)
    with pytest.raises(ValueError, match="limiares"):
        M.decision_curve(y, p, thresholds=[0.0, 0.5])


# ── Bootstrap ────────────────────────────────────────────────────────────────


def test_ic_contem_o_valor_pontual():
    y, p = _calibrado(n=3000, seed=4)
    res = M.bootstrap_ci(y, p, n_boot=300, seed=7)
    for nome in ("roc_auc", "pr_auc", "brier", "citl", "calibration_slope", "oe_ratio"):
        r = res[nome]
        assert r["ci_low"] <= r["value"] <= r["ci_high"], nome
        assert r["ci_low"] < r["ci_high"], nome
        assert r["n_valid"] == 300, nome


def test_semente_fixa_reproduz_o_ic_e_semente_diferente_muda():
    y, p = _calibrado(n=1000, seed=5)
    a = M.bootstrap_ci(y, p, n_boot=100, seed=11)
    b = M.bootstrap_ci(y, p, n_boot=100, seed=11)
    c = M.bootstrap_ci(y, p, n_boot=100, seed=12)
    assert a == b
    assert a["roc_auc"]["ci_low"] != c["roc_auc"]["ci_low"]


def test_bootstrap_estratificado_mantem_as_duas_classes_com_poucos_eventos():
    rng = np.random.default_rng(0)
    y = np.zeros(400, dtype=int)
    y[:3] = 1  # três eventos: sem estratificar, muitas réplicas ficariam sem positivo
    p = rng.random(400)
    res = M.bootstrap_ci(y, p, metrics=["roc_auc"], n_boot=200, seed=1)
    assert res["roc_auc"]["n_valid"] == 200


def test_ic_bate_com_o_bootstrap_ingenuo_via_sklearn():
    """O atalho por pesos reproduz o recálculo direto em cada réplica."""
    y, p = _calibrado(n=500, seed=6)
    res = M.bootstrap_ci(y, p, metrics=["roc_auc", "pr_auc"], n_boot=50, seed=3)
    rng = np.random.default_rng(3)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    auc, ap = [], []
    for _ in range(50):
        ip = pos[rng.integers(0, pos.size, pos.size)]
        ineg = neg[rng.integers(0, neg.size, neg.size)]
        idx = np.concatenate([ip, ineg])
        auc.append(sklearn_metrics.roc_auc_score(y[idx], p[idx]))
        ap.append(sklearn_metrics.average_precision_score(y[idx], p[idx]))
    assert res["roc_auc"]["ci_low"] == pytest.approx(np.quantile(auc, 0.025), abs=1e-12)
    assert res["pr_auc"]["ci_high"] == pytest.approx(np.quantile(ap, 0.975), abs=1e-12)


def test_resumo_e_tabela_trazem_todas_as_metricas():
    y, p = _calibrado(n=800, seed=8)
    s = M.performance_summary(y, p, n_boot=50, seed=1)
    assert set(s["metrics"]) == set(M.METRICS)
    assert s["warnings"] == []
    tab = M.summary_table(s)
    assert list(tab.columns) == ["Métrica", "Valor", "IC 95% inf.", "IC 95% sup.", "Ideal"]
    assert len(tab) == len(M.METRICS)


# ── EPV ──────────────────────────────────────────────────────────────────────


def test_epv_avisa_quando_ha_poucos_eventos_por_variavel():
    y = np.array([1] * 30 + [0] * 970)
    with pytest.warns(M.MetricWarning, match="EPV"):
        r = M.events_per_variable(y, n_predictors=6)
    assert r["n_events"] == 30
    assert r["epv"] == pytest.approx(5.0)
    assert not r["ok"]
    assert r["n_min"] == 2000  # 10 * 6 / 0,03


def test_epv_usa_a_classe_menos_frequente_e_nao_avisa_quando_basta():
    y = np.array([1] * 900 + [0] * 100)
    with warnings.catch_warnings():
        warnings.simplefilter("error", M.MetricWarning)
        r = M.events_per_variable(y, n_predictors=5)
    assert r["n_events"] == 100
    assert r["ok"]


# ── Casos degenerados: NaN com aviso, sem exceção opaca ─────────────────────


@pytest.mark.parametrize("fn", [
    M.roc_auc, M.average_precision, M.calibration_in_the_large,
    M.calibration_slope,
])
def test_uma_classe_so_devolve_nan_com_aviso(fn):
    y = np.zeros(50, dtype=int)
    p = np.linspace(0.1, 0.9, 50)
    with pytest.warns(M.MetricWarning, match="uma classe"):
        assert math.isnan(fn(y, p))


def test_uma_classe_so_no_resumo_e_na_decision_curve():
    y = np.ones(40, dtype=int)
    p = np.linspace(0.1, 0.9, 40)
    s = M.performance_summary(y, p, n_boot=20)
    assert math.isnan(s["metrics"]["roc_auc"]["value"])
    assert math.isnan(s["metrics"]["roc_auc"]["ci_low"])
    assert any("uma classe" in w for w in s["warnings"])
    # Brier e O:E seguem definidos com uma classe
    assert math.isfinite(s["metrics"]["brier"]["value"])
    with pytest.warns(M.MetricWarning):
        dc = M.decision_curve(y, p)
    assert (dc["net_benefit_none"] == 0).all()
    assert dc["net_benefit_model"].isna().all()


def test_probabilidade_zero_ou_um_exata_da_nan_no_logit_e_clip_resolve():
    y, p = _calibrado(n=2000, seed=9)
    p = p.copy()
    p[0], p[1] = 0.0, 1.0
    with pytest.warns(M.MetricWarning, match="exatamente 0 ou 1"):
        assert math.isnan(M.calibration_in_the_large(y, p))
    with pytest.warns(M.MetricWarning, match="exatamente 0 ou 1"):
        assert math.isnan(M.calibration_slope(y, p))
    assert math.isfinite(M.calibration_in_the_large(y, p, clip=1e-6))
    # as métricas que não usam logit seguem definidas
    assert math.isfinite(M.observed_expected_ratio(y, p))
    assert math.isfinite(M.expected_calibration_error(y, p))
    s = M.performance_summary(y, p, n_boot=20)
    assert math.isnan(s["metrics"]["citl"]["ci_low"])
    assert math.isfinite(s["metrics"]["roc_auc"]["ci_low"])


def test_separacao_completa_e_predicao_constante_dao_nan_no_slope():
    y = np.array([0] * 50 + [1] * 50)
    p = np.where(y == 1, 0.8, 0.2)
    with pytest.warns(M.MetricWarning, match="separação"):
        assert math.isnan(M.calibration_slope(y, p))
    with pytest.warns(M.MetricWarning, match="constante"):
        assert math.isnan(M.calibration_slope(y, np.full(100, 0.3)))


def test_amostra_vazia_e_soma_de_p_nula():
    with pytest.warns(M.MetricWarning, match="vazia"):
        assert math.isnan(M.brier_score([], []))
    with pytest.warns(M.MetricWarning, match="esperado nulo"):
        assert math.isnan(M.observed_expected_ratio([0, 1], [0.0, 0.0]))


def test_entrada_invalida_da_erro_claro():
    with pytest.raises(ValueError, match="tamanhos"):
        M.roc_auc([0, 1], [0.5])
    with pytest.raises(ValueError, match="binário"):
        M.roc_auc([0, 2], [0.1, 0.9])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        M.roc_auc([0, 1], [0.1, 1.2])
    with pytest.raises(ValueError, match="ausente"):
        M.roc_auc([0, 1], [0.1, np.nan])
