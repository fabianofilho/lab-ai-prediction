"""Métricas de desempenho clínico sobre probabilidades preditas.

Funções puras sobre ``y_true`` (0 ou 1) e ``y_prob`` (probabilidade do
desfecho), em numpy e pandas, sem Streamlit nem gráfico. A tela de resultados
e o relatório chamam daqui; os gráficos ficam em ``core/models/evaluation.py``.

O que o módulo entrega, e por quê:

- calibração: calibration-in-the-large (CITL), calibration slope, razão O:E e
  ECE. AUROC diz se o modelo ordena bem; a calibração diz se o número está
  certo, e é o número que vai para a conduta (CP9 da ml-checkpoints);
- incerteza: IC por bootstrap estratificado e semeado para AUROC, AP, Brier e
  as métricas de calibração ("métrica sem incerteza não é resultado", regras
  gerais da ml-checkpoints);
- utilidade: decision curve, com o benefício líquido do modelo, de tratar todos
  e de não tratar ninguém, e as faixas de limiar em que o modelo supera a
  melhor estratégia trivial, com e sem margem mínima (CP8 da ml-checkpoints e
  item 13 dos aprendizados do ai-lab-hub);
- tamanho de amostra: eventos por variável (EPV) e o N mínimo para o EPV
  pedido, como aviso (CP6).

Casos degenerados (amostra vazia, uma classe só, probabilidade exatamente 0 ou
1, predição constante, separação completa) não lançam exceção: a métrica volta
NaN e um ``MetricWarning`` explica o motivo. Entrada inválida (tamanhos
diferentes, rótulo fora de {0, 1}, probabilidade ausente ou fora de [0, 1])
lança ``ValueError`` dizendo o que está errado.

Todas as métricas são calculadas sobre a amostra agrupada por valor distinto
de probabilidade, com o peso de cada grupo. Assim o bootstrap não reordena a
amostra a cada réplica: só recontam os pesos, e mil réplicas numa amostra de
cem mil linhas levam segundos.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd


class MetricWarning(UserWarning):
    """Aviso de métrica indefinida ou de amostra pequena."""


NAN = float("nan")

METRIC_LABELS: dict[str, str] = {
    "roc_auc": "AUROC",
    "pr_auc": "AP (PR-AUC)",
    "brier": "Brier",
    "citl": "Calibration-in-the-large",
    "calibration_slope": "Calibration slope",
    "oe_ratio": "Razão O:E",
    "ece": "ECE",
}

METRIC_REFERENCE: dict[str, str] = {
    "roc_auc": "1; 0,5 = acaso",
    "pr_auc": "1; prevalência = acaso",
    "brier": "0; prev x (1 - prev) = sem informação",
    "citl": "0; > 0 subestima, < 0 superestima",
    "calibration_slope": "1; < 1 risco espalhado demais",
    "oe_ratio": "1; < 1 superestima, > 1 subestima",
    "ece": "0",
}

METRICS: tuple[str, ...] = tuple(METRIC_LABELS)


def _warn(msg: str) -> None:
    warnings.warn(msg, MetricWarning, stacklevel=3)


def _prepare(y_true, y_prob) -> tuple[np.ndarray, np.ndarray]:
    """Converte e valida a entrada: y em {0, 1}, p finito em [0, 1]."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(
            f"y_true e y_prob têm tamanhos diferentes: {y.size} e {p.size}."
        )
    if np.isnan(y).any():
        raise ValueError("y_true tem valor ausente; o rótulo precisa ser 0 ou 1.")
    if not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y_true precisa ser binário (0 ou 1).")
    if not np.isfinite(p).all():
        raise ValueError("y_prob tem valor ausente ou infinito.")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("y_prob precisa estar em [0, 1].")
    return y.astype(int), p


# ── Amostra agrupada por valor de p ──────────────────────────────────────────


class _Grouped:
    """Valores distintos de p (crescentes) e o peso de positivos e negativos."""

    def __init__(self, y: np.ndarray, p: np.ndarray):
        self.u, self.inv = np.unique(p, return_inverse=True)
        self.inv = self.inv.ravel()
        self.pos = np.bincount(self.inv[y == 1], minlength=self.u.size).astype(float)
        self.neg = np.bincount(self.inv[y == 0], minlength=self.u.size).astype(float)

    def logit(self, clip: float | None) -> np.ndarray | None:
        """logit dos valores distintos, ou None se há 0 ou 1 exato sem clip."""
        u = self.u
        if clip is not None:
            u = np.clip(u, clip, 1 - clip)
        elif u.size and (u[0] == 0 or u[-1] == 1):
            return None
        return np.log(u) - np.log1p(-u)


def _check_clip(clip: float | None) -> None:
    if clip is not None and not 0 < clip < 0.5:
        raise ValueError("clip precisa estar em (0, 0.5).")


def _k_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC por pesos: P(p_pos > p_neg) + 0,5 P(empate)."""
    wp, wn = pos.sum(), neg.sum()
    if wp == 0 or wn == 0:
        return NAN
    below = np.cumsum(neg) - neg
    return float(np.sum(pos * (below + 0.5 * neg)) / (wp * wn))


def _k_ap(pos: np.ndarray, neg: np.ndarray) -> float:
    """Average precision como no sklearn: soma de (R_n - R_n-1) * P_n."""
    wp = pos.sum()
    if wp == 0:
        return NAN
    pos_d, neg_d = pos[::-1], neg[::-1]
    tp = np.cumsum(pos_d)
    tot = tp + np.cumsum(neg_d)
    prec = np.divide(tp, tot, out=np.zeros_like(tp), where=tot > 0)
    return float(np.sum(pos_d * prec) / wp)


def _k_brier(u: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    w = pos.sum() + neg.sum()
    if w == 0:
        return NAN
    return float((np.sum(pos * (1 - u) ** 2) + np.sum(neg * u**2)) / w)


def _k_oe(u: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    expected = float(np.sum((pos + neg) * u))
    if expected == 0:
        return NAN
    return float(pos.sum()) / expected


def _k_ece(u: np.ndarray, pos: np.ndarray, neg: np.ndarray, n_bins: int) -> float:
    w = pos + neg
    total = w.sum()
    if total == 0:
        return NAN
    b = np.minimum((u * n_bins).astype(int), n_bins - 1)
    obs = np.bincount(b, weights=pos, minlength=n_bins)
    exp = np.bincount(b, weights=w * u, minlength=n_bins)
    return float(np.abs(obs - exp).sum() / total)  # soma de n_bin * |ȳ - p̄|


def _expit(z: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def _newton(
    lp: np.ndarray,
    pos: np.ndarray,
    neg: np.ndarray,
    fit_slope: bool,
    start: np.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-9,
    max_coef: float = 1e4,
) -> np.ndarray | None:
    """Máxima verossimilhança binomial da recalibração no logit.

    ``fit_slope=False``: ``logit P = a + lp`` (lp como offset, só o intercepto
    livre, o CITL). ``fit_slope=True``: ``logit P = b0 + b1 * lp``.

    Newton-Raphson a partir de ``start`` (zeros por padrão; o bootstrap parte
    da estimativa pontual). Passo grande (acima de 1 em algum coeficiente) é
    reduzido à metade até a verossimilhança subir. Linhas de peso zero não
    entram na conta. Devolve None quando não converge, quando a hessiana é
    singular ou quando algum coeficiente passa de ``max_coef`` (separação
    completa).
    """
    used = (pos + neg) > 0
    if not used.all():
        lp, pos, neg = lp[used], pos[used], neg[used]
    w = pos + neg
    k = 2 if fit_slope else 1
    beta = np.zeros(k) if start is None else np.asarray(start, dtype=float).copy()

    def linpred(b: np.ndarray) -> np.ndarray:
        return b[0] + b[1] * lp if fit_slope else b[0] + lp

    def loglik(b: np.ndarray) -> float:
        z = linpred(b)
        return float(np.dot(pos, z) - np.dot(w, np.logaddexp(0.0, z)))

    for _ in range(max_iter):
        q = _expit(linpred(beta))
        r = pos - w * q
        v = w * q * (1 - q)
        if fit_slope:
            vl = v * lp
            h01 = vl.sum()
            grad = np.array([r.sum(), np.dot(r, lp)])
            hess = np.array([[v.sum(), h01], [h01, np.dot(vl, lp)]])
        else:
            grad = np.array([r.sum()])
            hess = np.array([[v.sum()]])
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            return None
        if not np.isfinite(step).all():
            return None
        t = 1.0
        if np.abs(step).max() > 1.0:
            cur = loglik(beta)
            while loglik(beta + t * step) < cur and t > 1e-8:
                t /= 2
        beta = beta + t * step
        if np.abs(beta).max() > max_coef:
            return None
        if np.abs(t * step).max() < tol:
            return beta
    return None


def _k_citl(lp: np.ndarray, pos: np.ndarray, neg: np.ndarray, start=None) -> float:
    beta = _newton(lp, pos, neg, fit_slope=False, start=start)
    return NAN if beta is None else float(beta[0])


def _k_recal(
    lp: np.ndarray, pos: np.ndarray, neg: np.ndarray, start=None
) -> tuple[float, float]:
    used = (pos + neg) > 0
    if np.ptp(lp[used]) == 0:
        return NAN, NAN
    beta = _newton(lp, pos, neg, fit_slope=True, start=start)
    return (NAN, NAN) if beta is None else (float(beta[0]), float(beta[1]))


def _both_classes(y: np.ndarray, name: str) -> bool:
    if y.size == 0:
        _warn(f"{name}: amostra vazia; métrica indefinida.")
        return False
    if y.min() == y.max():
        _warn(f"{name}: só há uma classe no desfecho; métrica indefinida.")
        return False
    return True


def _logit_or_warn(g: _Grouped, name: str, clip: float | None) -> np.ndarray | None:
    lp = g.logit(clip)
    if lp is None:
        _warn(
            f"{name}: há probabilidade exatamente 0 ou 1, com logit infinito; "
            "métrica indefinida. Passe clip (ex.: 1e-6) para truncar de propósito."
        )
    return lp


# ── Discriminação e erro quadrático ──────────────────────────────────────────


def roc_auc(y_true, y_prob) -> float:
    """AUROC, com empate valendo meio. NaN com aviso quando só há uma classe."""
    y, p = _prepare(y_true, y_prob)
    if not _both_classes(y, "AUROC"):
        return NAN
    g = _Grouped(y, p)
    return _k_auc(g.pos, g.neg)


def average_precision(y_true, y_prob) -> float:
    """Average precision (PR-AUC), com a mesma definição do sklearn."""
    y, p = _prepare(y_true, y_prob)
    if not _both_classes(y, "AP"):
        return NAN
    g = _Grouped(y, p)
    return _k_ap(g.pos, g.neg)


def brier_score(y_true, y_prob) -> float:
    """Brier: média de (p - y)². NaN com aviso em amostra vazia."""
    y, p = _prepare(y_true, y_prob)
    if y.size == 0:
        _warn("Brier: amostra vazia; métrica indefinida.")
        return NAN
    return float(np.mean((p - y) ** 2))


# ── Calibração ───────────────────────────────────────────────────────────────


def calibration_in_the_large(y_true, y_prob, clip: float | None = None) -> float:
    """Calibration-in-the-large (CITL) por regressão logística com offset.

    Método: ajusta ``logit P(y=1) = a + logit(p)``, com ``logit(p)`` como
    offset (coeficiente fixo em 1) e só o intercepto ``a`` livre, por máxima
    verossimilhança (Newton-Raphson). ``a = 0`` é calibração perfeita na média;
    ``a > 0`` quer dizer que o modelo subestima o risco e ``a < 0``, que
    superestima.

    Não é o intercepto da recalibração logística com slope livre (ver
    ``logistic_recalibration``): os dois só coincidem quando o slope é 1.

    ``clip`` trunca p em [clip, 1 - clip] antes do logit. Sem ele, probabilidade
    exatamente 0 ou 1 devolve NaN com aviso.
    """
    _check_clip(clip)
    y, p = _prepare(y_true, y_prob)
    if not _both_classes(y, "CITL"):
        return NAN
    g = _Grouped(y, p)
    lp = _logit_or_warn(g, "CITL", clip)
    if lp is None:
        return NAN
    a = _k_citl(lp, g.pos, g.neg)
    if math.isnan(a):
        _warn("CITL: o ajuste não convergiu; métrica indefinida.")
    return a


def logistic_recalibration(y_true, y_prob, clip: float | None = None) -> tuple[float, float]:
    """Recalibração logística de Cox: ``logit P(y=1) = b0 + b1 * logit(p)``.

    Ajuste por máxima verossimilhança (Newton-Raphson), sem penalização.
    Devolve ``(b0, b1)``: ``b1`` é o calibration slope; ``b0`` é o intercepto
    da recalibração, que não é o CITL (esse fixa o slope em 1, ver
    ``calibration_in_the_large``).

    NaN com aviso quando só há uma classe, quando há probabilidade 0 ou 1
    exata sem ``clip``, quando a predição é constante (slope indefinido) ou
    quando o ajuste não converge (separação completa: o modelo separa as
    classes sem erro e o slope vai ao infinito).
    """
    _check_clip(clip)
    y, p = _prepare(y_true, y_prob)
    if not _both_classes(y, "Slope"):
        return NAN, NAN
    g = _Grouped(y, p)
    lp = _logit_or_warn(g, "Slope", clip)
    if lp is None:
        return NAN, NAN
    if np.ptp(lp) == 0:
        _warn("Slope: a probabilidade predita é constante; slope indefinido.")
        return NAN, NAN
    b0, b1 = _k_recal(lp, g.pos, g.neg)
    if math.isnan(b1):
        _warn(
            "Slope: a recalibração logística não convergiu (provável separação "
            "completa entre as classes); métrica indefinida."
        )
    return b0, b1


def calibration_slope(y_true, y_prob, clip: float | None = None) -> float:
    """Calibration slope: coeficiente de ``logit(p)`` na recalibração logística.

    1 é o ideal; abaixo de 1, o risco predito está espalhado demais (típico de
    sobreajuste ou de reamostragem); acima de 1, concentrado demais.
    """
    return logistic_recalibration(y_true, y_prob, clip=clip)[1]


def observed_expected_ratio(y_true, y_prob) -> float:
    """Razão O:E: eventos observados sobre eventos esperados (soma de p).

    1 é o ideal; abaixo de 1, o modelo superestima o risco; acima, subestima.
    NaN com aviso em amostra vazia ou quando a soma das probabilidades é 0.
    """
    y, p = _prepare(y_true, y_prob)
    if y.size == 0:
        _warn("O:E: amostra vazia; métrica indefinida.")
        return NAN
    expected = float(p.sum())
    if expected == 0:
        _warn("O:E: todas as probabilidades são 0, esperado nulo; métrica indefinida.")
        return NAN
    return float(y.sum()) / expected


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """ECE com ``n_bins`` bins de largura fixa em [0, 1].

    Média ponderada, pelo tamanho do bin, de ``|fração observada - p média|``.
    O bin i cobre [i/n, (i+1)/n); p = 1 entra no último bin. O valor depende
    do número de bins, que por isso é parâmetro e vai junto no relato. O ECE
    tem viés para cima em amostra pequena (o ruído dentro do bin nunca se
    cancela), então o IC por bootstrap tende a ficar acima do valor pontual.
    """
    if int(n_bins) < 1:
        raise ValueError("n_bins precisa ser pelo menos 1.")
    y, p = _prepare(y_true, y_prob)
    if y.size == 0:
        _warn("ECE: amostra vazia; métrica indefinida.")
        return NAN
    g = _Grouped(y, p)
    return _k_ece(g.u, g.pos, g.neg, int(n_bins))


# ── Intervalo de confiança por bootstrap ─────────────────────────────────────

_POINT = {
    "roc_auc": lambda y, p, n_bins, clip: roc_auc(y, p),
    "pr_auc": lambda y, p, n_bins, clip: average_precision(y, p),
    "brier": lambda y, p, n_bins, clip: brier_score(y, p),
    "citl": lambda y, p, n_bins, clip: calibration_in_the_large(y, p, clip=clip),
    "calibration_slope": lambda y, p, n_bins, clip: calibration_slope(y, p, clip=clip),
    "oe_ratio": lambda y, p, n_bins, clip: observed_expected_ratio(y, p),
    "ece": lambda y, p, n_bins, clip: expected_calibration_error(y, p, n_bins=n_bins),
}


def _replicate(name: str, u, lp, pos, neg, n_bins: int, starts: dict) -> float:
    if name == "roc_auc":
        return _k_auc(pos, neg)
    if name == "pr_auc":
        return _k_ap(pos, neg)
    if name == "brier":
        return _k_brier(u, pos, neg)
    if name == "oe_ratio":
        return _k_oe(u, pos, neg)
    if name == "ece":
        return _k_ece(u, pos, neg, n_bins)
    if lp is None:
        return NAN
    if name == "citl":
        return _k_citl(lp, pos, neg, start=starts.get("citl"))
    return _k_recal(lp, pos, neg, start=starts.get("recal"))[1]


def bootstrap_ci(
    y_true,
    y_prob,
    metrics: Sequence[str] | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    n_bins: int = 10,
    clip: float | None = None,
) -> dict[str, dict]:
    """IC por bootstrap percentil, estratificado pelo desfecho e semeado.

    Método: em cada uma das ``n_boot`` réplicas, sorteia com reposição tantos
    positivos quantos há na amostra, entre os positivos, e tantos negativos
    quantos há, entre os negativos. Toda réplica tem então a prevalência da
    amostra e as duas classes, e o AUROC nunca fica indefinido por acaso. O IC
    é o par de percentis ``alpha/2`` e ``1 - alpha/2`` das réplicas válidas. A
    mesma ``seed`` reproduz o mesmo IC.

    ``metrics`` são nomes de ``METRICS`` (padrão: todos). Devolve
    ``{nome: {"value", "ci_low", "ci_high", "n_valid"}}``, com o valor pontual
    na amostra inteira. Réplicas em que a métrica fica indefinida (NaN) são
    descartadas e contadas em ``n_valid``; se menos da metade for válida, o IC
    volta NaN com aviso.
    """
    names = list(METRICS if metrics is None else metrics)
    unknown = [m for m in names if m not in METRICS]
    if unknown:
        raise ValueError(f"Métrica desconhecida: {unknown}. Use {list(METRICS)}.")
    if int(n_boot) < 1:
        raise ValueError("n_boot precisa ser pelo menos 1.")
    if not 0 < alpha < 1:
        raise ValueError("alpha precisa estar em (0, 1).")
    if int(n_bins) < 1:
        raise ValueError("n_bins precisa ser pelo menos 1.")
    _check_clip(clip)
    y, p = _prepare(y_true, y_prob)
    n_boot, n_bins = int(n_boot), int(n_bins)

    out = {
        m: {"value": float(_POINT[m](y, p, n_bins, clip)), "ci_low": NAN,
            "ci_high": NAN, "n_valid": 0}
        for m in names
    }
    if y.size == 0:
        _warn("Bootstrap: amostra vazia; IC indefinido.")
        return out

    g = _Grouped(y, p)
    lp = g.logit(clip)
    # as réplicas partem da estimativa pontual: o Newton converge em poucos passos
    starts: dict = {}
    if lp is not None and "citl" in names and math.isfinite(out["citl"]["value"]):
        starts["citl"] = np.array([out["citl"]["value"]])
    if lp is not None and "calibration_slope" in names:
        b0, b1 = _k_recal(lp, g.pos, g.neg)
        if math.isfinite(b1):
            starts["recal"] = np.array([b0, b1])
    inv_pos, inv_neg = g.inv[y == 1], g.inv[y == 0]
    size = g.u.size
    rng = np.random.default_rng(seed)
    reps = {m: np.full(n_boot, np.nan) for m in names}
    for b in range(n_boot):
        pos = np.bincount(inv_pos[rng.integers(0, inv_pos.size, inv_pos.size)],
                          minlength=size).astype(float) if inv_pos.size else g.pos
        neg = np.bincount(inv_neg[rng.integers(0, inv_neg.size, inv_neg.size)],
                          minlength=size).astype(float) if inv_neg.size else g.neg
        for m in names:
            reps[m][b] = _replicate(m, g.u, lp, pos, neg, n_bins, starts)

    for m, vals in reps.items():
        valid = vals[np.isfinite(vals)]
        out[m]["n_valid"] = int(valid.size)
        if valid.size == 0 or valid.size < n_boot / 2:
            if math.isfinite(out[m]["value"]):
                _warn(
                    f"Bootstrap de {METRIC_LABELS[m]}: só {valid.size} de {n_boot} "
                    "réplicas válidas; IC indefinido."
                )
            continue
        out[m]["ci_low"] = float(np.quantile(valid, alpha / 2))
        out[m]["ci_high"] = float(np.quantile(valid, 1 - alpha / 2))
    return out


# ── Decision curve ───────────────────────────────────────────────────────────


def default_thresholds() -> np.ndarray:
    """Grade padrão de limiares da decision curve: 0,01 a 0,99, de 0,01 em 0,01."""
    return np.round(np.arange(1, 100) / 100, 2)


_DCA_COLUMNS = ["threshold", "net_benefit_model", "net_benefit_all",
                "net_benefit_none", "best_trivial", "gain"]


def decision_curve(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    """Benefício líquido do modelo, de tratar todos e de não tratar ninguém.

    Para cada limiar t, trata quem tem ``p >= t``:

    - modelo: ``VP/n - FP/n * t / (1 - t)``;
    - tratar todos: ``prev - (1 - prev) * t / (1 - t)``;
    - não tratar ninguém: 0.

    ``best_trivial`` é o máximo entre tratar todos e não tratar ninguém, e
    ``gain`` é o benefício do modelo menos ``best_trivial``, a comparação que o
    item 13 dos aprendizados pede, e não a superioridade sobre cada estratégia
    em separado. Com uma classe só, as colunas do modelo, de tratar todos e do
    ganho voltam NaN com aviso.
    """
    y, p = _prepare(y_true, y_prob)
    if thresholds is None:
        t = default_thresholds()
    else:
        t = np.asarray(thresholds, dtype=float).ravel()
    if t.size == 0 or not np.isfinite(t).all() or (t <= 0).any() or (t >= 1).any():
        raise ValueError("Os limiares da decision curve precisam estar em (0, 1).")
    df = pd.DataFrame({"threshold": t, "net_benefit_none": 0.0})
    if not _both_classes(y, "Decision curve"):
        for col in ("net_benefit_model", "net_benefit_all", "best_trivial", "gain"):
            df[col] = NAN
        return df[_DCA_COLUMNS]
    n = y.size
    odds = t / (1 - t)
    p_pos = np.sort(p[y == 1])
    p_neg = np.sort(p[y == 0])
    tp = p_pos.size - np.searchsorted(p_pos, t, side="left")  # p >= t
    fp = p_neg.size - np.searchsorted(p_neg, t, side="left")
    prev = p_pos.size / n
    df["net_benefit_model"] = tp / n - fp / n * odds
    df["net_benefit_all"] = prev - (1 - prev) * odds
    df["best_trivial"] = np.maximum(df["net_benefit_all"], 0.0)
    df["gain"] = df["net_benefit_model"] - df["best_trivial"]
    return df[_DCA_COLUMNS]


def _runs(thresholds: np.ndarray, mask: np.ndarray) -> list[tuple[float, float]]:
    """Trechos contíguos da grade em que ``mask`` é verdadeiro, como (início, fim)."""
    runs: list[tuple[float, float]] = []
    start = None
    for i, ok in enumerate(mask):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((float(thresholds[start]), float(thresholds[i - 1])))
            start = None
    if start is not None:
        runs.append((float(thresholds[start]), float(thresholds[len(mask) - 1])))
    return runs


def net_benefit_ranges(curve: pd.DataFrame, margin: float = 0.01) -> dict:
    """Faixas de limiar em que o modelo supera a melhor estratégia trivial.

    Devolve as duas faixas que o item 13 dos aprendizados pede separadas:

    - ``any_gain``: ganho acima de zero sobre ``max(tratar todos, ninguém)``,
      o critério ingênuo, que em limiar alto aceita ganho de terceira casa;
    - ``relevant_gain``: ganho de pelo menos ``margin`` em benefício líquido
      absoluto (0,01 é um verdadeiro positivo líquido a cada 100 pacientes).

    Cada faixa é uma lista de trechos contíguos da grade, ``(início, fim)``;
    ``*_contiguous`` diz se ela é um trecho só, para não reportar mínimo e
    máximo como intervalo quando há buraco no meio. A margem é decisão
    clínica, tomada antes de olhar a curva; o padrão 0,01 é o do exemplo dos
    aprendizados e fica inalcançável quando a prevalência é menor que isso.
    """
    if not margin >= 0:
        raise ValueError("margin precisa ser maior ou igual a 0.")
    t = curve["threshold"].to_numpy(dtype=float)
    gain = curve["gain"].to_numpy(dtype=float)
    finite = np.isfinite(gain)
    positive = finite & (gain > 1e-12)
    any_gain = _runs(t, positive)
    relevant = _runs(t, positive & (gain >= margin - 1e-12))
    best = int(np.nanargmax(gain)) if finite.any() else None
    return {
        "margin": float(margin),
        "any_gain": any_gain,
        "any_gain_contiguous": len(any_gain) <= 1,
        "relevant_gain": relevant,
        "relevant_gain_contiguous": len(relevant) <= 1,
        "max_gain": float(gain[best]) if best is not None else NAN,
        "threshold_max_gain": float(t[best]) if best is not None else NAN,
    }


def format_ranges(runs: list[tuple[float, float]]) -> str:
    """Texto de uma faixa: ``"18% a 28%"``, com trechos separados por ``;``."""
    if not runs:
        return "nenhuma"
    return "; ".join(
        f"{lo:.0%}" if lo == hi else f"{lo:.0%} a {hi:.0%}" for lo, hi in runs
    )


# ── Tamanho de amostra ───────────────────────────────────────────────────────


def events_per_variable(y_true, n_predictors: int, min_epv: float = 10.0) -> dict:
    """Eventos por variável (EPV) e o N mínimo para atingir ``min_epv``.

    Os eventos são a classe menos frequente. ``n_predictors`` é o número de
    parâmetros candidatos (com one-hot, cada nível conta). ``n_min`` é o N
    total que daria ``min_epv`` na prevalência atual:
    ``ceil(min_epv * n_predictors * n / eventos)``.

    Emite ``MetricWarning`` quando o EPV fica abaixo de ``min_epv``: com poucos
    casos da classe rara por variável, o modelo sobreajusta (CP6 da
    ml-checkpoints). O padrão de 10 é a regra clássica, um piso e não um alvo.
    """
    if int(n_predictors) < 1:
        raise ValueError("n_predictors precisa ser pelo menos 1.")
    y = np.asarray(y_true, dtype=float).ravel()
    if np.isnan(y).any() or not np.isin(y, (0.0, 1.0)).all():
        raise ValueError("y_true precisa ser binário (0 ou 1), sem ausentes.")
    n = int(y.size)
    n_pos = int(y.sum())
    n_events = min(n_pos, n - n_pos)
    k = int(n_predictors)
    epv = n_events / k
    ok = epv >= min_epv
    if n_events == 0:
        _warn("EPV: só há uma classe no desfecho (ou nenhuma linha); N mínimo indefinido.")
        n_min = NAN
    else:
        n_min = float(math.ceil(min_epv * k * n / n_events))
        if not ok:
            _warn(
                f"EPV {epv:.1f} abaixo de {min_epv:g}: {n_events} eventos para {k} "
                f"variáveis. Na prevalência atual, seriam precisos cerca de "
                f"{int(n_min):,} registros, ou menos variáveis."
            )
    return {
        "n": n,
        "n_events": n_events,
        "n_predictors": k,
        "epv": float(epv),
        "min_epv": float(min_epv),
        "n_min": n_min,
        "ok": bool(ok),
    }


# ── Resumo para a tela e o relatório ─────────────────────────────────────────


def performance_summary(
    y_true,
    y_prob,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    n_bins: int = 10,
    clip: float | None = None,
) -> dict:
    """Todas as ``METRICS`` com IC por bootstrap, mais os avisos em texto.

    Devolve ``{"n", "n_events", "prevalence", "n_boot", "alpha", "seed",
    "n_bins", "metrics", "warnings"}``, em que ``metrics`` é a saída de
    ``bootstrap_ci`` e ``warnings`` são os textos dos ``MetricWarning``, sem
    repetição, para a tela mostrar.
    """
    y, p = _prepare(y_true, y_prob)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", MetricWarning)
        res = bootstrap_ci(y, p, n_boot=n_boot, alpha=alpha, seed=seed,
                           n_bins=n_bins, clip=clip)
    msgs: list[str] = []
    for w in caught:
        if issubclass(w.category, MetricWarning) and str(w.message) not in msgs:
            msgs.append(str(w.message))
    return {
        "n": int(y.size),
        "n_events": int(y.sum()),
        "prevalence": float(y.mean()) if y.size else NAN,
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "seed": int(seed),
        "n_bins": int(n_bins),
        "metrics": res,
        "warnings": msgs,
    }


def summary_table(summary: dict, decimals: int = 4) -> pd.DataFrame:
    """Tabela da ``performance_summary``: métrica, valor, IC e valor ideal."""
    level = round((1 - summary.get("alpha", 0.05)) * 100)

    def _r(v: float) -> float:
        return round(v, decimals) if math.isfinite(v) else NAN

    rows = [
        {
            "Métrica": METRIC_LABELS.get(key, key),
            "Valor": _r(r["value"]),
            f"IC {level}% inf.": _r(r["ci_low"]),
            f"IC {level}% sup.": _r(r["ci_high"]),
            "Ideal": METRIC_REFERENCE.get(key, ""),
        }
        for key, r in summary["metrics"].items()
    ]
    return pd.DataFrame(rows)
