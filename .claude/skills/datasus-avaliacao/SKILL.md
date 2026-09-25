# Skill: Avaliação e Métricas — lab-ai-prediction

Referência para interpretar, modificar ou adicionar métricas e visualizações.

**Arquivo central:** `core/models/evaluation.py`
**Calibração:** `calibrate_model` em `core/models/pipeline.py`, chamada na seção Calibração da etapa de resultados (`pages/analise.py`)
**Benchmark entre estados:** `pages/calibracao.py` (etapa 8)
**SHAP:** integrado em `evaluation.py`, `pages/analise.py` e `pages/deploy.py`

As regras de métrica, ponto de corte, calibração e interpretabilidade são da skill `ml-checkpoints` do [labskills](https://github.com/labdaps/labskills/tree/main/skills/ml-checkpoints) (CP8 a CP10), a norma do laboratório, e o relatório segue a `ml-eval-report`. Esta skill diz onde cada coisa está no código do app e não repete essas regras.

---

## Funções de avaliação disponíveis

```python
from core.models.evaluation import (
    roc_chart,
    pr_chart,
    calibration_chart,
    importance_chart,
    shap_summary,
)
```

| Função               | Entrada                         | Saída (Plotly)                         |
|----------------------|---------------------------------|----------------------------------------|
| `roc_chart(y, p)`    | y_true, oof_probs               | Curva ROC com AUC no título            |
| `pr_chart(y, p)`     | y_true, oof_probs               | Curva PR + linha de baseline (prevalência) |
| `calibration_chart(y, p, n_bins=10)` | y_true, oof_probs | Scatter calibração + diagonal perfeita |
| `importance_chart(importances, top_n=20)` | feature_importances dict | Barras horizontais top-20 (usado apenas no relatório/deploy) |
| `shap_summary(model, X, max_display=20)` | pipeline treinado, X | Barras mean \|SHAP\| top-20 |
| `shap_beeswarm(model, X, max_display=15)` | pipeline treinado, X | Scatter por amostra: X=SHAP, Y=feature, cor=valor |

Todos retornam `plotly.graph_objects.Figure` — renderizar com `st.plotly_chart(fig)`.

Também em `evaluation.py`: `threshold_metrics(y, p, threshold=0.5)` (sensibilidade, especificidade, VPP, VPN e NNT num corte), `threshold_curve_chart(y, p)`, `subgroup_metrics_table(y, p, groups)`, `fold_metrics_table(fold_metrics)`, `shap_waterfall_chart(model, X, case_idx=0)`, `shap_values_dict(model, X, max_rows=500)`, `calibration_comparison_chart(y, probs_before, probs_after, method_label="Calibrado", n_bins=10)`, `metrics_comparison_table(comparison_results)` e `shap_comparison_chart(shap_dicts, labels, top_n=15)`.

---

## Que métricas reportar

A métrica principal, o ponto de corte e o que reportar de calibração e de utilidade clínica saem da `ml-checkpoints`: a métrica principal e o ponto de corte são escolhidos no CP8, antes de rodar, e a decision curve compara o modelo com tratar todos e com não tratar ninguém; a calibração é reportada com slope, intercepto e Brier (CP9). A tabela do relatório é a da `ml-eval-report`. O porquê está em `docs/aprendizados-pipeline-agentes.md`, no ai-lab-hub: o item 7 trata do O:E e do calibration-in-the-large que o balanceamento degrada, e o item 13, da margem sobre a melhor estratégia trivial na decision curve.

O que o app calcula hoje e o que falta:

| Item | No código do app |
|---|---|
| ROC-AUC, PR-AUC, F1, precisão, recall, especificidade, Brier | `_compute_metrics` em `core/models/pipeline.py`; as de classe no corte 0,5 |
| métricas num corte escolhido | `threshold_metrics(y, p, threshold)` |
| curva de calibração e Brier antes e depois de calibrar | `calibration_chart`, `calibrate_model`, `calibration_comparison_chart` |
| O:E, calibration-in-the-large, slope, IC e decision curve | ainda não existem no código: calcule fora do app e reporte ao lado |

---

## Interpretação da curva de calibração

```
Eixo X: Probabilidade média predita (em bins de 0.1)
Eixo Y: Fração real de positivos no bin

Diagonal perfeita = modelo bem calibrado
Curva acima da diagonal = subestimando risco (conservative)
Curva abaixo da diagonal = superestimando risco (overconfident)
```

Para recalibrar, use a função do app, que mede o Brier antes e depois em dado que nem o modelo nem o calibrador viram (ver skill `datasus-pipeline`):
```python
from core.models.pipeline import calibrate_model
cal = calibrate_model(res["model"], X, y, method="sigmoid")  # ou "isotonic"
# cal["brier_before"], cal["brier_after"], cal["raw_probs"], cal["cal_probs"], cal["y_eval"]
```

---

## Nomes limpos nos gráficos SHAP e Importância

Todos os gráficos que exibem nomes de features usam labels legíveis via `_label()` e `_apply_labels()` definidas no topo de `core/models/evaluation.py`:

- `_label(col)` tira o prefixo do `ColumnTransformer` (`cat_none__SEXO` vira `SEXO`) e o sufixo numérico do one-hot (`SEXO_1` vira `SEXO`), procura o nome em `get_info` de `core/features/data_dict.py` e, se não achar, devolve o nome limpo.
- `_apply_labels(names)` aplica `_label` a uma lista.

**Onde é aplicado:**

| Função                  | Como aplica                                  |
|-------------------------|----------------------------------------------|
| `importance_chart()`    | soma as importâncias por rótulo (`groupby(_label)`), então os níveis do one-hot da mesma variável se somam |
| `shap_summary()`        | `feat_labels = _apply_labels(feat_names)`    |
| `shap_beeswarm()`       | `feat_labels = _apply_labels(feat_names)`    |
| `shap_waterfall_chart()`| `index=_apply_labels(feat_names[:n])`        |
| `shap_comparison_chart()`| `top_labels = _apply_labels(top_features)`  |

**Regra:** Qualquer novo gráfico que exiba nomes de features deve chamar `_apply_labels()` antes de passar para o Plotly. Garantir que a variável esteja cadastrada em `core/features/data_dict.py`.

---

## SHAP Beeswarm

`shap_beeswarm(model, X, max_display=15)` — cada ponto é uma amostra:
- **X:** valor SHAP da feature para aquela amostra (negativo = reduz risco, positivo = aumenta)
- **Y:** posição da feature + jitter aleatório (seed=42) para separar os pontos
- **Cor:** valor normalizado da feature (azul = baixo, cinza = médio, vermelho = alto)
- Linha vertical tracejada em x=0 como referência
- Colorbar exibida apenas na última trace (evita repetição)

Na UI (`pages/analise.py`, seção `shap_global`):
```python
shap_fig = ev.shap_summary(results["model"], X_res.head(500))   # barras
shap_bee = ev.shap_beeswarm(results["model"], X_res.head(500))  # beeswarm
```
O `importance_chart` foi removido da aba SHAP Global — permanece apenas em `relatorio.py` e `deploy.py`.

---

## SHAP — lógica de extração

```python
# _unwrap_model separa o estimador do resto do pipeline; aceita também o
# CalibratedClassifierCV devolvido por calibrate_model (usa .estimator)
estimator, prep = _unwrap_model(model)       # model[-1], model[:-1]
X_arr = np.asarray(prep.transform(X), dtype=float)
feat_names = _feat_names_after_transform(prep, X.columns.tolist(), X_arr.shape[1])

# tenta TreeExplainer; se falhar, LinearExplainer com masker Independent
try:
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(X_arr, check_additivity=False)
except Exception:
    masker = shap.maskers.Independent(X_arr, max_samples=min(100, len(X_arr)))
    shap_values = shap.LinearExplainer(estimator, masker).shap_values(X_arr)

shap_values = _extract_shap_2d(shap_values)  # lista ou 3D vira (n, features) da classe 1
```

O passo de pré-processamento do pipeline se chama `prep`, não `preprocessor` (ver `build_pipeline`). Sem o pacote `shap` instalado, ou quando nenhum dos dois explainers funciona, as funções devolvem `None`.

SHAP para deploy individual (`pages/deploy.py`): usa `shap_waterfall_chart(model, input_df, case_idx=0)`, em Plotly.

---

## Brier Score

```
Brier Score = mean((p_pred - y_true)²)

0.0 = perfeito
0.25 = modelo sem informação (prevalência = 50%)
Para prevalências diferentes de 50%:
  Brier Score ref = prevalência × (1 - prevalência)
  Brier Skill Score = 1 - (BS / BS_ref)   → quanto melhor que o naive
```

---

## Análise de Equidade por Subgrupo (`pages/analise.py` — aba Equidade)

A UI detecta automaticamente quais variáveis demográficas estão presentes na coorte e oferece estratificação por elas.

**Candidatos reconhecidos** (ordem de prioridade, em `_fairness_candidates`):

```python
# SIH / SINASC
"SEXO", "RACA_COR", "UF_SIGLA", "UF_ZI", "UF_NASC", "MUNIC_RES",

# SINAN (TB, Hanseníase, Arboviroses, AIDS, Sífilis, Violência, Intoxicação)
"CS_SEXO", "CS_RACA", "age_group",
```

A lista é filtrada para mostrar apenas colunas presentes em `cohort.columns`.

**Mapeamento por base:**

| Base     | Sexo     | Raça/Cor  | Localização | Faixa etária |
|----------|----------|-----------|-------------|--------------|
| SIH      | `SEXO`   | `RACA_COR`| `UF_SIGLA`  | `age_group`  |
| SINASC   | `SEXO`   | `RACACORMAE` | `UF_SIGLA` | `age_group_mae` |
| SINAN_*  | `CS_SEXO`| `CS_RACA` | —           | `age_group`  |

`RACACORMAE` e `age_group_mae` ainda não estão em `_fairness_candidates`, então a tela não os oferece para o SINASC. Ao adicionar suporte a uma nova base, incluir as variáveis demográficas equivalentes na lista `_fairness_candidates` em `pages/analise.py`. A tabela da tela vem de `subgroup_metrics_table`, que omite grupo com N abaixo de 20 ou com uma classe só.

---

## Benchmarking por estado/período (`pages/calibracao.py`)

Etapa 8 da tela: aplica o modelo treinado (o calibrado, se houver) a coortes de outros estados e anos, baixadas de novo, ou a subgrupos de uma coluna no fluxo DIY, e compara o desempenho.

Fluxo em `pages/calibracao.py`:
1. Montar a coorte de cada grupo com `fetch` e `CohortBuilder`, como na etapa 2
2. `_apply_model_to_subset(X_sub, y_sub, label)`: prevê com o modelo ativo e calcula `roc_auc`, `pr_auc`, `f1`, `recall` e `brier` (corte 0,5), mais a importância por permutação do grupo; pula grupo com menos de 20 casos ou uma classe só
3. `_render_comparison(comp)`: tabela `metrics_comparison_table`, curva ROC por grupo e importâncias por grupo

A página não desenha curva de calibração por grupo. Para checar calibração entre estados, use `calibration_chart(r["y_true"], r["oof_probs"])` em cada resultado da comparação e reporte o que o CP9 da `ml-checkpoints` pede.

---

## Adicionando nova métrica

1. Calcular em `_compute_metrics(y_true, probs, preds)` em `core/models/pipeline.py`, que alimenta `fold_metrics` e `mean_metrics` do `train_cv`
2. Repetir o cálculo nos ramos de holdout e de corte temporal da etapa de treino em `pages/analise.py`, que montam o dict de métricas por conta própria, e em `_apply_model_to_subset` de `pages/calibracao.py`
3. Exibir na etapa de resultados (`pages/analise.py`, etapa 7) e, se for o caso, em `fold_metrics_table`
4. Se precisar de gráfico: adicionar função em `core/models/evaluation.py`

---

## Métricas por fold (CV)

```python
# Devolvido por train_cv()
{
    "fold_metrics": [                # um dict por fold
        {"roc_auc": ..., "pr_auc": ..., "f1": ..., "precision": ..., "recall": ...,
         "specificity": ..., "brier": ..., "fold": 1},
        ...
    ],
    "mean_metrics": {"roc_auc": ..., ...},   # só a média entre folds
    "oof_probs": np.array([...]),    # probabilidade out-of-fold, na ordem de y
    "feature_importances": {...},
    "model": pipeline,               # reajustado em X inteiro
    "X_columns": [...],
    "algorithm": "lgbm",
}
```

Não há `oof_true`: o `y` passado ao `train_cv` já está na ordem de `oof_probs`. O desvio padrão não vem pronto; calcule de `fold_metrics`, por exemplo `np.std([f["roc_auc"] for f in res["fold_metrics"]])`. Em holdout e corte temporal, a tela monta um dict parecido com um único item em `fold_metrics`, e `oof_probs` passa a ser a previsão no teste, com `y_eval` ao lado.
