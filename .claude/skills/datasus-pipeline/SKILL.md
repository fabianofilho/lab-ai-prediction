# Skill: Pipeline ML do lab-ai-prediction

Referência da API de treino do app, para modificar, entender ou depurar o pipeline. Descreve o que o código faz hoje. As decisões de método (separação, faltantes, desbalanceamento, métrica, calibração) são da skill `ml-checkpoints` do [labskills](https://github.com/labdaps/labskills/tree/main/skills/ml-checkpoints), a norma do laboratório; o porquê de cada regra está em `docs/aprendizados-pipeline-agentes.md`, no ai-lab-hub. Esta skill só diz onde cada decisão entra no código.

**Arquivo central:** `core/models/pipeline.py`
**Avaliação:** `core/models/evaluation.py` (ver skill `datasus-avaliacao`)

Se esta skill e o código divergirem, vale o código: corrija a skill no mesmo PR.

---

## Estrutura do pipeline

`build_pipeline(X, algorithm="lgbm", params=None, use_smote=False, balancing="none", treatment=None)` monta os passos nesta ordem:

| Passo | Quando entra | O que faz |
|---|---|---|
| `sentinel` | só se `treatment["null_sentinels"]` não estiver vazio | `SentinelReplacer(sentinels)` troca os valores listados por NaN |
| `prep` | sempre | `ColumnTransformer` de `_build_preprocessor`: imputação, encoding e escala por coluna |
| `scaler` | `logreg` e `mlp`, se `num_default` não for `standard` nem `minmax` | `StandardScaler()` |
| `resample` | `balancing` igual a `smote_over` ou `smote_under` | SMOTE ou SMOTETomek, só no `fit` |
| `model` | sempre | `_build_model(algorithm, params, class_weight)` |

Com reamostragem o retorno é um `imblearn.pipeline.Pipeline`; sem ela, um `sklearn.pipeline.Pipeline`. Se o `imbalanced-learn` não estiver instalado, a reamostragem é pulada sem aviso. Com `tabpfn` e mais de `TABPFN_MAX_TRAIN_SAMPLES` (10.000) linhas, levanta `ValueError`.

---

## Algoritmos disponíveis

`ALGORITHMS` mapeia o nome de exibição para a chave usada na API:

| Chave | Classe | Padrões em `_build_model` |
|---|---|---|
| `lgbm` | `LGBMClassifier` | n_estimators=300, learning_rate=0.05, max_depth=-1, num_leaves=31 |
| `xgb` | `XGBClassifier` | n_estimators=150, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=-1 |
| `catboost` | `CatBoostClassifier` | iterations=300, learning_rate=0.05, depth=6, verbose=0 |
| `logreg` | `LogisticRegression` | C=1.0, max_iter=1000; recebe o passo `scaler` |
| `rf` | `RandomForestClassifier` | n_estimators=200, n_jobs=-1 |
| `mlp` | `MLPClassifier` | hidden_layer_sizes=(64, 32), early_stopping=True; recebe o passo `scaler` |
| `tabpfn` | `TabPFNClassifier` | opcional (`pip install tabpfn`), n_estimators=8 |

`TabPFN` aparece sempre em `ALGORITHMS`; `TABPFN_AVAILABLE` diz se ele está instalado, e `_build_model` levanta `ImportError` quando não está. As famílias que escolhem a curva de treino (`training_curve`) são `BOOSTING_ALGORITHMS = {"lgbm", "xgb", "catboost"}` e `NEURAL_ALGORITHMS = {"mlp"}`.

---

## Balanceamento de classes

| `balancing` | Rótulo na tela | Mecanismo |
|---|---|---|
| `none` | Nenhum | sem ajuste; padrão da API e da tela |
| `class_weight` | Class Weight | `class_weight="balanced"` em `lgbm`, `logreg` e `rf`; `auto_class_weights="Balanced"` no `catboost`; sem efeito em `xgb`, `mlp` e `tabpfn` |
| `smote_over` | SMOTE (oversample) | SMOTE com `k_neighbors` ajustado ao tamanho da classe minoritária do fold (`_safe_over_sampler`) |
| `smote_under` | SMOTE + Undersampling | SMOTETomek com o mesmo SMOTE interno (`_safe_combine_sampler`) |

`use_smote=True` é o parâmetro legado: com `balancing="none"`, equivale a `smote_over`.

O padrão é `none`, na API e na tela. Balancear muda a probabilidade prevista, então `class_weight` ou reamostragem só entram com a calibração medida com e sem o balanceamento: compare o Brier (`train_cv(...)["mean_metrics"]["brier"]`) e a curva de calibração (`calibration_chart`) nas mesmas partições, e registre a decisão com o motivo. Quando considerar balancear é decisão do CP5 da `ml-checkpoints`, e o que medir na calibração é do CP9; esta skill não repete essas regras. Slope e O:E ainda não estão no código (ver `datasus-avaliacao`).

---

## Tratamento por coluna (`treatment`)

`treatment` é o dict que a tela grava em `st.session_state["treatment_config"]` no passo de tratamento de `pages/analise.py`:

```python
treatment = {
    "num_cols": [...],                 # colunas tratadas como numéricas
    "cat_cols": [...],                 # colunas tratadas como categóricas
    "num_default": "none",             # none | standard | minmax | robust | bin | drop
    "cat_default": "none",             # none | ohe | ordinal | target | drop
    "overrides": {"COLUNA": "ohe"},    # tratamento de uma coluna específica
    "null_sentinels": [],              # valores trocados por NaN em todas as colunas
}
```

O que `_build_preprocessor(X, treatment, algorithm)` monta para cada tratamento:

| Tipo | Tratamento | Transformação |
|---|---|---|
| numérica | `none` | `SimpleImputer(strategy="median")` |
| numérica | `standard`, `minmax`, `robust` | mediana e depois `StandardScaler`, `MinMaxScaler` ou `RobustScaler` |
| numérica | `bin` | mediana e depois `KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")` |
| categórica | `none` ou `ordinal` | `SimpleImputer(strategy="most_frequent")` e `OrdinalEncoder` (categoria nova vira -1) |
| categórica | `ohe` | moda e `OneHotEncoder(handle_unknown="ignore")` |
| categórica | `target` | `TargetEncoder(smooth="auto")`, sem imputador; cai para ordinal se o sklearn não tiver `TargetEncoder` |
| qualquer | `drop` | a coluna fica fora do modelo |

Padrões: com `treatment=None`, os tipos saem do dtype e o tratamento é `none` para numéricas e `ohe` para categóricas; como o `build_features` dos desfechos DATASUS já converte várias categóricas em código numérico, elas caem no grupo numérico. Na tela, os dois padrões são `none`, e o tipo de cada coluna vem primeiro do `data_dict` (Categórica e Ordinal viram categóricas), depois do dtype, e pode ser trocado coluna a coluna.

---

## Sentinelas (`SentinelReplacer`)

`SentinelReplacer(sentinels=None)` troca cada valor da lista por NaN em todas as colunas, antes do imputador:

```python
from core.models.pipeline import SentinelReplacer

SentinelReplacer(sentinels=[9, 99])
```

Na API não há lista padrão: o passo só entra quando `treatment["null_sentinels"]` tem valores. Na tela, a seleção vem com `[9, 99]` marcados, e a lista vale para todas as colunas: idade 9 ou 99 e a categoria que recebeu o código 9 também viram NaN. A norma do lab é sentinela por variável, a partir do dicionário (CP3 da `ml-checkpoints`).

---

## Separação e busca de hiperparâmetros

```python
from core.models.pipeline import split_train_test, optimize_hyperparams

X_tr, X_te, y_tr, y_te = split_train_test(X, y, "holdout", holdout_size=0.2)
# ou split_train_test(X, y, "temporal", dates=datas, cutoff="AAAA-MM-DD")
params = optimize_hyperparams(
    X_tr, y_tr, algorithm="lgbm", n_trials=50, n_folds=3,
    balancing="none", treatment=treatment, seed=42,
)
```

| Modo na tela | Função | Espaço de busca |
|---|---|---|
| Optuna (automático) | `optimize_hyperparams(X, y, algorithm, n_trials=50, n_folds=3, use_smote, balancing, treatment, progress_callback, seed=42)` | `_suggest_params(trial, algorithm)`, com `TPESampler(seed=seed)` e sem pruner |
| Random Search | `random_search(X, y, algorithm, n_iter=30, n_folds=3, balancing, treatment, progress_callback)` | `_RANDOM_GRIDS`, via `RandomizedSearchCV` |
| Grid Search | `grid_search(X, y, algorithm, n_folds=3, balancing, treatment, progress_callback)` | `_GRID_GRIDS`, via `GridSearchCV` |

As três maximizam ROC-AUC em `StratifiedKFold(n_folds, shuffle=True, random_state=42)` e devolvem o dict de hiperparâmetros sem o prefixo `model__`. Passe só a partição de treino. Em holdout e corte temporal, a tela separa com `split_train_test` antes da busca. Em validação cruzada, a tela busca na mesma coorte que o `train_cv` avalia depois, sem CV aninhada, e a métrica sai otimista.

---

## Validação cruzada

```python
from core.models.pipeline import train_cv

res = train_cv(
    X, y,
    algorithm="lgbm",
    params=params,
    n_folds=5,
    balancing="none",
    treatment=treatment,
)
```

- `StratifiedKFold(n_folds, shuffle=True, random_state=42)` por linha, sem grupo. Com identificador que repete, a separação por grupo é feita fora (CP2 da `ml-checkpoints`).
- Em cada fold, `build_pipeline` é ajustado no treino do fold e prevê o hold-out; as previsões formam `oof_probs`.
- `_compute_metrics` calcula por fold `roc_auc`, `pr_auc`, `f1`, `precision`, `recall`, `specificity` e `brier`. As métricas de classe usam o corte 0,5.
- Devolve um dict com `fold_metrics` (lista de dicts, com a chave `fold`), `mean_metrics` (só a média; NaN vira 0.0), `oof_probs`, `feature_importances` (média entre folds, com os nomes de saída do `ColumnTransformer`), `model` (pipeline reajustado em `X` inteiro), `X_columns` e `algorithm`.
- O desvio padrão não vem pronto: calcule a partir de `fold_metrics`.

---

## Calibração pós-treino

```python
from core.models.pipeline import calibrate_model

cal = calibrate_model(res["model"], X, y, method="sigmoid")  # sigmoid (Platt) | isotonic
```

- Clona o modelo, reajusta em 50% dos dados, ajusta o calibrador em 25% e mede o Brier antes e depois nos 25% restantes, que nenhum dos dois viu (partição estratificada, `random_state=7`).
- O calibrador é `CalibratedClassifierCV` sobre `FrozenEstimator` (sklearn 1.6 ou mais novo), ou com `cv="prefit"` nas versões antigas.
- Devolve `cal_model`, `method`, `raw_probs`, `cal_probs`, `y_eval`, `brier_before`, `brier_after` e `brier_delta`.
- Se a partição estratificada falhar (desfecho raríssimo), calibra o modelo recebido numa fração `cal_fraction` e mede o Brier nessa mesma fração: o antes e depois deixa de ser held-out, e isso precisa aparecer no relatório.
- Na tela, a chamada fica na seção Calibração da etapa de resultados (`pages/analise.py`).

---

## Adicionando novo algoritmo

1. Construir o estimador em `_build_model(algorithm, params, class_weight)`, repassando `class_weight` se ele aceitar.
2. Incluir as grades em `_RANDOM_GRIDS` e `_GRID_GRIDS`, com o prefixo `model__`.
3. Incluir o espaço do Optuna em `_suggest_params(trial, algorithm)`.
4. Incluir o nome de exibição e a chave em `ALGORITHMS`.
5. Se for boosting ou rede neural, incluir a chave em `BOOSTING_ALGORITHMS` ou `NEURAL_ALGORITHMS`. Boosting também precisa de ramo em `_boosting_round_aucs`, `_staged_proba` e `_extract_trees`, que desenham a curva de treino.

---

## Fluxo na tela (`pages/analise.py`)

```
Etapa 1       desfecho: escolhido no catálogo (pages/datasus.py), lido de OUTCOMES[key]
Etapa 2       dados: fetch(...) de core/data/downloader.py; CohortBuilder(outcome).build(raw)
Etapa 3       seleção de features; X, y de CohortBuilder(outcome).get_Xy(cohort)
Etapa 4       tratamento: treatment_config, com tratamento por coluna e null_sentinels
Etapa 5       modelo: algoritmos, validação (k-fold, holdout ou temporal), balancing, modo de busca
Etapa 6       treino: split_train_test (holdout e temporal), busca de hiperparâmetros,
              train_cv (k-fold) ou build_pipeline(...).fit (holdout e temporal)
Etapa 7       resultados: roc_chart, pr_chart, calibration_chart, shap_summary,
              shap_beeswarm, calibrate_model
```

As etapas 8 a 10 ficam em `pages/calibracao.py` (benchmark entre estados), `pages/deploy.py` e `pages/relatorio.py`.
