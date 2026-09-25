# Skill: Feature Engineering DATASUS — lab-ai-prediction

Referência para construção e transformação de features nos pipelines.

**Arquivo central:** `core/features/engineering.py`
**Cohort builder:** `core/features/cohort.py` → `CohortBuilder`
**Metadados:** `core/features/data_dict.py` → `get_info(feature_name)`

---

## Funções disponíveis em `engineering.py`

```python
from core.features import engineering as eng

# Label-encoding seguro para categóricas (NaN preservado)
df = eng.encode_categoricals(df, cat_cols=["SEXO", "RACACORMAE"])

# ICD-10: extrai capítulo (letra) como código numérico
df["diag_chapter"] = eng.icd10_chapter(df["DIAG_PRINC"])

# ICD-10: extrai bloco (3 caracteres) como código numérico
df["diag_block"] = eng.icd10_block(df["DIAG_PRINC"])

# Faixas etárias (bins padrão: 0,1,5,18,40,60,80,120 → labels 0-6)
df["age_group"] = eng.age_group(pd.to_numeric(df["IDADE"], errors="coerce"))

# Indicador de missingness (adiciona coluna {col}_missing)
df = eng.flag_missing(df, cols=["PESO", "CONSULTAS"])

# Clipa outliers nos percentis 1% e 99%
df = eng.clip_outliers(df, col="VAL_TOT", lower_q=0.01, upper_q=0.99)
```

---

## Encoding manual de categóricas (padrão nos outcomes)

```python
# Forma mais comum nos outcome.build_features()
for col in ["PARTO", "GRAVIDEZ", "SEXO"]:
    if col in df.columns:
        df[col] = pd.Categorical(df[col].astype(str)).codes.astype(float)
        # Resultado: -1 → NaN (códigos de missing automáticos)
```

---

## CohortBuilder

```python
from core.features.cohort import CohortBuilder
from core.outcomes import OUTCOMES

outcome = OUTCOMES["prematuridade"]
builder = CohortBuilder(outcome)

# raw_data = dict[str, DataFrame], um DataFrame bruto por sistema (ex.: {"SINASC": df})
cohort = builder.build(raw_data)   # outcome.build_cohort(raw_data) e depois outcome.build_features(cohort)
X, y = builder.get_Xy(cohort)      # y = outcome.get_target(cohort); X = suggested_features presentes

# Também: builder.split(cohort, test_size=0.2) e builder.class_balance(cohort)
```

`get_Xy` usa as colunas de `suggested_features` que existem na coorte; se nenhuma existir, usa todas as numéricas menos o alvo.

---

## Metadados de features (data_dict.py)

```python
from core.features.data_dict import get_info

info = get_info("GESTACAO")
# → {"label": "Semanas de Gestação", "desc": "...", "type": "Ordinal",
#    "values": {"1": "< 22 semanas", ...}}

info = get_info("preterm")
# → {"label": "Prematuridade", "type": "Derivada", ...}

# Retorna None se variável não cadastrada
```

Usar `info.get("label", col)` como fallback seguro na UI.

---

## Features derivadas comuns e como criá-las

| Feature derivada   | Coluna origem | Lógica                                              |
|--------------------|---------------|-----------------------------------------------------|
| `preterm`          | GESTACAO      | `GESTACAO.isin({"1","2","3","4"}).astype(int)`      |
| `low_birth_weight` | PESO          | `(PESO < 2500).astype(int)`                         |
| `used_icu`         | UTI_MES_TO    | `(UTI_MES_TO > 0).astype(int)`                      |
| `length_of_stay_days` | DT_INTER, DT_SAIDA | `(DT_SAIDA - DT_INTER).dt.days`            |
| `age_group`        | IDADE         | `eng.age_group(pd.to_numeric(IDADE, errors='coerce'))` |
| `age_group_mae`    | IDADEMAE      | `eng.age_group(pd.to_numeric(IDADEMAE, errors='coerce'))` |
| `diag_chapter`     | DIAG_PRINC    | `eng.icd10_chapter(DIAG_PRINC)`                     |
| `diag_block`       | DIAG_PRINC    | `eng.icd10_block(DIAG_PRINC)`                       |
| `n_diag_sec`       | DIAG_SEC, DIAGSEC1 ou DIAG_SECUN | `(~df[diag_sec_col].isna()).astype(int)`, presença do diagnóstico secundário |
| `hiv_pos`          | HIV           | `(HIV.astype(str) == "1").astype(int)`              |
| `dot`              | TRAT_SUPER    | `(TRAT_SUPER == "1").astype(int)`                   |

---

## Regras anti-leakage por desfecho

| Desfecho             | Colunas a REMOVER de features         | Motivo                          |
|----------------------|----------------------------------------|---------------------------------|
| prematuridade        | GESTACAO, preterm, PESO, APGAR1, APGAR5 | Target deriva de GESTACAO       |
| baixo_peso_nascer    | PESO, low_birth_weight, very_low_birth_weight | Target deriva de PESO  |
| apgar_baixo          | APGAR5, low_apgar5                     | Target = APGAR5 < 7             |
| permanencia_prolongada | DIAS_PERM, length_of_stay_days       | Target deriva de DIAS_PERM      |
| custo_elevado        | VAL_TOT                                | Target = percentil de VAL_TOT   |
| infeccao_hospitalar  | DIAG_SEC (cols T80-T88)               | Target é o próprio diagnóstico  |
| abandono_tb          | SITUA_ENCE                             | Target deriva de SITUA_ENCE     |
| abandono_hanseniase  | coluna de encerramento                | Target = encerramento ≠ cura    |

Remover com: `df.drop(columns=[...], errors="ignore")`

A lista que os testes travam é `LEAK_BLACKLIST` em `tests/test_invariants.py`: nenhuma coluna dela pode estar em `suggested_features` do desfecho. Coluna nova que vaza o alvo entra lá.

---

## Sentinel values DATASUS (antes do pipeline)

Valores que indicam "ignorado" nos dados brutos:

```python
# No pré-processamento (core/data/*.py) — substituir por NaN:
df.replace({"CAMPO": {9: np.nan, 99: np.nan}}, inplace=True)

# No pipeline ML, o SentinelReplacer só entra se o tratamento trouxer null_sentinels
# (a tela marca 9 e 99 para todas as colunas; a API não tem lista padrão).
# A norma é sentinela por variável (CP3 da ml-checkpoints), e o pré-processador
# trata os casos específicos (ex: IDADE no SIM)
```

---

## Tratamento da coluna IDADE no SIM (sistema de óbitos)

O SIM codifica IDADE com prefixo numérico:

```
4XX = anos  (ex: 465 = 65 anos)
3XX = meses (ex: 306 = 6 meses)
2XX = dias  (ex: 210 = 10 dias)
1XX = horas (ex: 112 = 12 horas)
```

Decodificação: `core/data/sim.py::_decode_idade(series)`
