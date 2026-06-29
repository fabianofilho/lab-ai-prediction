---
name: benchlab-dataset
description: Adiciona um novo dataset ao BenchLab (comparativo de algoritmos x datasets) do lab-ai-prediction. Cobre datasets internacionais (UCI/sklearn, download ou vendorizado) e brasileiros (DATASUS, sample materializado dos desfechos). Cria o loader, registra no catálogo, e valida via run_benchlab. Triggers on /benchlab-dataset, "adiciona um dataset no benchlab", "novo dataset no benchmark", "inclui dataset no benchlab", "materializa dataset brasileiro pro benchlab".
---

# Skill: Adicionar Dataset ao BenchLab — lab-ai-prediction

Use sempre que precisar incluir um novo dataset no BenchLab (`pages/benchlab.py`),
o comparativo global de algoritmos x datasets.

O BenchLab **incorpora automaticamente** todo `Benchmark` com `status="ok"` e
`loader is not None`. Não é preciso mexer em `pages/benchlab.py` para um dataset
internacional novo; basta criar o loader e registrar no catálogo.

---

## Arquitetura

```
core/data/benchmarks/
├── __init__.py        ← dataclass Benchmark, BENCHMARK_GROUPS, BENCHMARKS, helper _br()
├── {dataset}.py       ← loader internacional: load() + DICT_META
├── br_datasus.py      ← loader brasileiro: loader_for(key) lê data/br/<key>.parquet
└── data/
    ├── ckd.csv        ← exemplo de CSV vendorizado (internacional)
    └── br/            ← samples DATASUS materializados + manifest.json
core/benchlab/runner.py  ← run_benchlab(): chama bench.loader(), usa bench.target_col
```

**Contrato do loader:** função sem argumentos que devolve um `pd.DataFrame` com a
coluna alvo `target` (0/1) + as features. NaN e categóricas string são tratados
pelo pré-processamento do pipeline (`core/models/pipeline.py`: SimpleImputer +
OneHotEncoder), então o loader não precisa imputar nem encodar.

---

## Fluxo A — Dataset internacional (UCI / sklearn)

### 1. Criar `core/data/benchmarks/{dataset}.py`

```python
"""Nome do dataset — fonte. CSV público, sem credencial."""
from __future__ import annotations
import pandas as pd

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/.../arquivo.data"

DICT_META = {  # rótulos amigáveis (usados pela página de Análise, não pelo BenchLab)
    "feat_a": {"label": "Rótulo", "desc": "unidade", "type": "Numérica"},
    "target": {"label": "Desfecho", "desc": "1=evento, 0=não", "type": "Derivada"},
}

def load() -> pd.DataFrame:
    df = pd.read_csv(URL, header=None, names=[...], na_values="?")
    df["target"] = (df["coluna_alvo"] >= 1).astype(int)  # binariza
    df = df.drop(columns=["coluna_alvo"])
    return df
```

Variações: sklearn embarcado (`from sklearn.datasets import load_*`, sem rede) ou
CSV vendorizado quando a fonte só publica formato chato (ex.: `.rar`/ARFF do CKD,
parseado uma vez e salvo em `data/<dataset>.csv`).

### 2. Registrar em `core/data/benchmarks/__init__.py`

Import junto dos demais (após a dataclass, mesmo padrão do arquivo):
```python
from .{dataset} import load as _load_x, DICT_META as _X_DICT
```
E um `Benchmark` no grupo certo de `BENCHMARK_GROUPS` (ex.: "Tabulares clássicos (UCI)"):
```python
Benchmark(
    key="bench_{dataset}", name="Nome", source="UCI", icon="material_symbol",
    est_min=1, status="ok", note="n amostras, descrição. Notas de leakage.",
    url="https://...", target_col="target", loader=_load_x, dict_meta=_X_DICT,
),
```

---

## Fluxo B — Dataset brasileiro (DATASUS)

Datasets brasileiros são **samples materializados** dos desfechos do app
(`core/outcomes`), salvos em `data/br/<key>.parquet`. Vendorizar porque baixar do
DATASUS é lento, é por UF/ano e pode falhar (cai no upload manual). Marcados com
`country="br"` para a página separá-los e oferecer o sorteio.

### 1. Materializar o sample (rodar uma vez)

Requer `datasus-dbc` instalado (`pip install datasus-dbc`). Escolher UF/ano com
volume suficiente e classes presentes. Desfechos single-source (sem linkage) saem
de um único download; vários desfechos compartilham o mesmo arquivo (ex.: cesárea,
baixo peso, prematuridade vêm todos de um SINASC).

```python
import warnings, json; warnings.filterwarnings("ignore")
from pathlib import Path
from core.data.downloader import fetch
from core.outcomes import OUTCOMES
from core.features.cohort import CohortBuilder

UF, YEAR, CAP = "GO", 2023, 6000
OUT = Path("core/data/benchmarks/data/br"); OUT.mkdir(parents=True, exist_ok=True)
POOL = {  # system: (lista de chaves de desfecho, max_rows do download)
    "SINASC":   (["cesarea", "baixo_peso_nascer", "prematuridade"], 60000),
    "SIH":      (["uso_uti", "custo_elevado"], 40000),
    "SINAN_TB": (["obito_tb", "abandono_tb"], None),
}

def strat_sample(df, cap, seed=42):
    if len(df) <= cap: return df.reset_index(drop=True)
    frac = cap / len(df)
    return (df.groupby("target", group_keys=False)
              .apply(lambda g: g.sample(frac=frac, random_state=seed))
              .reset_index(drop=True))

manifest = {}
for system, (keys, mr) in POOL.items():
    raw = fetch(system, UF, YEAR, max_rows=mr)   # max_rows evita OOM e não cacheia
    for key in keys:
        b = CohortBuilder(OUTCOMES[key])
        cohort = b.build({system: raw})
        X, y = b.get_Xy(cohort)
        df = X.copy(); df["target"] = y.astype(int).values
        df = strat_sample(df.dropna(subset=["target"]), CAP)
        df.to_parquet(OUT / f"{key}.parquet", index=False)
        vc = dict(df["target"].value_counts())
        manifest[key] = {"system": system, "uf": UF, "year": YEAR,
                         "n": len(df), "n_features": df.shape[1]-1,
                         "pos": int(vc.get(1,0)), "neg": int(vc.get(0,0))}
(OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
```

Conferir cada sample: `n` razoável (centenas a milhares), ambas as classes
presentes. Se um desfecho ficar minúsculo ou single-class, trocar a UF.

Evitar arquivos SINAN nacionais gigantes (dengue/violência) a menos que necessário;
SINASC + SIH + SINAN_TB já cobrem 3 sistemas. O cache `data/raw/` é gitignored.

### 2. Registrar em `core/data/benchmarks/__init__.py`

O helper `_br()` monta o Benchmark a partir do manifest. Adicionar a chave ao
grupo "Brasil (DATASUS)" de `BENCHMARK_GROUPS`:
```python
_br("uso_uti", "Uso de UTI", "SIH", "bed",
    "Uso de UTI na internação a partir do SIH (internações)."),
```
(`country="br"` já vem do helper.) O loader vem de `br_datasus.loader_for(key)`.

---

## Validação (sempre rodar antes de concluir)

```python
import warnings; warnings.filterwarnings("ignore")
from core.data.benchmarks import BENCHMARKS
from core.benchlab import run_benchlab

# 1. loaders carregam com shape/target corretos
for k in ["bench_..."]:
    b = BENCHMARKS[k]; df = b.loader()
    print(k, df.shape, dict(df["target"].value_counts()))

# 2. end-to-end (mix internacional + brasileiro), zero erros esperado
res = run_benchlab(benchmarks=[BENCHMARKS[k] for k in [...]],
                   algorithm_keys=["lgbm", "logreg"], n_folds=3)
print("ERROS:", res["errors"])
```

Também: `python3 -m py_compile` nos arquivos novos e `ruff check`. O `__init__.py`
acumula warnings E402 (imports após a dataclass) que são pré-existentes/aceitos no
arquivo, não uma regressão.

---

## Anti-leakage (mesmas regras dos desfechos)

Remover do loader colunas que vazam o alvo, com `df.drop(columns=[...], errors="ignore")`:
- BI-RADS no Mammographic (é a avaliação de malignidade do laudo).
- `time` no Heart Failure (seguimento correlaciona com óbito).
- SINASC: não usar `PESO` em baixo peso, `GESTACAO` em prematuridade, `APGAR5` em apgar.
- SINAN: não usar `SITUA_ENCE` quando o target deriva dela.

---

## Referências no código
- Internacional simples: `core/data/benchmarks/heart_failure.py` (download CSV).
- Internacional vendorizado: `core/data/benchmarks/ckd.py` + `data/ckd.csv`.
- Brasileiro: `core/data/benchmarks/br_datasus.py` + `data/br/` + helper `_br()` no `__init__.py`.
- UI do sorteio brasileiro: `pages/benchlab.py` (bloco "Brasil (DATASUS)").
