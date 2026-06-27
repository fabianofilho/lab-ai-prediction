---
title: Lab AI Prediction
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.32.0
app_file: app.py
pinned: false
license: mit
---

<p align="center">
  <img src="favicon.png" alt="Lab AI Prediction" width="80" />
</p>

<h1 align="center">Lab AI Prediction</h1>

<p align="center">
  Laboratório aberto de modelagem preditiva em saúde. Rode o mesmo pipeline em microdados brasileiros do DataSUS, na sua própria base ou em benchmarks internacionais.
</p>

<p align="center">
  <a href="https://lab-ai-prediction.streamlit.app"><img src="https://img.shields.io/badge/Acessar%20Plataforma-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Acessar Plataforma" /></a>
</p>

<p align="center">
  <a href="https://streamlit.io"><img src="https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit" alt="Streamlit" /></a>
  <a href="https://python.org"><img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License: MIT" /></a>
</p>

---

## O que é

Plataforma sem código para epidemiologistas, residentes e cientistas de dados rodarem o pipeline completo de ML em saúde: coorte, features, treino, calibração, SHAP, relatório. A home oferece três entradas independentes para a mesma esteira de modelagem:

| Frente | Quando usar |
|---|---|
| **DATASUS** | Quer um dos 16 desfechos prontos sobre microdados brasileiros (SIH, SIM, SINASC, SINAN) com download automático |
| **DIY** | Tem a sua própria base tabular em CSV ou Parquet e quer rodar o pipeline nela |
| **BENCHMARKS** | Quer comparar performance em datasets clássicos abertos (UCI, PhysioNet) curados do [awesome-health-datasets](https://github.com/fabianofilho/awesome-health-datasets) |

---

## DATASUS — 20 desfechos

Download automático com cache local. Todos os desfechos têm pipeline pronto
ponta a ponta. Os que dependem de pareamento entre registros usam linkage
determinístico ou probabilístico por quase-identificadores (o dado público
não traz CNS/CPF). Cada card tem um botão **Metodologia** que abre um painel
lateral explicando como o dado é puxado e como o desfecho é derivado.

### Saúde Materno-Infantil (SINASC, SIM)
| Desfecho | Fonte | Status |
|---|---|---|
| Baixo Peso ao Nascer | SINASC | ok |
| Prematuridade | SINASC | ok |
| Apgar Baixo no 5º Minuto | SINASC | ok |
| Parto Cesariana | SINASC | ok |
| Anomalia Congênita | SINASC | ok (raro ~0,9%) |
| Mortalidade Neonatal | SINASC + SIM | ok (linkage DTNASC+SEXO+PESO) |

### Internação Hospitalar (SIH, SIM)
| Desfecho | Fonte | Status |
|---|---|---|
| Readmissão Hospitalar 30 dias | SIH | ok (linkage nascimento+sexo+município+CEP) |
| Permanência Hospitalar Prolongada | SIH | ok |
| Uso de UTI na Internação | SIH | ok |
| Infecção Hospitalar | SIH | ok |
| Custo Hospitalar Elevado | SIH | ok |
| Mortalidade Hospitalar | SIH + SIM | ok (óbito intra-hospitalar) |

### SINAN
| Desfecho | Fonte | Status |
|---|---|---|
| Abandono de Tratamento TB | SINAN | ok |
| Abandono de Tratamento — Hanseníase | SINAN | ok |
| Risco de Violência Autoprovocada | SINAN | ok |
| Dengue Grave / Sinais de Alarme | SINAN | ok |
| Hospitalização por Chikungunya | SINAN | ok |
| Óbito por AIDS | SINAN | ok |
| Não-Cura de Sífilis Adquirida | SINAN | ok |
| Desfecho Adverso em Intoxicação Exógena | SINAN | ok |

> Os arquivos SINAN são nacionais e filtrados por UF durante a leitura. Dengue é o
> maior arquivo, então o primeiro download é mais lento (cacheado depois).

---

## BENCHMARKS — datasets abertos

Loaders prontos que aterrissam no mesmo pipeline DIY:

| Dataset | Fonte | Loader |
|---|---|---|
| Pima Indians Diabetes | UCI | pronto |
| Heart Disease Cleveland | UCI | pronto |
| MIMIC-IV Demo | PhysioNet | externo (catálogo) |
| eICU-CRD Demo | PhysioNet | externo (catálogo) |
| PTB-XL ECG | PhysioNet | externo (catálogo) |

Adicionar mais benchmarks: implemente um loader em `core/data/benchmarks/<nome>.py` que retorne um `DataFrame` com coluna alvo binária, e registre em `core/data/benchmarks/__init__.py`.

---

## Arquitetura

```
lab-ai-prediction/
├── app.py                        # Home: 3 cards (DATASUS / DIY / BENCHMARKS)
├── pages/
│   ├── datasus.py                # Catálogo de 16 desfechos DATASUS
│   ├── benchmarks.py             # Catálogo de datasets externos
│   ├── upload.py                 # Fluxo DIY (upload CSV/Parquet)
│   ├── analise.py                # Wizard completo de 10 etapas
│   ├── calibracao.py
│   ├── deploy.py
│   └── relatorio.py
├── core/
│   ├── data/
│   │   ├── downloader.py         # Cascata: cache > HTTP mirror > FTP > upload
│   │   ├── sih.py / sim.py / sinasc.py / sinan*.py   # Pré-processadores DATASUS
│   │   ├── linker.py             # Record linkage determinístico + probabilístico
│   │   └── benchmarks/           # Loaders de datasets externos
│   ├── features/cohort.py        # CohortBuilder com janelas temporais
│   ├── models/
│   │   ├── pipeline.py           # train_cv() + Optuna HPO
│   │   └── evaluation.py         # ROC, PR, calibração, SHAP (Plotly)
│   └── outcomes/                 # Desfechos DATASUS implementados (OutcomeConfig)
```

### Stack
- **Download:** `datasus-dbc` (DBC → DBF sem compilador C) + mirror HTTP DigitalOcean + FTP DataSUS
- **ML:** LightGBM, XGBoost, CatBoost, Logistic Regression, Random Forest, MLP
- **Otimização:** Optuna
- **Explicabilidade:** SHAP com gráficos interativos
- **Validação:** StratifiedKFold(5) + amostragem estratificada
- **Calibração:** Platt Scaling, comparação entre estados/períodos

---

## Pipeline (10 etapas)

```
1.  Desfecho     → escolha o que prever
2.  Dados        → seleção UF/ano + download automático (ou upload)
3.  Features     → distribuição, balanceamento, seleção de variáveis
4.  Tratamento   → encoding/escalonamento por coluna, sentinelas
5.  Modelo       → algoritmo(s), validação, busca de hiperparâmetros
6.  Treinamento  → visualização adaptativa (MLP, boosting, curvas)
7.  Resultados   → ROC/PR, SHAP, métricas clínicas, equidade
8.  Benchmark    → comparação lado a lado de algoritmos
9.  Deploy       → inferência individual com SHAP local
10. Relatório    → exporta o estudo completo
```

---

## Rodar localmente

```bash
git clone https://github.com/fabianofilho/lab-ai-prediction
cd lab-ai-prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## Sistemas DataSUS suportados

| Sistema | Descrição | Cobertura |
|---|---|---|
| **SIH** | Sistema de Informações Hospitalares | 2008–atual, mensal por UF |
| **SIM** | Sistema de Informações sobre Mortalidade | 1996–atual, anual por UF |
| **SINASC** | Sistema de Informações sobre Nascidos Vivos | 1996–atual, anual por UF |
| **SINAN** | Sistema de Informação de Agravos de Notificação | Dengue, TB, Hanseníase, AIDS, Sífilis, Chikungunya, Violência, Intoxicação |

---

## Contribuindo

- **Novo desfecho DATASUS:** subclasse de `OutcomeConfig` em `core/outcomes/` seguindo os exemplos existentes
- **Novo benchmark:** loader em `core/data/benchmarks/<nome>.py` + registro no `BENCHMARK_GROUPS`

---

## Licença

MIT — use livremente para pesquisa e ensino.
