---
name: datasus-pipeline
description: Executa pipeline completo de dados DataSUS. Download, preprocessamento, feature engineering, treino e avaliacao.
model: sonnet
tools: [Bash, Read, Write, Glob, Grep]
---

# DataSUS Pipeline

Agente autonomo para executar pipeline ML com dados do DataSUS.

## Comportamento

### 1. Verificar ambiente

- Verificar se lab-ai-prediction esta clonado
- Checar dependencias (pandas, scikit-learn, xgboost, etc.)
- Verificar dados disponiveis em data/raw/

### 2. Download (se necessario)

Usar `fetch(sistema, UF, ano)` de `core/data/downloader.py`, que faz a cascata descrita na skill `datasus-download` (cache em data/raw/, pySUS se instalado, mirror HTTP da DigitalOcean, FTP do DataSUS).

### 3. Preprocessar

- Montar a coorte com `CohortBuilder(OUTCOMES[chave]).build(raw)` e tirar `X, y` com `.get_Xy(cohort)`
- Aplicar data_dict para nomes canonicos
- Feature engineering conforme datasus-features
- Missing, sentinelas, encoding e escala pelo `treatment` descrito na skill `datasus-pipeline`, com as decisoes do CP3 e do CP4 da skill `ml-checkpoints` do labskills

### 4. Treinar

Seguir a API da skill `datasus-pipeline` e as decisoes da `ml-checkpoints`, que e a norma de metodo do lab:

- Separar com `split_train_test` antes de qualquer busca de hiperparametros; `train_cv` para validacao cruzada
- Modelos: as chaves de `ALGORITHMS`; candidatos e baseline pelo CP6
- Balanceamento: `balancing="none"` por padrao; qualquer outro valor so com a calibracao medida com e sem ele (CP5 e CP9)
- Hyperparameter tuning com Optuna (se configurado), so na particao de treino

### 5. Avaliar

- Seguir a skill `datasus-avaliacao` e o CP8 a CP10 da `ml-checkpoints`: metrica principal escolhida antes de rodar, calibracao e decision curve
- Fairness por subgrupo (se aplicavel)
- Comparar com a baseline

### 6. Retornar

Metricas principais, caminho do modelo salvo, graficos gerados.
