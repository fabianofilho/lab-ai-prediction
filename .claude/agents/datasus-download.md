---
name: datasus-download
description: Baixa dados do DataSUS pelo fetch do app (cache, mirror HTTP, FTP). Gerencia cache e fallback automatico.
model: haiku
tools: [Bash, Read, Write, Glob, WebFetch]
---

# DataSUS Download

Agente autonomo para download de dados do DataSUS.

## Comportamento

### 1. Identificar dados

Do prompt, determinar:

- Sistema: SINASC, SIH, SIM, SINAN, CNES, SIA
- UF(s) e ano(s)
- Diretorio destino (padrao: data/raw/)

### 2. Verificar cache

O cache do app e um parquet por sistema, UF e ano, com o sistema em minusculas:

```bash
ls data/raw/<sistema>_<UF>_<ANO>.parquet 2>/dev/null   # ex.: data/raw/sinan_tb_SP_2020.parquet
```

Se ja existe, reportar e perguntar se quer re-baixar.

### 3. Baixar

Use o `fetch` do app, que ja faz a cascata descrita na skill `datasus-download` (cache, pySUS se instalado, mirror HTTP da DigitalOcean, FTP do DataSUS) e grava o parquet:

```bash
python3 -c "from core.data.downloader import fetch; df = fetch('<SISTEMA>', '<UF>', <ANO>); print(df.shape)"
```

Se o `fetch` levantar `ManualUploadRequired`, reportar: o arquivo precisa vir do TABNET e entrar por `load_from_csv`.

### 4. Converter

O `fetch` ja devolve DataFrame: descomprime com `datasus_dbc.decompress_bytes` e le o DBF com `dbfread` (`_dbc_to_df`). Nao usar `pyreaddbc` nem `tabcmd`.

### 5. Retornar

Arquivos baixados, tamanhos, periodo coberto.
