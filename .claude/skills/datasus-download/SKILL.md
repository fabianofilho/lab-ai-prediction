# Skill: Download de Dados DATASUS — lab-ai-prediction

Referência para entender, depurar ou estender o sistema de download.

**Arquivo central:** `core/data/downloader.py`

---

## Estratégia em cascata (ordem de tentativa)

Ordem real de `fetch(system, state, year, progress_callback=None, max_rows=None)`:

```
0. Cache parquet local
   RAW_DIR/{system.lower()}_{UF}_{ano}.parquet   (ex.: sinan_tb_SP_2020.parquet)
   (RAW_DIR = "data/raw/" local | "/tmp/datasus_raw" se data/raw não puder ser criado)
   Só lido quando max_rows é None.

1. pySUS, se estiver instalado (_try_pysus)
   Só SIH, SIM, SINASC e SINAN_TB. Não está no requirements.txt (linha comentada),
   então onde não foi instalado à parte, como no deploy, o passo falha e a cascata segue.

2. HTTP mirror (DigitalOcean CDN), constante HTTP_MIRROR
   https://datasus-ftp-mirror.nyc3.cdn.digitaloceanspaces.com/{path}

3. FTP oficial, constante FTP_HOST
   ftp.datasus.gov.br (modo passivo)

4. Upload manual
   fetch levanta ManualUploadRequired; a tela mostra o file_uploader e o CSV
   exportado do TABNET entra por load_from_csv(csv_bytes, system, state, year)
```

Com `max_rows`, o resultado é parcial e não é gravado no cache.

---

## Padrões de arquivo por sistema

| Sistema    | Padrão de arquivo             | Frequência | Filtro   |
|------------|-------------------------------|------------|----------|
| SIH        | `RD{UF}{aa}{mm}.dbc`          | Mensal     | Por UF   |
| SIM        | `DO{UF}{aaaa}.dbc`            | Anual      | Por UF   |
| SINASC     | `DN{UF}{aaaa}.dbc`            | Anual      | Por UF   |
| SINAN_TB   | `TUBEBR{aa}.dbc`              | Anual      | Nacional (filtrado por SG_UF_NOT) |
| SINAN_HANS | `HANSBR{aa}.dbc`              | Anual      | Nacional |
| SINAN_DENG | `DENGBR{aa}.dbc`              | Anual      | Nacional |
| SINAN_CHIK | `CHIKBR{aa}.dbc`              | Anual      | Nacional |
| SINAN_AIDS | `AIDABR{aa}.dbc`              | Anual      | Nacional |
| SINAN_SIFA | `SIFABR{aa}.dbc`              | Anual      | Nacional |
| SINAN_VIOL | `VIOLBR{aa}.dbc`              | Anual      | Nacional |
| SINAN_IEXO | `IEXOBR{aa}.dbc`              | Anual      | Nacional |

Onde: `{UF}` = sigla (ex: SP, RJ), `{aa}` = ano 2 dígitos, `{aaaa}` = ano 4 dígitos, `{mm}` = mês 2 dígitos.

Os padrões e pastas estão em `FTP_CONFIG`. O SIH baixa os 12 meses do ano e concatena. Os arquivos SINAN são nacionais e são filtrados pela UF (`uf_col`, hoje `SG_UF_NOT` em todos) durante a leitura, em `_dbc_to_df`.

---

## Descompressão DBC

Biblioteca: `datasus-dbc` (wheel pré-compilado, sem compilador C) para descomprimir e `dbfread` para ler o DBF. O `datasus-dbc` 0.1.3 expõe só `decompress` e `decompress_bytes`; não existem `read_dbc` nem `dbc_to_parquet`.

```python
from datasus_dbc import decompress_bytes

dbf_bytes = decompress_bytes(dbc_bytes)   # DBC em bytes -> DBF em bytes
# o app grava o DBF num arquivo temporário e lê com dbfread.DBF(..., encoding="latin-1"),
# com o parser tolerante de _make_tolerant_parser (data e número inválidos viram None)
```

O caminho completo, com o filtro de UF e o `max_rows`, é `_dbc_to_df(dbc_bytes, max_rows=None, uf_col=None, uf_code=None)` em `core/data/downloader.py`.

Não acrescente `pyreaddbc`: ele exige compilador C e falha no Streamlit Cloud. O pySUS depende dele, por isso está comentado no `requirements.txt`; o passo 1 da cascata só roda onde ele já estiver instalado.

---

## Códigos dos estados (27 UFs)

```python
STATES = [
    "AC","AL","AM","AP","BA","CE","DF","ES","GO",
    "MA","MG","MS","MT","PA","PB","PE","PI","PR",
    "RJ","RN","RO","RR","RS","SC","SE","SP","TO"
]
```

Mapeamento IBGE (código numérico → sigla):
```python
IBGE_TO_UF = {
    11:"RO", 12:"AC", 13:"AM", 14:"RR", 15:"PA", 16:"AP", 17:"TO",
    21:"MA", 22:"PI", 23:"CE", 24:"RN", 25:"PB", 26:"PE", 27:"AL",
    28:"SE", 29:"BA", 31:"MG", 32:"ES", 33:"RJ", 35:"SP", 41:"PR",
    42:"SC", 43:"RS", 50:"MS", 51:"MT", 52:"GO", 53:"DF"
}
```

---

## Sentinel values DATASUS

Valores que significam "ignorado/desconhecido" e devem ser tratados como `NaN`:

| Valor | Significado                        |
|-------|------------------------------------|
| `9`   | Ignorado (campos com 1 dígito)     |
| `99`  | Ignorado (campos com 2 dígitos)    |
| `999` | Ignorado (campos com 3 dígitos)    |

O download não troca sentinela nenhuma. No pipeline, o `SentinelReplacer` só entra quando o tratamento traz `null_sentinels`: na tela a seleção vem com 9 e 99, aplicados a todas as colunas, e na API a lista padrão é vazia (ver skill `datasus-pipeline`). A norma do lab é sentinela por variável, a partir do dicionário (CP3 da `ml-checkpoints` do labskills). Nas funções de pré-processamento (`core/data/*.py`), outros valores especiais são tratados caso a caso (ex: IDADE no SIM usa prefixo para indicar unidade).

---

## Lógica de cache parquet

```python
def _cache_path(system, state, year):
    return RAW_DIR / f"{system.lower()}_{state.upper()}_{year}.parquet"

cache = _cache_path(system, state, year)
if cache.exists() and max_rows is None:
    return pd.read_parquet(cache)
# ... pySUS, mirror, FTP ...
return df if max_rows else _save(df, system, state, year)   # _save grava o parquet
```

Para forçar re-download: deletar o `.parquet` correspondente em `RAW_DIR`. `cached_files()` lista o que está no cache.

---

## Pré-processadores por sistema

| Sistema  | Módulo                  | Função principal   | Notas                              |
|----------|-------------------------|--------------------|-------------------------------------|
| SIH      | `core.data.sih`         | `preprocess(df)`   | Mantém `KEEP_COLS`, calcula `length_of_stay_days`, `is_death`, `used_icu` e `diag_chapter` |
| SIM      | `core.data.sim`         | `preprocess(df)`   | Decodifica campo IDADE (prefixo)    |
| SINASC   | `core.data.sinasc`      | `preprocess(df)`   | Deriva preterm, low_birth_weight   |
| SINAN_TB | `core.data.sinan`       | `preprocess(df)`   | Decodifica NU_IDADE_N em `idade_anos` e deriva `abandono`, `cura` e `obito_tb` (NaN na censura); `filter_closed_cases` e `drop_censored` ficam à parte |
| Outros   | `core.data.sinan_*`     | `preprocess(df)`   | Específico por doença               |

---

## Adicionando nova fonte de dados

1. Adicionar entrada em `FTP_CONFIG` em `core/data/downloader.py` com padrão de arquivo e caminho FTP
2. Criar `core/data/{nova_fonte}.py` com função `preprocess(df) -> pd.DataFrame`
3. Referenciar a nova fonte em `data_sources` do OutcomeConfig correspondente
