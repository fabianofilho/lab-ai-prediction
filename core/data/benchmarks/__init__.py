"""Catálogo de datasets externos (benchmarks) usáveis no mesmo pipeline tabular do app.

Cada benchmark expõe um loader que devolve `(df, target_col, features, dict_meta)`
no mesmo contrato consumido por `pages/analise.py` no modo DIY.

Status:
    "ok"       — loader pronto, baixa e roda
    "external" — só catálogo, requer credencial (PhysioNet, UK Biobank, etc)
    "dev"      — em desenvolvimento
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd


@dataclass
class Benchmark:
    key: str
    name: str
    source: str
    icon: str
    est_min: int
    status: str
    note: str
    url: str
    target_col: str = "target"
    loader: Optional[Callable[[], pd.DataFrame]] = None
    dict_meta: Optional[dict] = None


from .pima import load as _load_pima, DICT_META as _PIMA_DICT
from .heart import load as _load_heart, DICT_META as _HEART_DICT
from .heart_failure import load as _load_hf, DICT_META as _HF_DICT
from .breast_cancer import load as _load_bc, DICT_META as _BC_DICT
from .ilpd import load as _load_ilpd, DICT_META as _ILPD_DICT
from .mammographic import load as _load_mammo, DICT_META as _MAMMO_DICT
from .parkinsons import load as _load_park, DICT_META as _PARK_DICT
from .ckd import load as _load_ckd, DICT_META as _CKD_DICT


BENCHMARK_GROUPS: dict[str, list[Benchmark]] = {
    "Tabulares clássicos (UCI)": [
        Benchmark(
            key="bench_pima",
            name="Pima Indians Diabetes",
            source="UCI",
            icon="bloodtype",
            est_min=1,
            status="ok",
            note="768 mulheres Pima, 8 features clínicas, diabetes em 5 anos. Clássico tabular binário.",
            url="https://archive.ics.uci.edu/dataset/34/diabetes",
            target_col="target",
            loader=_load_pima,
            dict_meta=_PIMA_DICT,
        ),
        Benchmark(
            key="bench_heart",
            name="Heart Disease (Cleveland)",
            source="UCI",
            icon="favorite",
            est_min=1,
            status="ok",
            note="303 pacientes, 13 features clínicas, doença coronariana. Multi-classe colapsado em binário.",
            url="https://archive.ics.uci.edu/dataset/45/heart+disease",
            target_col="target",
            loader=_load_heart,
            dict_meta=_HEART_DICT,
        ),
        Benchmark(
            key="bench_heart_failure",
            name="Heart Failure Clinical Records",
            source="UCI",
            icon="ecg_heart",
            est_min=1,
            status="ok",
            note="299 pacientes com insuficiência cardíaca, 11 features clínicas, óbito no seguimento. Coluna 'time' removida (vazamento).",
            url="https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records",
            target_col="target",
            loader=_load_hf,
            dict_meta=_HF_DICT,
        ),
        Benchmark(
            key="bench_breast_cancer",
            name="Breast Cancer Wisconsin",
            source="UCI / sklearn",
            icon="oncology",
            est_min=1,
            status="ok",
            note="569 amostras, 30 features de aspirado por agulha fina, maligno vs benigno. Embarcado no scikit-learn (sem download).",
            url="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic",
            target_col="target",
            loader=_load_bc,
            dict_meta=_BC_DICT,
        ),
        Benchmark(
            key="bench_ilpd",
            name="Indian Liver Patient (ILPD)",
            source="UCI",
            icon="gastroenterology",
            est_min=1,
            status="ok",
            note="583 pacientes, 10 features bioquímicas (bilirrubina, enzimas hepáticas, albumina), doença hepática sim/não.",
            url="https://archive.ics.uci.edu/dataset/225/ilpd+indian+liver+patient+dataset",
            target_col="target",
            loader=_load_ilpd,
            dict_meta=_ILPD_DICT,
        ),
        Benchmark(
            key="bench_mammographic",
            name="Mammographic Mass",
            source="UCI",
            icon="mammography",
            est_min=1,
            status="ok",
            note="961 casos, features de forma/margem/densidade + idade, massa maligna vs benigna. Coluna BI-RADS removida (avaliação do laudo, vazamento).",
            url="https://archive.ics.uci.edu/dataset/161/mammographic+mass",
            target_col="target",
            loader=_load_mammo,
            dict_meta=_MAMMO_DICT,
        ),
        Benchmark(
            key="bench_parkinsons",
            name="Parkinsons (voz)",
            source="UCI",
            icon="neurology",
            est_min=1,
            status="ok",
            note="195 gravações de voz, 22 features acústicas (jitter, shimmer, HNR), saudável vs Parkinson. Coluna 'name' removida (identificador).",
            url="https://archive.ics.uci.edu/dataset/174/parkinsons",
            target_col="target",
            loader=_load_park,
            dict_meta=_PARK_DICT,
        ),
        Benchmark(
            key="bench_ckd",
            name="Chronic Kidney Disease",
            source="UCI",
            icon="nephrology",
            est_min=1,
            status="ok",
            note="399 pacientes, 24 features clínicas/laboratoriais, doença renal crônica sim/não. CSV vendorizado do ARFF oficial (1 linha corrompida descartada).",
            url="https://archive.ics.uci.edu/dataset/336/chronic+kidney+disease",
            target_col="target",
            loader=_load_ckd,
            dict_meta=_CKD_DICT,
        ),
    ],
    "Cuidado intensivo (PhysioNet)": [
        Benchmark(
            key="bench_mimic_iv_demo",
            name="MIMIC-IV Demo",
            source="PhysioNet",
            icon="monitor_heart",
            est_min=15,
            status="external",
            note="100 pacientes de UTI desidentificados. Múltiplas tabelas, requer ETL próprio antes do pipeline tabular.",
            url="https://physionet.org/content/mimic-iv-demo/",
        ),
        Benchmark(
            key="bench_eicu_demo",
            name="eICU-CRD Demo",
            source="PhysioNet",
            icon="bed",
            est_min=20,
            status="external",
            note="Dataset multi-centro de UTIs dos EUA. Requer credencial PhysioNet (data use agreement).",
            url="https://eicu-crd.mit.edu/",
        ),
    ],
    "Sinais e Bioinformática": [
        Benchmark(
            key="bench_ptbxl",
            name="PTB-XL ECG",
            source="PhysioNet",
            icon="ecg_heart",
            est_min=30,
            status="external",
            note="21k ECGs de 12 derivações. Features tabulares disponíveis em metadata.csv, sinais em WFDB.",
            url="https://physionet.org/content/ptb-xl/",
        ),
    ],
}


BENCHMARKS: dict[str, Benchmark] = {
    b.key: b for group in BENCHMARK_GROUPS.values() for b in group
}


__all__ = ["Benchmark", "BENCHMARKS", "BENCHMARK_GROUPS"]
