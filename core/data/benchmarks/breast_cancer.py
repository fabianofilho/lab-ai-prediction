"""Breast Cancer Wisconsin (Diagnostic) — UCI, embarcado no scikit-learn.

Sem rede: usa sklearn.datasets.load_breast_cancer. 569 amostras, 30 features
de imagem de aspirado por agulha fina; alvo = maligno (1) vs benigno (0).
"""
from __future__ import annotations

import pandas as pd

# Rótulos amigáveis para as 30 features (média/erro/pior de 10 medidas).
DICT_META = {
    "target": {"label": "Maligno", "desc": "1 = tumor maligno, 0 = benigno", "type": "Derivada"},
}


def load() -> pd.DataFrame:
    """Devolve o dataset como DataFrame com coluna alvo binária 'target'.

    Convenção do scikit-learn: target 0 = malignant, 1 = benign. Aqui o alvo é
    invertido para 1 = maligno (o evento de interesse clínico).
    """
    from sklearn.datasets import load_breast_cancer

    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.copy()
    # sklearn: 0=malignant, 1=benign → inverte para 1=maligno
    df["target"] = (df["target"] == 0).astype(int)
    # nomes de coluna sem espaços para o pipeline
    df.columns = [c.replace(" ", "_") for c in df.columns]
    return df
