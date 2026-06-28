"""Uso de UTI na internação — SIH."""

from __future__ import annotations

import pandas as pd

from core.outcomes.base import OutcomeConfig
from core.data import sih as sih_prep
from core.features import engineering as eng

# Colunas que vazam o alvo (derivadas direta ou indiretamente do uso de UTI)
# VAL_TOT entra aqui porque o valor total da AIH SOMA o VAL_UTI: custo alto
# implica UTI usada (vazamento matemático, corr ~0.71 com o alvo).
_LEAKY_COLS = [
    "UTI_MES_TO", "UTI_INT_TO", "VAL_UTI", "used_icu", "MARCA_UTI",
    "UTI_MES_IN", "UTI_MES_AN", "UTI_MES_AL",
    "UTI_INT_IN", "UTI_INT_AN", "UTI_INT_AL",
    "VAL_UCI", "MARCA_UCI", "VAL_UTI_FED", "VAL_TOT",
]


class UsoUTI(OutcomeConfig):
    def __init__(self):
        super().__init__(
            key="uso_uti",
            name="Uso de UTI na Internação",
            description=(
                "Prediz a probabilidade de uma internação utilizar UTI, "
                "auxiliando no dimensionamento de leitos críticos e no "
                "planejamento de recursos hospitalares. Utiliza características "
                "de admissão do SIH-RD. O alvo é derivado de UTI_MES_TO "
                "(dias-UTI no mês > 0)."
            ),
            data_sources=["SIH"],
            observation_window_days=0,
            prediction_window_days=0,
            requires_linkage=False,
            icon="vital_signs",
            estimated_download_min=10,
            suggested_features=[
                "IDADE", "SEXO", "diag_chapter", "diag_block",
                "length_of_stay_days", "CAR_INT", "RACA_COR",
                "PROC_REA", "age_group",
            ],
            target_col="uso_uti",
        )

    def build_cohort(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = sih_prep.preprocess(data["SIH"])

        # Alvo: a internação usou UTI (used_icu já criado pelo preprocessor
        # como UTI_MES_TO > 0). Fallback para MARCA_UTI se necessário.
        if "used_icu" in df.columns:
            df["uso_uti"] = df["used_icu"].fillna(0).astype(int)
        elif "UTI_MES_TO" in df.columns:
            df["uso_uti"] = (
                pd.to_numeric(df["UTI_MES_TO"], errors="coerce").fillna(0) > 0
            ).astype(int)
        elif "MARCA_UTI" in df.columns:
            mu = df["MARCA_UTI"].astype(str).str.strip()
            df["uso_uti"] = (~mu.isin(["00", "", "nan", "None"])).astype(int)
        else:
            df["uso_uti"] = 0

        # Anti-leakage: remove qualquer coluna que revele o alvo.
        df = df.drop(columns=[c for c in _LEAKY_COLS if c in df.columns])

        return df

    def build_features(self, cohort: pd.DataFrame) -> pd.DataFrame:
        df = cohort.copy()

        if "DIAG_PRINC" in df.columns:
            df["diag_chapter"] = eng.icd10_chapter(df["DIAG_PRINC"])
            df["diag_block"] = eng.icd10_block(df["DIAG_PRINC"])

        if "IDADE" in df.columns:
            df["IDADE"] = pd.to_numeric(df["IDADE"], errors="coerce")
            df["age_group"] = eng.age_group(df["IDADE"])

        if "length_of_stay_days" in df.columns:
            df["length_of_stay_days"] = pd.to_numeric(
                df["length_of_stay_days"], errors="coerce"
            )

        for col in ["SEXO", "RACA_COR", "CAR_INT"]:
            if col in df.columns:
                df[col] = pd.Categorical(df[col].astype(str)).codes.astype(float)

        if "PROC_REA" in df.columns:
            df["PROC_REA"] = pd.Categorical(df["PROC_REA"].astype(str)).codes.astype(float)

        df = eng.clip_outliers(df, "VAL_TOT")

        return df

    def get_target(self, cohort: pd.DataFrame) -> pd.Series:
        return cohort[self.target_col].fillna(0).astype(int)
