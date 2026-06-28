"""Óbito por tuberculose — SINAN-TB."""

from __future__ import annotations

import pandas as pd

from core.outcomes.base import OutcomeConfig
from core.data import sinan as sinan_prep
from core.features import engineering as eng


class ObitoTB(OutcomeConfig):
    def __init__(self):
        super().__init__(
            key="obito_tb",
            name="Óbito em Tuberculose",
            description=(
                "Prediz o risco de óbito (por TB ou outra causa) ao encerramento do "
                "caso de tuberculose (SITUA_ENCE = 2). Utiliza dados do SINAN-TB. "
                "Features incluem forma clínica, baciloscopia, cultura, co-infecção HIV, "
                "supervisão do tratamento (DOT) e características sociodemográficas, "
                "todas disponíveis na notificação inicial."
            ),
            data_sources=["SINAN_TB"],
            observation_window_days=0,
            prediction_window_days=180,
            requires_linkage=False,
            icon="sentiment_very_dissatisfied",
            estimated_download_min=8,
            suggested_features=[
                "idade_anos", "CS_SEXO", "CS_RACA", "CS_ESCOL_N",
                "FORMA", "BACILOSC_E", "CULTURA_ES",
                "hiv_pos", "AGRAVAIDS",
                "dot", "TRATAMENTO", "RAIOX_TORA",
                "age_group",
            ],
            target_col="obito_tb",
        )

    def build_cohort(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = sinan_prep.preprocess(data["SINAN_TB"])
        # Apenas casos encerrados (desfecho definitivo conhecido)
        df = sinan_prep.filter_closed_cases(df)
        # Remove colunas-fonte do alvo para evitar vazamento
        df = df.drop(columns=["SITUA_ENCE", "abandono", "cura"], errors="ignore")
        return df

    def build_features(self, cohort: pd.DataFrame) -> pd.DataFrame:
        df = cohort.copy()

        if "idade_anos" in df.columns:
            df["age_group"] = eng.age_group(df["idade_anos"])

        for col in ["CS_SEXO", "CS_RACA", "CS_ESCOL_N", "FORMA",
                    "BACILOSC_E", "CULTURA_ES", "AGRAVAIDS",
                    "TRATAMENTO", "RAIOX_TORA"]:
            if col in df.columns:
                df[col] = pd.Categorical(df[col].astype(str)).codes.astype(float)

        return df

    def get_target(self, cohort: pd.DataFrame) -> pd.Series:
        return cohort[self.target_col].fillna(0).astype(int)
