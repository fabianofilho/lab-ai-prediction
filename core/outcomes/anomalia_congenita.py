"""Anomalia congênita ao nascer — SINASC."""

from __future__ import annotations

import pandas as pd

from core.outcomes.base import OutcomeConfig
from core.data import sinasc as sinasc_prep
from core.features import engineering as eng


class AnomaliaCongenita(OutcomeConfig):
    def __init__(self):
        super().__init__(
            key="anomalia_congenita",
            name="Anomalia Congênita",
            description=(
                "Prediz a probabilidade de um recém-nascido apresentar anomalia ou "
                "malformação congênita identificada ao nascer (campo IDANOMAL do "
                "SINASC). Utiliza apenas dados disponíveis no momento do nascimento: "
                "características maternas, peso, idade gestacional, Apgar, tipo de "
                "gravidez e número de consultas pré-natais. Desfecho raro "
                "(~0,9% dos nascimentos), portanto fortemente desbalanceado."
            ),
            data_sources=["SINASC"],
            observation_window_days=0,
            prediction_window_days=0,
            requires_linkage=False,
            icon="neurology",
            estimated_download_min=5,
            suggested_features=[
                "IDADEMAE", "PESO", "GESTACAO", "GRAVIDEZ", "PARTO",
                "CONSULTAS", "APGAR1", "APGAR5", "SEXO",
                "RACACORMAE", "RACACOR", "ESCMAE", "ESTCIVMAE",
                "TPAPRESENT", "preterm", "low_apgar5", "low_birth_weight",
                "age_group_mae",
            ],
            target_col="anomalia_congenita",
        )

    def build_cohort(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = sinasc_prep.preprocess(data["SINASC"])

        # Target: IDANOMAL == 1 (sim). 2=não, 9=ignorado, vazio -> 0.
        if "IDANOMAL" in df.columns:
            idanomal = df["IDANOMAL"].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
            df["anomalia_congenita"] = (idanomal == "1").astype(int)
        else:
            df["anomalia_congenita"] = 0

        # Remove as colunas-fonte do alvo para evitar vazamento.
        df = df.drop(columns=["IDANOMAL", "CODANOMAL"], errors="ignore")
        return df

    def build_features(self, cohort: pd.DataFrame) -> pd.DataFrame:
        df = cohort.copy()

        if "IDADEMAE" in df.columns:
            df["IDADEMAE"] = pd.to_numeric(df["IDADEMAE"], errors="coerce")
            df["age_group_mae"] = eng.age_group(df["IDADEMAE"])

        for col in ["PESO", "APGAR1", "APGAR5", "CONSULTAS"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in ["GESTACAO", "GRAVIDEZ", "PARTO", "SEXO", "TPAPRESENT",
                    "ESCMAE", "RACACORMAE", "RACACOR", "ESTCIVMAE"]:
            if col in df.columns:
                df[col] = pd.Categorical(df[col].astype(str)).codes.astype(float)

        df = eng.clip_outliers(df, "PESO")

        return df

    def get_target(self, cohort: pd.DataFrame) -> pd.Series:
        return cohort[self.target_col].fillna(0).astype(int)
