"""Parto cesariana (PARTO == 2) — SINASC."""

from __future__ import annotations

import pandas as pd

from core.outcomes.base import OutcomeConfig
from core.data import sinasc as sinasc_prep
from core.features import engineering as eng


class Cesarea(OutcomeConfig):
    def __init__(self):
        super().__init__(
            key="cesarea",
            name="Parto Cesariana",
            description=(
                "Prediz a probabilidade de o parto ser cesáreo a partir de dados "
                "do SINASC disponíveis no momento do nascimento: características "
                "maternas (idade, escolaridade, raça/cor, paridade), número de "
                "consultas pré-natais, idade gestacional, tipo de gravidez "
                "(única/múltipla) e apresentação fetal. O Brasil tem uma das "
                "maiores taxas de cesárea do mundo (~55%), bem acima do limite "
                "recomendado pela OMS."
            ),
            data_sources=["SINASC"],
            observation_window_days=0,
            prediction_window_days=0,
            requires_linkage=False,
            icon="pregnant_woman",
            estimated_download_min=5,
            suggested_features=[
                "IDADEMAE", "ESCMAE", "RACACORMAE", "ESTCIVMAE",
                "CONSULTAS", "GESTACAO", "GRAVIDEZ", "TPAPRESENT",
                "QTDFILVIVO", "QTDFILMORT", "SEXO", "STTRABPART",
                "age_group_mae", "preterm",
            ],
            target_col="cesarea",
        )

    def build_cohort(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = sinasc_prep.preprocess(data["SINASC"])
        parto = (
            pd.to_numeric(df["PARTO"], errors="coerce")
            if "PARTO" in df.columns
            else pd.Series(dtype=float)
        )
        df["cesarea"] = (parto == 2).astype(int)
        # Remove the source column and any field that leaks delivery mode.
        # PARTO is the target source. STCESPARTO (planned cesarean flag) is only
        # defined for cesarean births and is a near-perfect proxy of the target.
        df = df.drop(columns=["PARTO", "STCESPARTO"], errors="ignore")
        return df

    def build_features(self, cohort: pd.DataFrame) -> pd.DataFrame:
        df = cohort.copy()

        if "IDADEMAE" in df.columns:
            df["IDADEMAE"] = pd.to_numeric(df["IDADEMAE"], errors="coerce")
            df["age_group_mae"] = eng.age_group(df["IDADEMAE"])

        if "CONSULTAS" in df.columns:
            df["CONSULTAS"] = pd.to_numeric(df["CONSULTAS"], errors="coerce")

        for col in ["QTDFILVIVO", "QTDFILMORT"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").clip(0, 20)

        for col in ["GESTACAO", "GRAVIDEZ", "TPAPRESENT", "SEXO",
                    "ESCMAE", "RACACORMAE", "ESTCIVMAE", "STTRABPART"]:
            if col in df.columns:
                df[col] = pd.Categorical(df[col].astype(str)).codes.astype(float)

        return df

    def get_target(self, cohort: pd.DataFrame) -> pd.Series:
        return cohort[self.target_col].fillna(0).astype(int)
