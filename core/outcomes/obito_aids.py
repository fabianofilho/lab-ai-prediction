"""Óbito por AIDS — SINAN-AIDS."""
from __future__ import annotations
import pandas as pd
from core.outcomes.base import OutcomeConfig
from core.data import rotulo
from core.data import sinan_aids as aids_prep
from core.features import engineering as eng


class ObitoAIDS(OutcomeConfig):
    def __init__(self):
        super().__init__(
            key="obito_aids",
            name="Óbito por AIDS",
            description=(
                "Prediz a probabilidade de óbito por AIDS (EVOLUCAO = 2), contra vivo "
                "(EVOLUCAO = 1), em pacientes recém-notificados. Óbito por outras causas "
                "(EVOLUCAO = 3) é censura e sai da coorte. "
                "Features incluem doenças definidoras de AIDS presentes "
                "no diagnóstico (tuberculose, candidíase, toxoplasmose, etc.), "
                "via de transmissão, critério diagnóstico e características demográficas. "
                "Utiliza SINAN-AIDS Adulto."
            ),
            data_sources=["SINAN_AIDS"],
            observation_window_days=0,
            prediction_window_days=365,
            requires_linkage=False,
            icon="🔴",
            estimated_download_min=5,
            suggested_features=[
                "idade_anos", "CS_SEXO", "CS_RACA", "CS_ESCOL_N",
                "ANT_TUBERC", "ANT_PNEUMO", "ANT_TOXO", "ANT_CANDID",
                "ANT_SARCOM", "ANT_CAQUEX", "ANT_DIARRE", "ANT_FEBRE",
                "ANT_CRIPTO", "ANT_LINFOM",
                "n_doencas_aids", "ANT_DROGA", "ANT_TRASMI",
                "CRITERIO", "age_group",
            ],
            target_col="obito_aids",
        )

    def build_cohort(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = aids_prep.preprocess(data["SINAN_AIDS"])
        df = aids_prep.filter_with_outcome(df)
        # Óbito por outras causas (3) sai da coorte em vez de virar 0
        df = aids_prep.drop_censored(df, self.target_col)
        df = df.drop(columns=["vivo", "EVOLUCAO", "DT_OBITO"], errors="ignore")
        return df

    def build_features(self, cohort: pd.DataFrame) -> pd.DataFrame:
        df = cohort.copy()
        if "idade_anos" in df.columns:
            df["age_group"] = eng.age_group(df["idade_anos"])
            df["idade_anos"] = pd.to_numeric(df["idade_anos"], errors="coerce")
        for col in ["CS_SEXO", "CS_RACA", "CS_ESCOL_N", "CS_GESTANT", "CRITERIO", "DEF_DIAGNO"]:
            if col in df.columns:
                df[col] = pd.Categorical(df[col].astype(str)).codes.astype(float)
        if "n_doencas_aids" in df.columns:
            df["n_doencas_aids"] = pd.to_numeric(df["n_doencas_aids"], errors="coerce")
        df = eng.clip_outliers(df, "idade_anos")
        return df

    def get_target(self, cohort: pd.DataFrame) -> pd.Series:
        # Sem fillna(0): censura não é negativo. build_cohort já exclui esses casos.
        return rotulo.exigir_rotulo(cohort[self.target_col], self.key)
