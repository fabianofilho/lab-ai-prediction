"""Desfecho adverso em intoxicação exógena — SINAN-IEXO."""
from __future__ import annotations
import pandas as pd
from core.outcomes.base import OutcomeConfig
from core.data import rotulo
from core.data import sinan_iexo as iexo_prep
from core.features import engineering as eng


class IntoxicacaoGrave(OutcomeConfig):
    def __init__(self):
        super().__init__(
            key="intoxicacao_grave",
            name="Desfecho Adverso em Intoxicação Exógena",
            description=(
                "Prediz a probabilidade de um caso de intoxicação exógena resultar em "
                "desfecho adverso: óbito por intoxicação (EVOLUCAO = 3) ou cura com "
                "sequela (EVOLUCAO = 2), contra cura sem sequela (EVOLUCAO = 1). Óbito "
                "por outra causa (4) e perda de seguimento (5) são censura; ignorado (9) "
                "e em branco também saem da coorte. "
                "Features incluem agente tóxico, circunstância (acidental vs. tentativa de suicídio), "
                "via de exposição, hospitalização e características demográficas. "
                "Utiliza SINAN-Intoxicação Exógena."
            ),
            data_sources=["SINAN_IEXO"],
            observation_window_days=0,
            prediction_window_days=30,
            requires_linkage=False,
            icon="☠️",
            estimated_download_min=8,
            suggested_features=[
                "idade_anos", "CS_SEXO", "CS_RACA", "CS_ESCOL_N",
                "AGENTE_TOX", "AGENTE_1", "CIRCUNSTAN",
                "VIA_1", "TPATENDE", "hospitalizado",
                "tentativa_suicidio", "SIT_TRAB", "CS_GESTANT",
                "age_group",
            ],
            target_col="desfecho_adverso",
        )

    def build_cohort(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = iexo_prep.preprocess(data["SINAN_IEXO"])
        df = iexo_prep.filter_confirmed(df)
        # Só casos com desfecho conhecido: censura (4, 5), ignorado e em branco
        # saem com contagem, em vez de virar 0
        df = iexo_prep.drop_censored(df, self.target_col)
        df = df.drop(columns=["obito", "sequela", "EVOLUCAO", "DT_OBITO"], errors="ignore")
        return df

    def build_features(self, cohort: pd.DataFrame) -> pd.DataFrame:
        df = cohort.copy()
        if "idade_anos" in df.columns:
            df["age_group"] = eng.age_group(df["idade_anos"])
            df["idade_anos"] = pd.to_numeric(df["idade_anos"], errors="coerce")
        for col in ["CS_SEXO", "CS_RACA", "CS_ESCOL_N", "CS_GESTANT",
                    "AGENTE_TOX", "AGENTE_1", "CIRCUNSTAN", "VIA_1", "TPATENDE", "SIT_TRAB"]:
            if col in df.columns:
                df[col] = pd.Categorical(df[col].astype(str)).codes.astype(float)
        df = eng.clip_outliers(df, "idade_anos")
        return df

    def get_target(self, cohort: pd.DataFrame) -> pd.Series:
        # Sem fillna(0): censura não é negativo. build_cohort já exclui esses casos.
        return rotulo.exigir_rotulo(cohort[self.target_col], self.key)
