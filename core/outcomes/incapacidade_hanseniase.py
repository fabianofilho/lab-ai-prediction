"""Incapacidade física grau 2 (G2D) ao diagnóstico de hanseníase — SINAN-HANS.

G2D ao diagnóstico é o indicador de detecção tardia monitorado pela OMS.
Prediz a partir de características disponíveis no diagnóstico quais casos
chegam já com incapacidade grau 2, para priorização de busca ativa.
"""
from __future__ import annotations

import pandas as pd

from core.outcomes.base import OutcomeConfig
from core.data import rotulo
from core.data import sinan_hans as hans_prep
from core.features import engineering as eng


class IncapacidadeHanseniase(OutcomeConfig):
    def __init__(self):
        super().__init__(
            key="incapacidade_hanseniase",
            name="Incapacidade Grau 2 — Hanseníase",
            description=(
                "Prediz a presença de incapacidade física grau 2 (G2D) ao diagnóstico "
                "de hanseníase (AVALIA_N = 2), indicador de detecção tardia monitorado "
                "pela OMS, contra grau zero e grau I (AVALIA_N = 0 ou 1). Não avaliado "
                "(AVALIA_N = 3) e em branco saem da coorte. Features incluem forma "
                "clínica, classificação operacional, modo de detecção, baciloscopia, "
                "tempo entre notificação e diagnóstico, e características "
                "sociodemográficas. Utiliza SINAN-Hanseníase."
            ),
            data_sources=["SINAN_HANS"],
            observation_window_days=0,
            prediction_window_days=0,
            requires_linkage=False,
            icon="accessible",
            estimated_download_min=5,
            suggested_features=[
                "idade_anos", "CS_SEXO", "CS_RACA", "CS_ESCOL_N",
                "FORMACLINI", "mb", "CLASSOPERA",
                "MODOENTR", "MODODETECT", "BACILOSCOP", "ESQ_INI_N",
                "dias_notif_diag", "age_group",
            ],
            target_col="incapacidade_g2",
        )

    def build_cohort(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = hans_prep.preprocess(data["SINAN_HANS"])

        # Alvo: grau II ao diagnóstico (AVALIA_N 2) contra grau zero e grau I.
        # Não avaliado (3) é censura e o em branco fica sem código: os dois
        # saem da coorte em vez de virar 0. Sem a coluna não há rótulo, e um
        # alvo todo zero seria uma coorte 100% negativa sem aviso.
        if "AVALIA_N" not in df.columns:
            raise KeyError("AVALIA_N ausente: sem rótulo para incapacidade_g2")
        df["incapacidade_g2"] = rotulo.alvo_com_censura(
            rotulo.codigo(df["AVALIA_N"]), positivos={"2"}, negativos={"0", "1"},
        )
        df = rotulo.excluir_sem_rotulo(
            df, "incapacidade_g2", "AVALIA_N", hans_prep.AVALIA_NAO_AVALIADO,
        )

        # Tempo entre notificação e diagnóstico (proxy de atraso)
        if "DT_NOTIFIC" in df.columns and "DT_DIAG" in df.columns:
            df["dias_notif_diag"] = (
                pd.to_datetime(df["DT_DIAG"], errors="coerce")
                - pd.to_datetime(df["DT_NOTIFIC"], errors="coerce")
            ).dt.days

        # Anti-leakage / remoção de info pós-diagnóstico (futuro):
        # AVALIA_N e grau_incapacidade são a fonte do alvo; TPALTA_N/abandono/cura/
        # DOSE_RECEB só são conhecidos ao fim do tratamento.
        df = df.drop(columns=[
            "AVALIA_N", "grau_incapacidade",
            "TPALTA_N", "abandono", "cura", "DOSE_RECEB",
        ], errors="ignore")
        return df

    def build_features(self, cohort: pd.DataFrame) -> pd.DataFrame:
        df = cohort.copy()

        if "idade_anos" in df.columns:
            df["age_group"] = eng.age_group(df["idade_anos"])
            df["idade_anos"] = pd.to_numeric(df["idade_anos"], errors="coerce")

        for col in ["CS_SEXO", "CS_RACA", "CS_ESCOL_N",
                    "FORMACLINI", "CLASSOPERA", "MODOENTR",
                    "MODODETECT", "BACILOSCOP", "ESQ_INI_N"]:
            if col in df.columns:
                df[col] = pd.Categorical(df[col].astype(str)).codes.astype(float)

        if "dias_notif_diag" in df.columns:
            df["dias_notif_diag"] = pd.to_numeric(df["dias_notif_diag"], errors="coerce")
            df = eng.clip_outliers(df, "dias_notif_diag")

        df = eng.clip_outliers(df, "idade_anos")
        return df

    def get_target(self, cohort: pd.DataFrame) -> pd.Series:
        # Sem fillna(0): não avaliado não é grau zero. build_cohort já exclui.
        return rotulo.exigir_rotulo(cohort[self.target_col], self.key)
