"""Readmissão hospitalar em 30 dias — SIH only."""

from __future__ import annotations

import pandas as pd

from core.outcomes.base import OutcomeConfig
from core.data import sih as sih_prep
from core.data import linker
from core.features import engineering as eng


class ReadmissaoHospitalar30d(OutcomeConfig):
    def __init__(self):
        super().__init__(
            key="readmissao_30d",
            name="Readmissão Hospitalar 30 dias",
            description=(
                "Prediz se um paciente será reinternado em até 30 dias após a alta hospitalar. "
                "Utiliza apenas dados do SIH-RD (internações SUS). "
                "O índice é a alta hospitalar (não morte). Como o SIH-RD público não expõe "
                "CNS/CPF, o paciente é identificado por uma chave probabilística "
                "(nascimento + sexo + município + CEP), no espírito da metodologia de "
                "record linkage por quase-identificadores usada em estudos de readmissão por AIH. "
                "Internações no mesmo dia da alta são tratadas como transferência."
            ),
            data_sources=["SIH"],
            observation_window_days=365,
            prediction_window_days=30,
            requires_linkage=False,
            icon="🔁",
            estimated_download_min=10,
            suggested_features=[
                "IDADE", "SEXO", "diag_chapter", "diag_block",
                "length_of_stay_days", "used_icu", "QT_DIARIAS",
                "n_diag_sec", "VAL_TOT", "proc_rea_code",
                "age_group", "RACA_COR",
            ],
            target_col="readmissao_30d",
        )

    def build_cohort(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        df = sih_prep.preprocess(data["SIH"])
        # Keep only alive discharges (index events)
        df = sih_prep.filter_alive_discharges(df)
        df = df.dropna(subset=["DT_SAIDA"])
        df = df.sort_values("DT_SAIDA")

        # Identificador de paciente: CNS/CPF direto se houver (dados não-públicos),
        # senão a chave probabilística por quase-identificadores do SIH-RD público.
        id_col = next(
            (c for c in ["CNS_PAC", "CPF_PAC"]
             if c in df.columns and df[c].astype(str).str.len().gt(5).any()),
            None,
        )
        if id_col is None:
            df["_patient_key"] = linker.build_patient_key(df)
            id_col = "_patient_key"

        df = linker.flag_readmission_self(df, id_col, window_days=30)
        return df.drop(columns=["_patient_key"], errors="ignore")

    def build_features(self, cohort: pd.DataFrame) -> pd.DataFrame:
        df = cohort.copy()

        # ICD-10 encoding
        if "DIAG_PRINC" in df.columns:
            df["diag_chapter"] = eng.icd10_chapter(df["DIAG_PRINC"])
            df["diag_block"] = eng.icd10_block(df["DIAG_PRINC"])

        # Number of secondary diagnoses
        diag_sec_col = next((c for c in ["DIAG_SEC", "DIAGSEC1", "DIAG_SECUN"] if c in df.columns), None)
        if diag_sec_col:
            df["n_diag_sec"] = (~df[diag_sec_col].isna()).astype(int)

        # Age group
        if "IDADE" in df.columns:
            df["age_group"] = eng.age_group(df["IDADE"])

        # Procedure code
        if "PROC_REA" in df.columns:
            df["proc_rea_code"] = pd.Categorical(df["PROC_REA"].astype(str)).codes.astype(float)

        # Sex binary
        if "SEXO" in df.columns:
            df["SEXO"] = pd.Categorical(df["SEXO"].astype(str)).codes.astype(float)

        # Race
        if "RACA_COR" in df.columns:
            df["RACA_COR"] = pd.Categorical(df["RACA_COR"].astype(str)).codes.astype(float)

        # Clip outliers on length of stay
        df = eng.clip_outliers(df, "length_of_stay_days")
        df = eng.clip_outliers(df, "VAL_TOT")

        return df

    def get_target(self, cohort: pd.DataFrame) -> pd.Series:
        return cohort[self.target_col].fillna(0).astype(int)
