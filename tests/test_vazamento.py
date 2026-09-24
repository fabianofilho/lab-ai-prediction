"""Features proibidas por desfecho, pelo momento da predição (sem rede).

Falha se uma feature que só existe depois do momento da predição voltar às
suggested_features. Complementa o LEAK_BLACKLIST de test_invariants.py, que
cobre só a coluna-fonte do alvo.
"""
import pandas as pd
import pytest

from core.outcomes import OUTCOMES

# SIH: tudo que só se conhece na alta (permanência, diárias, UTI, valor total).
SIH_POS_ALTA = {
    "length_of_stay_days", "DIARIAS", "QT_DIARIAS", "DIAS_PERM",
    "used_icu", "UTI_MES_TO", "UTI_INT_TO", "VAL_UTI", "MARCA_UTI",
    "VAL_TOT",
}

# Desfechos do SIH com predição na admissão.
SIH_ADMISSAO = [
    "mortalidade_hospitalar",
    "permanencia_prolongada",
    "custo_elevado",
    "uso_uti",
    "infeccao_hospitalar",
]
# Desfechos do SIH com predição na alta: permanência, UTI e custo são
# informação legítima ali, então ficam fora da lista de proibidas.
SIH_NA_ALTA = ["readmissao_30d"]

PROIBIDAS: dict[str, set[str]] = {k: SIH_POS_ALTA for k in SIH_ADMISSAO}
# Dengue: internação é consequência da gravidade.
PROIBIDAS["dengue_grave"] = {"hospitalizado", "HOSPITALIZ"}

# Prefixos proibidos: sinais de alarme e de gravidade definem a classe final
# da dengue e são critério de internação na chikungunya.
PREFIXOS_PROIBIDOS: dict[str, tuple[str, ...]] = {
    "dengue_grave": ("ALRM_", "GRAV_"),
    "chikungunya_hospitalizado": ("ALRM_", "GRAV_"),
}


@pytest.mark.parametrize("key", sorted(PROIBIDAS))
def test_sem_feature_proibida(key):
    feats = set(OUTCOMES[key].suggested_features or [])
    vazam = feats & PROIBIDAS[key]
    assert not vazam, f"{key}: features que só existem depois da predição: {sorted(vazam)}"


@pytest.mark.parametrize("key", sorted(PREFIXOS_PROIBIDOS))
def test_sem_prefixo_proibido(key):
    feats = OUTCOMES[key].suggested_features or []
    vazam = [f for f in feats if f.startswith(PREFIXOS_PROIBIDOS[key])]
    assert not vazam, f"{key}: sinais de alarme ou gravidade como feature: {vazam}"


def test_todo_desfecho_sih_declara_momento_da_predicao():
    """Um desfecho novo do SIH precisa entrar em SIH_ADMISSAO ou SIH_NA_ALTA."""
    sih = {k for k, oc in OUTCOMES.items() if "SIH" in oc.data_sources}
    sem_momento = sih - set(SIH_ADMISSAO) - set(SIH_NA_ALTA)
    assert not sem_momento, f"desfechos do SIH sem momento de predição: {sorted(sem_momento)}"


def test_dengue_coorte_sem_alarme_gravidade_nem_internacao():
    oc = OUTCOMES["dengue_grave"]
    raw = pd.DataFrame({
        "CLASSI_FIN": ["10", "11", "12"],
        "HOSPITALIZ": ["2", "1", "1"],
        "ALRM_ABDOM": ["2", "1", "1"],
        "ALRM_VOM": ["2", "1", "2"],
        "GRAV_CONV": ["2", "2", "1"],
        "FEBRE": ["1", "1", "1"],
    })
    cohort = oc.build_cohort({"SINAN_DENG": raw})
    sobra = [c for c in cohort.columns
             if c.startswith(("ALRM_", "GRAV_")) or c in {"HOSPITALIZ", "hospitalizado"}]
    assert not sobra, f"colunas de vazamento na coorte de dengue_grave: {sobra}"
    assert "FEBRE" in cohort.columns


def test_chikungunya_coorte_sem_alarme():
    oc = OUTCOMES["chikungunya_hospitalizado"]
    raw = pd.DataFrame({
        "CLASSI_FIN": ["13", "13"],
        "HOSPITALIZ": ["1", "2"],
        "ALRM_PLAQ": ["1", "2"],
        "ALRM_VOM": ["1", "2"],
        "FEBRE": ["1", "1"],
    })
    cohort = oc.build_cohort({"SINAN_CHIK": raw})
    assert not [c for c in cohort.columns if c.startswith(("ALRM_", "GRAV_"))]
    assert cohort["hospitalizado"].tolist() == [1, 0]
