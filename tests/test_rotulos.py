"""Rótulos derivados dos códigos do SINAN (sem rede, determinístico).

Trava os mapas de código corrigidos na revisão da Onda 0: SITUA_ENCE da
tuberculose (com censura), CLASSI_FIN da dengue e a flag multibacilar da
hanseníase.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from core.data import sinan as tb_prep
from core.data import sinan_deng as deng_prep
from core.outcomes.abandono_tb import AbandonoTB
from core.outcomes.dengue_grave import DengueGrave
from core.outcomes.obito_tb import ObitoTB

# ── Tuberculose: SITUA_ENCE ──────────────────────────────────────────────────

# Código bruto -> (abandono esperado, obito_tb esperado). NaN = censura ou
# código fora do dicionário, que sai da coorte.
TB_ESPERADO = {
    "1": (0, 0),              # cura
    "2": (1, 0),              # abandono
    "3": (0, 1),              # óbito por TB
    "4": (0, 1),              # óbito por outras causas
    "5": (np.nan, np.nan),    # transferência (censura)
    "6": (np.nan, np.nan),    # mudança de diagnóstico (censura)
    "7": (np.nan, np.nan),    # TB-DR (censura)
    "8": (np.nan, np.nan),    # mudança de esquema (censura)
    "9": (0, 0),              # falência
    "10": (1, 0),             # abandono primário
    "": (np.nan, np.nan),     # em branco: sem código válido
}


def _tb_raw(codigos):
    n = len(codigos)
    return pd.DataFrame({
        "SITUA_ENCE": codigos,
        "DT_ENCERRA": pd.to_datetime(["2023-06-30"] * n),
        "NU_IDADE_N": [4035] * n,
        "CS_SEXO": ["M"] * n,
        "HIV": ["2"] * n,
        "TRAT_SUPER": ["1"] * n,
    })


def test_tb_mapa_abandono_e_obito():
    df = tb_prep.preprocess(_tb_raw(list(TB_ESPERADO)))
    for i, (cod, (ab, ob)) in enumerate(TB_ESPERADO.items()):
        got_ab, got_ob = df["abandono"].iloc[i], df["obito_tb"].iloc[i]
        if np.isnan(ab):
            assert np.isnan(got_ab), f"SITUA_ENCE {cod!r}: abandono deveria ser censura"
            assert np.isnan(got_ob), f"SITUA_ENCE {cod!r}: obito_tb deveria ser censura"
        else:
            assert got_ab == ab, f"SITUA_ENCE {cod!r}: abandono {got_ab} != {ab}"
            assert got_ob == ob, f"SITUA_ENCE {cod!r}: obito_tb {got_ob} != {ob}"


def test_tb_codigo_bruto_com_espaco_e_float():
    df = tb_prep.preprocess(_tb_raw([" 2", "2.0", "10 ", 3]))
    assert df["abandono"].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert df["obito_tb"].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_tb_positivos_e_censura_sao_os_do_dicionario():
    assert tb_prep.SITUA_ABANDONO == {"2", "10"}
    assert tb_prep.SITUA_OBITO == {"3", "4"}
    assert tb_prep.SITUA_CENSURA == {"5", "6", "7", "8"}
    # obito por TB (3) nunca pode voltar a ser abandono, nem abandono (2) obito
    assert "3" not in tb_prep.SITUA_ABANDONO
    assert "2" not in tb_prep.SITUA_OBITO


@pytest.mark.parametrize("outcome_cls, positivos", [
    (AbandonoTB, {"2", "10"}),
    (ObitoTB, {"3", "4"}),
])
def test_tb_coorte_exclui_censura_e_conta(outcome_cls, positivos, caplog):
    oc = outcome_cls()
    raw = _tb_raw(list(TB_ESPERADO))
    with caplog.at_level(logging.WARNING, logger="core.data.sinan"):
        cohort = oc.build_cohort({"SINAN_TB": raw})

    # 4 censurados (5 a 8) e 1 sem código saem; os 6 encerramentos rotulados ficam
    assert cohort.attrs["exclusoes_rotulo"] == {"censura": 4, "sem_codigo": 1, "mantidos": 6}
    assert "4 casos censurados" in caplog.text

    y = oc.get_target(oc.build_features(cohort))
    assert len(y) == 6
    assert y.notna().all()
    assert set(y.unique()) == {0, 1}
    esperado_pos = sum(1 for c in TB_ESPERADO if c in positivos)
    assert int(y.sum()) == esperado_pos


@pytest.mark.parametrize("outcome_cls", [AbandonoTB, ObitoTB])
def test_tb_get_target_nao_preenche_censura_com_zero(outcome_cls):
    oc = outcome_cls()
    cohort = pd.DataFrame({oc.target_col: [1.0, 0.0, np.nan]})
    with pytest.raises(ValueError, match="censura"):
        oc.get_target(cohort)


# ── Dengue: CLASSI_FIN (layout 2014 em diante) ───────────────────────────────

# Código -> (na coorte confirmada?, dengue_grave esperado)
DENGUE_ESPERADO = {
    "5": (False, None),   # descartado
    "8": (False, None),   # inconclusivo
    "10": (True, 0),      # dengue
    "11": (True, 1),      # dengue com sinais de alarme
    "12": (True, 1),      # dengue grave
    "13": (False, None),  # chikungunya
}


def _deng_raw(codigos):
    n = len(codigos)
    return pd.DataFrame({
        "CLASSI_FIN": codigos,
        "NU_IDADE_N": [4030] * n,
        "CS_SEXO": ["F"] * n,
        "FEBRE": ["1"] * n,
    })


def test_dengue_mapa_classi_fin():
    df = deng_prep.preprocess(_deng_raw(list(DENGUE_ESPERADO)))
    for i, (cod, (confirmado, grave)) in enumerate(DENGUE_ESPERADO.items()):
        assert df["dengue_confirmado"].iloc[i] == int(confirmado), f"CLASSI_FIN {cod}"
        if confirmado:
            assert df["dengue_grave"].iloc[i] == grave, f"CLASSI_FIN {cod}"


def test_dengue_coorte_so_confirmados_e_grave_fica():
    oc = DengueGrave()
    cohort = oc.build_cohort({"SINAN_DENG": _deng_raw(["5", "8", "10", "11", "12", "13", "12.0"])})
    y = oc.get_target(oc.build_features(cohort))
    # 10, 11, 12 e 12.0 ficam; 5, 8 e 13 saem
    assert len(y) == 4
    assert y.tolist() == [0, 1, 1, 1]


def test_dengue_inconclusivo_nunca_e_positivo():
    assert deng_prep.CLASSI_INCONCLUSIVO not in deng_prep.CLASSI_CONFIRMADO
    assert deng_prep.CLASSI_CONFIRMADO == {"10", "11", "12"}
    assert {deng_prep.CLASSI_ALARME, deng_prep.CLASSI_GRAVE} == {"11", "12"}
