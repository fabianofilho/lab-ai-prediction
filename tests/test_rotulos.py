"""Rótulos derivados dos códigos do SINAN (sem rede, determinístico).

Trava os mapas de código corrigidos na revisão da Onda 0: SITUA_ENCE da
tuberculose (com censura), CLASSI_FIN da dengue e a flag multibacilar da
hanseníase. Na Onda 2, os mapas conferidos contra o dicionário oficial
(PySUS 0.15.0 e microdatasus 3.0.0) e a censura dos desfechos do SINAN
pelo helper comum de core/data/rotulo.py.
"""
import logging

import numpy as np
import pandas as pd
import pytest

from core.data import rotulo
from core.data import sinan as tb_prep
from core.data import sinan_deng as deng_prep
from core.data import sinan_hans as hans_prep
from core.data import sinan_iexo as iexo_prep
from core.outcomes.abandono_hanseniase import AbandonoHanseniase
from core.outcomes.abandono_tb import AbandonoTB
from core.outcomes.dengue_grave import DengueGrave
from core.outcomes.incapacidade_hanseniase import IncapacidadeHanseniase
from core.outcomes.intoxicacao_grave import IntoxicacaoGrave
from core.outcomes.obito_tb import ObitoTB

# ── Helper de rótulo com censura ─────────────────────────────────────────────


def test_codigo_normaliza_espaco_float_e_zero_a_esquerda():
    bruto = pd.Series([" 2", "2.0", 2, "02", "10", "0", "00", np.nan, None, "", "M"])
    assert rotulo.codigo(bruto).tolist() == ["2", "2", "2", "2", "10", "0", "0", "", "", "", "M"]


def test_alvo_com_censura_so_da_zero_ao_negativo_explicito():
    cod = pd.Series(["1", "2", "3", "9", ""])
    alvo = rotulo.alvo_com_censura(cod, positivos={"2"}, negativos={"1"})
    assert alvo.iloc[:2].tolist() == [0.0, 1.0]
    assert alvo.iloc[2:].isna().all(), "censura, ignorado e em branco não podem virar 0"


def test_alvo_com_censura_recusa_codigo_nos_dois_lados():
    with pytest.raises(ValueError, match="positivos e negativos"):
        rotulo.alvo_com_censura(pd.Series(["1"]), positivos={"1"}, negativos={"1"})


def test_exigir_rotulo_falha_com_nan():
    with pytest.raises(ValueError, match="censura"):
        rotulo.exigir_rotulo(pd.Series([1.0, np.nan]), "x")
    assert rotulo.exigir_rotulo(pd.Series([1.0, 0.0]), "x").tolist() == [1, 0]

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


# ── Hanseníase: flag multibacilar ────────────────────────────────────────────

def _hans_raw():
    # FORMACLINI e CLASSOPERA discordam de propósito: tuberculoide (2) é PB e
    # dimorfa/virchowiana (3, 4) são MB.
    return pd.DataFrame({
        "FORMACLINI": ["1", "2", "3", "4", "2"],
        "CLASSOPERA": ["1", "1", "2", "2", "2.0"],
        "TPALTA_N":   ["1", "1", "1", "1", "1"],
        "AVALIA_N":   ["0", "0", "2", "1", "0"],
        "NU_IDADE_N": [4040] * 5,
        "CS_SEXO":    ["M"] * 5,
    })


def test_hanseniase_mb_vem_de_classopera():
    df = hans_prep.preprocess(_hans_raw())
    assert df["mb"].tolist() == [0, 0, 1, 1, 1]


def test_hanseniase_mb_ausente_sem_classopera():
    raw = _hans_raw().drop(columns=["CLASSOPERA"])
    df = hans_prep.preprocess(raw)
    assert "mb" not in df.columns, "mb não pode voltar a ser derivado de FORMACLINI"


@pytest.mark.parametrize("outcome_cls", [AbandonoHanseniase, IncapacidadeHanseniase])
def test_hanseniase_desfechos_usam_mb_do_classopera(outcome_cls):
    oc = outcome_cls()
    assert "mb" in oc.suggested_features
    cohort = oc.build_features(oc.build_cohort({"SINAN_HANS": _hans_raw()}))
    assert cohort["mb"].tolist() == [0, 0, 1, 1, 1]


# ── Hanseníase: TPALTA_N (tipo de saída) ─────────────────────────────────────

# Dicionário do SINAN-Hanseníase (PySUS 0.15.0, HANS.csv, campo 19).
# Código -> abandono esperado. NaN = censura ou sem código, sai da coorte.
HANS_TPALTA_ESPERADO = {
    "1": 0,            # cura
    "2": np.nan,       # transferência para o mesmo município
    "3": np.nan,       # transferência para outro município (o app lia abandono)
    "4": np.nan,       # transferência para outro estado
    "5": np.nan,       # transferência para outro país
    "6": np.nan,       # óbito (risco competitivo)
    "7": 1,            # abandono
    "8": np.nan,       # erro diagnóstico
    "9": np.nan,       # transferência não especificada
    "": np.nan,        # em branco
}


def _hans_saida_raw(codigos):
    n = len(codigos)
    return pd.DataFrame({
        "TPALTA_N": codigos,
        "CLASSOPERA": ["2"] * n,
        "AVALIA_N": ["0"] * n,
        "NU_IDADE_N": [4040] * n,
        "CS_SEXO": ["F"] * n,
    })


def test_hanseniase_tpalta_mapa_abandono():
    df = hans_prep.preprocess(_hans_saida_raw(list(HANS_TPALTA_ESPERADO)))
    for i, (cod, esperado) in enumerate(HANS_TPALTA_ESPERADO.items()):
        got = df["abandono"].iloc[i]
        if np.isnan(esperado):
            assert np.isnan(got), f"TPALTA_N {cod!r}: abandono deveria ficar sem rótulo, veio {got}"
        else:
            assert got == esperado, f"TPALTA_N {cod!r}: abandono {got} != {esperado}"


def test_hanseniase_filtro_de_encerrados_mantem_o_abandono():
    df = hans_prep.filter_closed_cases(hans_prep.preprocess(_hans_saida_raw(["1", "6", "7", "8", "9", "", "0"])))
    assert df["TPALTA_N"].tolist() == ["1", "6", "7", "8", "9"]


def test_hanseniase_abandono_coorte_exclui_censura_e_conta(caplog):
    oc = AbandonoHanseniase()
    raw = _hans_saida_raw(list(HANS_TPALTA_ESPERADO) + ["7.0", " 1"])
    with caplog.at_level(logging.WARNING):
        cohort = oc.build_cohort({"SINAN_HANS": raw})

    # 7 censurados (2 a 6, 8 e 9); o em branco sai no filtro de encerrados
    assert cohort.attrs["exclusoes_rotulo"] == {"censura": 7, "sem_codigo": 0, "mantidos": 4}
    assert "7 casos censurados (TPALTA_N 2, 3, 4, 5, 6, 8, 9)" in caplog.text
    y = oc.get_target(oc.build_features(cohort))
    assert y.tolist() == [0, 1, 1, 0]


def test_hanseniase_abandono_get_target_nao_preenche_censura_com_zero():
    oc = AbandonoHanseniase()
    with pytest.raises(ValueError, match="censura"):
        oc.get_target(pd.DataFrame({oc.target_col: [1.0, 0.0, np.nan]}))


# ── Hanseníase: AVALIA_N (grau de incapacidade no diagnóstico) ───────────────

# Dicionário do SINAN-Hanseníase (PySUS 0.15.0, HANS.csv, campo 37).
# Código -> (grau_incapacidade esperado, incapacidade_g2 esperado).
HANS_AVALIA_ESPERADO = {
    "0": (0.0, 0),          # grau zero
    "1": (1.0, 0),          # grau I
    "2": (2.0, 1),          # grau II
    "3": (np.nan, None),    # não avaliado: nem grau 3, nem grau zero
    "": (np.nan, None),     # em branco
}


def _hans_avalia_raw(codigos):
    raw = _hans_saida_raw(["1"] * len(codigos))
    raw["AVALIA_N"] = codigos
    raw["DT_NOTIFIC"] = pd.to_datetime(["2023-03-01"] * len(codigos))
    raw["DT_DIAG"] = pd.to_datetime(["2023-02-20"] * len(codigos))
    return raw


def test_hanseniase_avalia_nao_avaliado_nao_e_grau():
    df = hans_prep.preprocess(_hans_avalia_raw(list(HANS_AVALIA_ESPERADO)))
    for i, (cod, (grau, _)) in enumerate(HANS_AVALIA_ESPERADO.items()):
        got = df["grau_incapacidade"].iloc[i]
        if np.isnan(grau):
            assert np.isnan(got), f"AVALIA_N {cod!r}: grau_incapacidade deveria ser ausente, veio {got}"
        else:
            assert got == grau, f"AVALIA_N {cod!r}: grau_incapacidade {got} != {grau}"


def test_hanseniase_incapacidade_coorte_exclui_nao_avaliado(caplog):
    oc = IncapacidadeHanseniase()
    with caplog.at_level(logging.WARNING):
        cohort = oc.build_cohort({"SINAN_HANS": _hans_avalia_raw(list(HANS_AVALIA_ESPERADO))})
    assert cohort.attrs["exclusoes_rotulo"] == {"censura": 1, "sem_codigo": 1, "mantidos": 3}
    assert "1 casos censurados (AVALIA_N 3)" in caplog.text
    y = oc.get_target(oc.build_features(cohort))
    esperado = [g2 for g2 in (v[1] for v in HANS_AVALIA_ESPERADO.values()) if g2 is not None]
    assert y.tolist() == esperado
    assert "AVALIA_N" not in cohort.columns and "grau_incapacidade" not in cohort.columns


def test_hanseniase_incapacidade_get_target_nao_preenche_com_zero():
    oc = IncapacidadeHanseniase()
    with pytest.raises(ValueError, match="censura"):
        oc.get_target(pd.DataFrame({oc.target_col: [1.0, np.nan]}))


# ── Intoxicação exógena: EVOLUCAO ────────────────────────────────────────────

# Dicionário do SINAN-Intoxicação Exógena (PySUS 0.15.0, IEXO.csv, campo 68).
# Código -> desfecho_adverso esperado. NaN = censura ou sem código.
IEXO_EVOLUCAO_ESPERADO = {
    "1": 0,          # cura sem sequela
    "2": 1,          # cura com sequela
    "3": 1,          # óbito por intoxicação exógena
    "4": np.nan,     # óbito por outra causa (risco competitivo; o app contava como grave)
    "5": np.nan,     # perda de seguimento (o app contava como grave)
    "9": np.nan,     # ignorado
    "": np.nan,      # em branco
}


def _iexo_raw(evolucao, circunstan=None):
    n = len(evolucao)
    return pd.DataFrame({
        "CLASSI_FIN": ["1"] * n,
        "EVOLUCAO": evolucao,
        "CIRCUNSTAN": circunstan if circunstan is not None else ["02"] * n,
        "NU_IDADE_N": [4025] * n,
        "CS_SEXO": ["F"] * n,
        "AGENTE_TOX": ["01"] * n,
    })


def test_intoxicacao_mapa_evolucao():
    df = iexo_prep.preprocess(_iexo_raw(list(IEXO_EVOLUCAO_ESPERADO)))
    for i, (cod, esperado) in enumerate(IEXO_EVOLUCAO_ESPERADO.items()):
        got = df["desfecho_adverso"].iloc[i]
        if np.isnan(esperado):
            assert np.isnan(got), f"EVOLUCAO {cod!r}: desfecho_adverso deveria ficar sem rótulo, veio {got}"
        else:
            assert got == esperado, f"EVOLUCAO {cod!r}: desfecho_adverso {got} != {esperado}"
    assert df["obito"].tolist() == [0, 0, 1, 0, 0, 0, 0], "obito é só o óbito por intoxicação (3)"


def test_intoxicacao_coorte_exclui_censura_e_conta(caplog):
    oc = IntoxicacaoGrave()
    raw = _iexo_raw(list(IEXO_EVOLUCAO_ESPERADO))
    raw.loc[len(raw)] = raw.iloc[0].to_dict() | {"CLASSI_FIN": "2", "EVOLUCAO": "3"}  # só exposição
    with caplog.at_level(logging.WARNING):
        cohort = oc.build_cohort({"SINAN_IEXO": raw})
    # 4 e 5 censurados, 9 e em branco sem código; a exposição sai no filtro
    assert cohort.attrs["exclusoes_rotulo"] == {"censura": 2, "sem_codigo": 2, "mantidos": 3}
    assert "2 casos censurados (EVOLUCAO 4, 5)" in caplog.text
    assert "EVOLUCAO" not in cohort.columns
    y = oc.get_target(oc.build_features(cohort))
    assert y.tolist() == [0, 1, 1]


def test_intoxicacao_get_target_nao_preenche_censura_com_zero():
    oc = IntoxicacaoGrave()
    with pytest.raises(ValueError, match="censura"):
        oc.get_target(pd.DataFrame({oc.target_col: [1.0, np.nan]}))


# ── Intoxicação exógena: CIRCUNSTAN ──────────────────────────────────────────

# Dicionário do SINAN-Intoxicação Exógena (PySUS 0.15.0, IEXO.csv, campo 55):
# 02 é uso acidental e 10 é tentativa de suicídio.
IEXO_CIRCUNSTAN_ESPERADO = {
    "02": 0,     # acidental (o app lia "2" como tentativa de suicídio)
    "2": 0,      # acidental sem o zero à esquerda
    "10": 1,     # tentativa de suicídio
    "10.0": 1,
    "12": 0,     # violência/homicídio
    "99": 0,     # ignorado (a troca de 0 por ausente nas features fica para a migração)
}


def test_intoxicacao_tentativa_suicidio_e_circunstan_10():
    codigos = list(IEXO_CIRCUNSTAN_ESPERADO)
    df = iexo_prep.preprocess(_iexo_raw(["1"] * len(codigos), circunstan=codigos))
    assert df["tentativa_suicidio"].tolist() == list(IEXO_CIRCUNSTAN_ESPERADO.values())
