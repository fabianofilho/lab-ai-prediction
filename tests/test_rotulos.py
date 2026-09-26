"""Rótulos derivados dos códigos do SINAN (sem rede, determinístico).

Trava os mapas de código corrigidos na revisão da Onda 0: SITUA_ENCE da
tuberculose (com censura), CLASSI_FIN da dengue e a flag multibacilar da
hanseníase. Na Onda 2, os mapas conferidos contra o dicionário oficial
(PySUS 0.15.0 e microdatasus 3.0.0) e a censura dos desfechos do SINAN
pelo helper comum de core/data/rotulo.py.
"""
import logging
import re

import numpy as np
import pandas as pd
import pytest

from core.data import rotulo
from core.data import sinan as tb_prep
from core.data import sinan_aids as aids_prep
from core.data import sinan_chik as chik_prep
from core.data import sinan_deng as deng_prep
from core.data import sinan_hans as hans_prep
from core.data import sinan_iexo as iexo_prep
from core.features.data_dict import FEATURE_DICT
from core.methodology import METHODOLOGY
from core.outcomes.abandono_hanseniase import AbandonoHanseniase
from core.outcomes.abandono_tb import AbandonoTB
from core.outcomes.dengue_grave import DengueGrave
from core.outcomes.incapacidade_hanseniase import IncapacidadeHanseniase
from core.outcomes.intoxicacao_grave import IntoxicacaoGrave
from core.outcomes.obito_aids import ObitoAIDS
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
# código fora do dicionário, que sai da coorte. No abandono, o óbito é
# censura por risco competitivo.
TB_ESPERADO = {
    "1": (0, 0),              # cura
    "2": (1, 0),              # abandono
    "3": (np.nan, 1),         # óbito por TB
    "4": (np.nan, 1),         # óbito por outras causas
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
    for i, (cod, esperados) in enumerate(TB_ESPERADO.items()):
        for col, esperado in zip(("abandono", "obito_tb"), esperados):
            got = df[col].iloc[i]
            if np.isnan(esperado):
                assert np.isnan(got), f"SITUA_ENCE {cod!r}: {col} deveria ser censura, veio {got}"
            else:
                assert got == esperado, f"SITUA_ENCE {cod!r}: {col} {got} != {esperado}"


def test_tb_codigo_bruto_com_espaco_e_float():
    df = tb_prep.preprocess(_tb_raw([" 2", "2.0", "10 ", 3, " 1"]))
    assert df["abandono"].iloc[[0, 1, 2, 4]].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert np.isnan(df["abandono"].iloc[3]), "óbito por TB é censura no abandono"
    assert df["obito_tb"].tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]


def test_tb_codigo_com_zero_a_esquerda():
    # Desde o helper comum (96cb137), "02" é o mesmo código que "2". Na Onda 0
    # o código com zero à esquerda caía fora do mapa e saía como sem código.
    df = tb_prep.preprocess(_tb_raw(["01", "02", "03", "05", "09", "10"]))
    abandono = df["abandono"].tolist()
    assert [abandono[i] for i in (0, 1, 4, 5)] == [0.0, 1.0, 0.0, 1.0]
    assert np.isnan(abandono[2]) and np.isnan(abandono[3]), "óbito e transferência são censura"
    assert df["obito_tb"].iloc[[0, 1, 2, 4, 5]].tolist() == [0.0, 0.0, 1.0, 0.0, 0.0]

    cohort = AbandonoTB().build_cohort({"SINAN_TB": _tb_raw(["01", "02", "03", "05", "09", "10"])})
    assert cohort.attrs["exclusoes_rotulo"] == {"censura": 2, "sem_codigo": 0, "mantidos": 4}


def test_tb_positivos_e_censura_sao_os_do_dicionario():
    assert tb_prep.SITUA_ABANDONO == {"2", "10"}
    assert tb_prep.SITUA_OBITO == {"3", "4"}
    assert tb_prep.SITUA_CENSURA == {"5", "6", "7", "8"}
    assert tb_prep.CENSURA_POR_ALVO["abandono"] == {"3", "4", "5", "6", "7", "8"}
    assert tb_prep.CENSURA_POR_ALVO["obito_tb"] == {"5", "6", "7", "8"}
    assert tb_prep.SITUA_NEGATIVO_ABANDONO == {"1", "9"}
    # obito por TB (3) nunca pode voltar a ser abandono, nem abandono (2) obito
    assert "3" not in tb_prep.SITUA_ABANDONO
    assert "2" not in tb_prep.SITUA_OBITO


@pytest.mark.parametrize("outcome_cls, positivos, n_censura", [
    (AbandonoTB, {"2", "10"}, 6),   # 3 a 8
    (ObitoTB, {"3", "4"}, 4),       # 5 a 8
])
def test_tb_coorte_exclui_censura_e_conta(outcome_cls, positivos, n_censura, caplog):
    oc = outcome_cls()
    raw = _tb_raw(list(TB_ESPERADO))
    with caplog.at_level(logging.WARNING, logger="core.data.sinan"):
        cohort = oc.build_cohort({"SINAN_TB": raw})

    # censurados e 1 sem código saem; os encerramentos rotulados ficam
    mantidos = len(TB_ESPERADO) - n_censura - 1
    assert cohort.attrs["exclusoes_rotulo"] == {
        "censura": n_censura, "sem_codigo": 1, "mantidos": mantidos,
    }
    assert f"{n_censura} casos censurados" in caplog.text

    y = oc.get_target(oc.build_features(cohort))
    assert len(y) == mantidos
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


def test_hanseniase_tpalta_classes_cobrem_o_dicionario():
    # cura, abandono e censura partem o dicionário sem sobra nem sobreposição
    classes = [hans_prep.TPALTA_CURA, hans_prep.TPALTA_ABANDONO, hans_prep.TPALTA_CENSURA_ABANDONO]
    assert set().union(*classes) == set(hans_prep.TPALTA_N_MAPA)
    assert sum(len(c) for c in classes) == len(hans_prep.TPALTA_N_MAPA)


def test_hanseniase_abandono_coorte_mantem_o_abandono_e_conta_o_resto():
    oc = AbandonoHanseniase()
    cohort = oc.build_cohort({"SINAN_HANS": _hans_saida_raw(["1", "6", "7", "8", "9", "", "0"])})
    # o abandono (7) fica; em branco e "0" (fora do dicionário) saem contados
    assert cohort["TPALTA_N"].tolist() == ["1", "7"]
    assert cohort.attrs["exclusoes_rotulo"] == {"censura": 3, "sem_codigo": 2, "mantidos": 2}


def test_hanseniase_abandono_coorte_exclui_censura_e_conta(caplog):
    oc = AbandonoHanseniase()
    raw = _hans_saida_raw(list(HANS_TPALTA_ESPERADO) + ["7.0", " 1"])
    with caplog.at_level(logging.WARNING):
        cohort = oc.build_cohort({"SINAN_HANS": raw})

    # 7 censurados (2 a 6, 8 e 9); o em branco sai contado como sem código
    assert cohort.attrs["exclusoes_rotulo"] == {"censura": 7, "sem_codigo": 1, "mantidos": 4}
    assert "7 casos censurados (TPALTA_N 2, 3, 4, 5, 6, 8, 9) e 1 sem código válido" in caplog.text
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


def test_hanseniase_incapacidade_sem_avalia_n_falha_em_vez_de_zerar():
    raw = _hans_avalia_raw(["0", "2"]).drop(columns=["AVALIA_N"])
    with pytest.raises(KeyError, match="AVALIA_N ausente"):
        IncapacidadeHanseniase().build_cohort({"SINAN_HANS": raw})


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


# ── Chikungunya: EVOLUCAO ────────────────────────────────────────────────────

# Dicionário do SINAN-Chikungunya (PySUS 0.15.0, CHIK.csv, campo 65; o
# microdatasus concorda). Nenhum desfecho usa a flag obito hoje.
CHIK_EVOLUCAO_ESPERADO = {
    "1": 0,     # cura
    "2": 1,     # óbito pelo agravo
    "3": 0,     # óbito por outras causas (o app lia como óbito por chikungunya)
    "4": 0,     # óbito em investigação
    "9": 0,     # ignorado
}


def test_chikungunya_obito_pelo_agravo_e_evolucao_2():
    codigos = list(CHIK_EVOLUCAO_ESPERADO)
    raw = pd.DataFrame({
        "CLASSI_FIN": ["13"] * len(codigos),
        "EVOLUCAO": codigos,
        "HOSPITALIZ": ["2"] * len(codigos),
        "NU_IDADE_N": [4050] * len(codigos),
    })
    df = chik_prep.preprocess(raw)
    assert df["obito"].tolist() == list(CHIK_EVOLUCAO_ESPERADO.values())


# ── AIDS adulto: EVOLUCAO ────────────────────────────────────────────────────

# Mapa do app (1 vivo, 2 óbito por aids, 3 óbito por outras causas, 9
# ignorado), ainda sem dicionário oficial conferido. Código -> obito_aids.
AIDS_EVOLUCAO_ESPERADO = {
    "1": 0,          # vivo
    "2": 1,          # óbito por aids
    "3": np.nan,     # óbito por outras causas (risco competitivo; o app dava 0)
    "9": np.nan,     # ignorado
    "": np.nan,      # em branco
}


def _aids_raw(codigos):
    n = len(codigos)
    return pd.DataFrame({
        "EVOLUCAO": codigos,
        "NU_IDADE_N": [4035] * n,
        "CS_SEXO": ["M"] * n,
        "ANT_TUBERC": ["1"] * n,
    })


def test_aids_obito_por_outras_causas_e_censura():
    df = aids_prep.preprocess(_aids_raw(list(AIDS_EVOLUCAO_ESPERADO)))
    for i, (cod, esperado) in enumerate(AIDS_EVOLUCAO_ESPERADO.items()):
        got = df["obito_aids"].iloc[i]
        if np.isnan(esperado):
            assert np.isnan(got), f"EVOLUCAO {cod!r}: obito_aids deveria ficar sem rótulo, veio {got}"
        else:
            assert got == esperado, f"EVOLUCAO {cod!r}: obito_aids {got} != {esperado}"


def test_aids_coorte_exclui_censura_e_conta(caplog):
    oc = ObitoAIDS()
    with caplog.at_level(logging.WARNING):
        cohort = oc.build_cohort({"SINAN_AIDS": _aids_raw(list(AIDS_EVOLUCAO_ESPERADO) + ["2.0"])})
    # 3 sai como censura; 9 e em branco saem contados como sem código, não
    # num filtro anterior que os descartava sem contar
    assert cohort.attrs["exclusoes_rotulo"] == {"censura": 1, "sem_codigo": 2, "mantidos": 3}
    assert "1 casos censurados (EVOLUCAO 3) e 2 sem código válido" in caplog.text
    y = oc.get_target(oc.build_features(cohort))
    assert y.tolist() == [0, 1, 1]


def test_aids_get_target_nao_preenche_censura_com_zero():
    oc = ObitoAIDS()
    with pytest.raises(ValueError, match="censura"):
        oc.get_target(pd.DataFrame({oc.target_col: [1.0, np.nan]}))


# ── Dicionário de dados exibido na tela (core/features/data_dict.py) ─────────

def test_data_dict_raca_cor_do_sih_tem_parda_em_3():
    # microdatasus 3.0.0, R/process_sih.R: no SIH, 3 é parda e 4 é amarela
    valores = FEATURE_DICT["RACA_COR"]["values"]
    assert valores["3"] == "Parda"
    assert valores["4"] == "Amarela"
    assert "Parda (3), Amarela (4)" in FEATURE_DICT["RACA_COR"]["desc"]


def test_data_dict_modo_de_entrada_e_deteccao_da_hanseniase():
    # PySUS 0.15.0, HANS.csv, campos 38 e 39
    assert FEATURE_DICT["MODOENTR"]["values"] == {
        "1": "Caso novo", "2": "Transferência do mesmo município",
        "3": "Transferência de outro município", "4": "Transferência de outro estado",
        "5": "Transferência de outro país", "6": "Recidiva", "7": "Outros reingressos",
        "9": "Ignorado",
    }
    assert FEATURE_DICT["MODODETECT"]["values"] == {
        "1": "Encaminhamento", "2": "Demanda espontânea", "3": "Exame de coletividade",
        "4": "Exame de contatos", "5": "Outros modos", "9": "Ignorado",
    }


@pytest.mark.parametrize("chave", ["RACA_COR", "MODOENTR", "MODODETECT"])
def test_data_dict_desc_cita_os_mesmos_codigos_dos_valores(chave):
    # A descrição é o que a tela mostra; não pode divergir do mapa de valores
    entrada = FEATURE_DICT[chave]
    for cod, rotulo_valor in entrada["values"].items():
        assert f"{rotulo_valor} ({cod})" in entrada["desc"], f"{chave}: falta '{rotulo_valor} ({cod})'"


# ── Textos da tela (drawer e descrição) citam os códigos das constantes ──────

OUTCOME_CLS = {
    "abandono_hanseniase": AbandonoHanseniase,
    "incapacidade_hanseniase": IncapacidadeHanseniase,
    "intoxicacao_grave": IntoxicacaoGrave,
    "obito_aids": ObitoAIDS,
}

# chave -> (campo, positivos, negativos, censura), das constantes do preprocessador
ROTULOS_CITADOS = {
    "abandono_hanseniase": ("TPALTA_N", hans_prep.TPALTA_ABANDONO, hans_prep.TPALTA_CURA,
                            hans_prep.TPALTA_CENSURA_ABANDONO),
    "incapacidade_hanseniase": ("AVALIA_N", {"2"}, {"0", "1"}, hans_prep.AVALIA_NAO_AVALIADO),
    "intoxicacao_grave": ("EVOLUCAO", iexo_prep.EVOLUCAO_ADVERSO, iexo_prep.EVOLUCAO_CURA_SEM_SEQUELA,
                          iexo_prep.EVOLUCAO_CENSURA),
    "obito_aids": ("EVOLUCAO", aids_prep.EVOLUCAO_OBITO_AIDS, aids_prep.EVOLUCAO_VIVO,
                   aids_prep.EVOLUCAO_CENSURA),
}


def _citados(texto, campo):
    """Códigos citados como 'CAMPO = n' ou 'CAMPO = n ou m' (com ou sem espaço)."""
    achados = re.findall(rf"\b{campo}\s*=\s*(\d+)(?:\s+ou\s+(\d+))?", texto)
    return {c for par in achados for c in par if c}


def _positivo_negativo_resto(texto, campo):
    """Códigos antes de 'contra', na frase do 'contra' e depois dela."""
    assert "contra" in texto, "o texto precisa dizer o positivo 'contra' o negativo"
    antes, depois = texto.split("contra", 1)
    frase, _, resto = depois.partition(".")
    return _citados(antes, campo), _citados(frase, campo), _citados(resto, campo)


@pytest.mark.parametrize("chave", sorted(ROTULOS_CITADOS))
@pytest.mark.parametrize("fonte", ["drawer", "descricao"])
def test_textos_citam_positivo_e_negativo_das_constantes(chave, fonte):
    campo, positivos, negativos, censura = ROTULOS_CITADOS[chave]
    if fonte == "drawer":
        texto = METHODOLOGY[chave]["target"]
    else:
        texto = OUTCOME_CLS[chave]().description
    pos, neg, resto = _positivo_negativo_resto(texto, campo)
    assert pos == set(positivos), f"{chave} ({fonte}): positivo citado {sorted(pos)}"
    assert neg == set(negativos), f"{chave} ({fonte}): negativo citado {sorted(neg)}"
    assert resto <= set(censura), f"{chave} ({fonte}): {sorted(resto - set(censura))} não é censura"


@pytest.mark.parametrize("chave", sorted(ROTULOS_CITADOS))
def test_drawer_so_cita_censura_na_coorte(chave):
    campo, _, _, censura = ROTULOS_CITADOS[chave]
    citados = _citados(METHODOLOGY[chave]["pull"], campo)
    assert citados <= set(censura), f"{chave}: o pull cita {sorted(citados - set(censura))} fora da censura"


def test_textos_do_abandono_tb_poem_o_obito_na_censura():
    meth = METHODOLOGY["abandono_tb"]
    assert _citados(meth["target"], "SITUA_ENCE") == tb_prep.SITUA_ABANDONO
    for texto in (meth["target"], AbandonoTB().description):
        # a frase dos negativos: "Cura (1) e falência (9) são negativos" ou "contra ..."
        achado = re.search(r"(?:contra [^.]*|[^.]*são negativos)", texto)
        assert achado, f"sem frase de negativos: {texto!r}"
        frase = achado.group(0)
        for cod in tb_prep.SITUA_NEGATIVO_ABANDONO:
            assert f"({cod})" in frase, f"negativo {cod} fora da frase: {frase!r}"
        assert "óbito" not in frase.lower(), f"óbito entre os negativos: {frase!r}"
        for cod in tb_prep.CENSURA_POR_ALVO["abandono"]:
            assert f"({cod})" not in frase, f"censura {cod} entre os negativos: {frase!r}"
    for cod in tb_prep.CENSURA_POR_ALVO["abandono"]:
        assert f"({cod})" in meth["pull"], f"censura {cod} fora do drawer"

