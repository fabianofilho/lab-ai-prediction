"""Invariantes estruturais dos desfechos (sem rede, determinístico).

Trava as regressões corrigidas na auditoria: anti-leakage por desfecho,
cobertura de metodologia, consistência do registry e dos grupos.
"""
import pytest

from core.outcomes import OUTCOMES, OUTCOME_GROUPS
from core.methodology import METHODOLOGY
from core.data.benchmarks import BENCHMARKS, BENCHMARK_GROUPS


# Colunas que, se aparecerem em suggested_features, vazam o alvo do desfecho.
# Cada bug de leakage corrigido vira uma linha aqui.
LEAK_BLACKLIST = {
    "baixo_peso_nascer":        ["PESO", "low_birth_weight"],
    "prematuridade":            ["GESTACAO", "preterm"],
    "apgar_baixo":              ["APGAR5", "APGAR1", "low_apgar5"],
    "cesarea":                  ["PARTO", "STCESPARTO"],
    "anomalia_congenita":       ["IDANOMAL", "CODANOMAL"],
    "mortalidade_neonatal":     ["neonatal_death"],
    "permanencia_prolongada":   ["length_of_stay_days"],  # é o alvo, não feature
    "uso_uti":                  ["VAL_TOT", "VAL_UTI", "UTI_MES_TO", "used_icu", "MARCA_UTI"],
    "infeccao_hospitalar":      ["DIAGSEC1"],
    "custo_elevado":            ["VAL_TOT"],
    "mortalidade_hospitalar":   ["MORTE", "is_death", "linked_death"],
    "readmissao_30d":           ["_patient_key"],
    "abandono_tb":              ["SITUA_ENCE"],
    "obito_tb":                 ["SITUA_ENCE", "abandono", "cura"],
    "abandono_hanseniase":      ["TPALTA_N", "SITUA_ENCE"],
    "incapacidade_hanseniase":  ["AVALIA_N", "grau_incapacidade", "TPALTA_N"],
    "violencia_autoprovocada":  ["LES_AUTOP", "CONS_SUIC", "VIOL_AUTO"],
    "intoxicacao_grave":        ["EVOLUCAO"],
    "dengue_grave":             ["CLASSI_FIN", "EVOLUCAO"],
    "chikungunya_hospitalizado":["HOSPITALIZ", "EVOLUCAO"],
    "obito_aids":               ["EVOLUCAO"],
    "sifilis_nao_cura":         ["EVOLUCAO"],
}

ALL_KEYS = sorted(OUTCOMES.keys())


@pytest.mark.parametrize("key", ALL_KEYS)
def test_outcome_instantiates(key):
    """Cada desfecho instancia (carrega a classe real) sem erro."""
    oc = OUTCOMES[key]
    oc._load()
    assert oc.target_col, f"{key} sem target_col"
    assert isinstance(oc.suggested_features, list)


@pytest.mark.parametrize("key", ALL_KEYS)
def test_target_not_in_features(key):
    """O target_col nunca pode estar entre as features (vazamento direto)."""
    oc = OUTCOMES[key]
    assert oc.target_col not in (oc.suggested_features or []), (
        f"{key}: target_col '{oc.target_col}' está em suggested_features (leakage)"
    )


@pytest.mark.parametrize("key", ALL_KEYS)
def test_no_known_leak_columns(key):
    """Nenhuma coluna-fonte conhecida do alvo pode aparecer nas features."""
    oc = OUTCOMES[key]
    feats = set(oc.suggested_features or [])
    leaks = feats.intersection(LEAK_BLACKLIST.get(key, []))
    assert not leaks, f"{key}: features vazam o alvo: {sorted(leaks)}"


@pytest.mark.parametrize("key", ALL_KEYS)
def test_has_methodology(key):
    """Todo desfecho tem entrada de metodologia (pull + target)."""
    assert key in METHODOLOGY, f"{key} sem entrada em METHODOLOGY"
    m = METHODOLOGY[key]
    assert m.get("pull"), f"{key}: metodologia sem 'pull'"
    assert m.get("target"), f"{key}: metodologia sem 'target'"


def test_methodology_has_no_orphans():
    """METHODOLOGY não tem chaves que não existem em OUTCOMES."""
    orphans = set(METHODOLOGY) - set(OUTCOMES)
    assert not orphans, f"METHODOLOGY com chaves órfãs: {sorted(orphans)}"


def test_groups_match_registry():
    """Todo desfecho dos grupos existe no registry e vice-versa."""
    grouped = {k for keys in OUTCOME_GROUPS.values() for k in keys}
    assert grouped == set(OUTCOMES), (
        f"divergência grupos x registry: "
        f"só nos grupos={sorted(grouped - set(OUTCOMES))}, "
        f"só no registry={sorted(set(OUTCOMES) - grouped)}"
    )


def test_no_duplicate_keys_in_groups():
    """Nenhum desfecho aparece em mais de um grupo."""
    seen = []
    for keys in OUTCOME_GROUPS.values():
        seen.extend(keys)
    dups = {k for k in seen if seen.count(k) > 1}
    assert not dups, f"desfechos duplicados nos grupos: {sorted(dups)}"


# ── Benchmarks ────────────────────────────────────────────────────────────────

def test_benchmark_groups_consistent():
    flat = {b.key for g in BENCHMARK_GROUPS.values() for b in g}
    assert flat == set(BENCHMARKS), "BENCHMARK_GROUPS x BENCHMARKS divergem"


@pytest.mark.parametrize("bkey", sorted(BENCHMARKS.keys()))
def test_benchmark_ok_has_loader(bkey):
    """Benchmark com status 'ok' precisa de loader chamável e target_col."""
    b = BENCHMARKS[bkey]
    if b.status == "ok":
        assert callable(b.loader), f"{bkey}: status ok sem loader"
        assert b.target_col, f"{bkey}: sem target_col"
