"""BenchLab: samples brasileiros com rótulo inválido ficam fora do comparativo."""
import json
from pathlib import Path

import pytest

from core.data.benchmarks import BENCHMARKS

_MANIFEST = json.loads(
    (Path(__file__).resolve().parent.parent
     / "core" / "data" / "benchmarks" / "data" / "br" / "manifest.json").read_text(encoding="utf-8")
)

# Gerados com SITUA_ENCE trocado; só voltam depois de regenerados.
INVALIDOS = ["abandono_tb", "obito_tb"]


@pytest.mark.parametrize("key", INVALIDOS)
def test_manifest_marca_rotulo_invalido(key):
    assert _MANIFEST[key].get("rotulo_invalido") is True
    assert _MANIFEST[key].get("aviso")


@pytest.mark.parametrize("key", INVALIDOS)
def test_benchlab_nao_usa_sample_com_rotulo_invalido(key):
    b = BENCHMARKS[f"bench_br_{key}"]
    assert b.status != "ok"
    assert b.loader is None
    assert "INDISPONÍVEL" in b.note
    # mesmo filtro de pages/benchlab.py (OK_BENCHES)
    ok = [x for x in BENCHMARKS.values() if x.status == "ok" and x.loader is not None]
    assert b not in ok


def test_demais_samples_br_seguem_disponiveis():
    validos = [k for k, m in _MANIFEST.items() if not m.get("rotulo_invalido")]
    assert validos
    for k in validos:
        b = BENCHMARKS[f"bench_br_{k}"]
        assert b.status == "ok" and callable(b.loader), k
