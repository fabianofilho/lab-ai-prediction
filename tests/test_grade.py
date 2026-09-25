"""Grade de desfechos (pages/datasus.py) lida por AST, sem importar streamlit."""
import ast
from pathlib import Path

import pytest

DATASUS = Path(__file__).resolve().parent.parent / "pages" / "datasus.py"


@pytest.fixture(scope="module")
def grade():
    tree = ast.parse(DATASUS.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "OUTCOME_GROUPS" for t in node.targets
        ):
            grupos = ast.literal_eval(node.value)
            return {o["key"]: o for itens in grupos.values() for o in itens}
    raise AssertionError("OUTCOME_GROUPS não encontrado em pages/datasus.py")


def test_mortalidade_neonatal_fora_do_ok(grade):
    """Sem linkage SINASC e SIM validado, o desfecho não pode aparecer como ok."""
    assert grade["mortalidade_neonatal"]["status"] != "ok"


def test_mortalidade_hospitalar_descrita_como_intra_hospitalar(grade):
    o = grade["mortalidade_hospitalar"]
    texto = f'{o["note"]} {o["linkage"]}'.lower()
    assert "intra-hospitalar" in texto
    assert "enriquece" not in texto
