"""As skills e os subagentes de .claude citam só o que existe no código.

Trava a deriva que deixou a skill datasus-pipeline descrevendo uma API que
não existia mais (SentinelReplacer(sentinel_values=...), train_cv(..., cv=...)).
Lê os módulos por AST, sem importar, para rodar no CI sem o stack de ML.

Confere, em cada .md:
- caminhos de arquivo citados entre crases (core/..., pages/..., tests/...);
- módulos citados em notação de ponto (core.data.sih);
- os imports `from core... import ...` dos blocos python;
- as chamadas a funções e classes do app: argumentos nomeados que existem
  na assinatura e quantidade de posicionais que cabe nela;
- assinaturas citadas como `funcao(a, b, c=1)` em texto corrido;
- nomes privados (`_build_model`) e chamadas (`fetch(...)`) citados entre
  crases precisam aparecer no código do app, para que uma função renomeada
  não passe calada pelas checagens acima, que só conferem o que acham.
"""
from __future__ import annotations

import ast
import re
from functools import cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DOCS = sorted((ROOT / ".claude" / "skills").glob("*/SKILL.md")) + sorted(
    (ROOT / ".claude" / "agents").glob("*.md")
)

# Módulos cujas funções e classes de topo são reconhecidas pelo nome, mesmo
# sem import no bloco (a skill costuma citar só a chamada).
CALL_MODULES = [
    "core.models.pipeline",
    "core.models.evaluation",
    "core.models.metrics",
    "core.data.downloader",
    "core.features.cohort",
    "core.features.engineering",
    "core.features.data_dict",
]

PATH_RE = re.compile(r"`((?:core|pages|tests|scripts)/[\w/.-]+\.py|app\.py)")
MOD_RE = re.compile(r"`(core(?:\.\w+)+)`")
BLOCK_RE = re.compile(r"```python\n(.*?)```", re.DOTALL)
INLINE_RE = re.compile(r"`([^`\n]+\))`")
IMPORT_RE = re.compile(r"^from (core[\w.]*) import (\([^)]*\)|[^\n]+)", re.MULTILINE)
SIGNATURE_RE = re.compile(r"^(\w+)\((.*)\)$")
NAME_SPAN_RE = re.compile(r"`(_?[A-Za-z]\w*)(\(.*?\))?`")


def _module_file(mod: str) -> Path | None:
    base = ROOT.joinpath(*mod.split("."))
    for cand in (base.with_suffix(".py"), base / "__init__.py"):
        if cand.exists():
            return cand
    return None


@cache
def _top_names(mod: str) -> dict[str, ast.AST]:
    """Nomes definidos no topo do módulo: def, class, atribuição e import."""
    tree = ast.parse(_module_file(mod).read_text(encoding="utf-8"))
    names: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if isinstance(t, ast.Name):
                    names[t.id] = node
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                names[(alias.asname or alias.name).split(".")[0]] = node
    return names


@cache
def _callables_by_name() -> dict[str, list[tuple[str, ast.AST]]]:
    found: dict[str, list[tuple[str, ast.AST]]] = {}
    for mod in CALL_MODULES:
        for name, node in _top_names(mod).items():
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                found.setdefault(name, []).append((mod, node))
    return found


def _signature(node: ast.AST):
    """(nomes aceitos, n de posicionais, *args?, **kwargs?) ou None."""
    skip = 0
    if isinstance(node, ast.ClassDef):
        node = next(
            (n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"),
            None,
        )
        skip = 1  # self
    if not isinstance(node, ast.FunctionDef):
        return None  # dataclass sem __init__ explícito, reexport etc.
    a = node.args
    positional = [x.arg for x in a.posonlyargs + a.args][skip:]
    names = set(positional) | {x.arg for x in a.kwonlyargs}
    return names, len(positional), a.vararg is not None, a.kwarg is not None


def _check_call(call: ast.Call, label: str, node: ast.AST) -> list[str]:
    sig = _signature(node)
    if sig is None:
        return []
    names, n_pos, varargs, varkw = sig
    errors = []
    if not varargs and len(call.args) > n_pos:
        errors.append(f"{label}: {len(call.args)} posicionais, a assinatura aceita {n_pos}")
    if not varkw:
        for kw in call.keywords:
            if kw.arg is not None and kw.arg not in names:
                errors.append(f"{label}: parâmetro {kw.arg!r} não existe")
    return errors


def _unique(name: str):
    hits = _callables_by_name().get(name, [])
    return hits[0] if len(hits) == 1 else None


def _check_imports(tree: ast.AST, bound: dict, modules: dict) -> list[str]:
    errors = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("core")):
            continue
        if _module_file(node.module) is None:
            errors.append(f"módulo {node.module} não existe")
            continue
        for alias in node.names:
            local = alias.asname or alias.name
            sub = f"{node.module}.{alias.name}"
            if _module_file(sub) is not None:
                modules[local] = sub
            elif alias.name in _top_names(node.module):
                bound[local] = (node.module, _top_names(node.module)[alias.name])
            else:
                errors.append(f"{alias.name} não existe em {node.module}")
    return errors


@cache
def _app_source() -> str:
    files = [ROOT / "app.py"]
    for sub in ("core", "pages", "tests", "scripts"):
        files += sorted((ROOT / sub).rglob("*.py"))
    this = Path(__file__).resolve()
    return "\n".join(
        f.read_text(encoding="utf-8") for f in files if f.exists() and f.resolve() != this
    )


def _check_names(text: str) -> list[str]:
    """Nome privado ou chamada citada entre crases que não aparece no código."""
    src = _app_source()
    errors = []
    for name, call in set(NAME_SPAN_RE.findall(text)):
        if (name.startswith("_") or call) and not re.search(rf"\b{re.escape(name)}\b", src):
            errors.append(f"{name} não aparece no código do app")
    return errors


def _check_block(code: str) -> tuple[list[str], int]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # bloco-modelo com marcadores ({chave}): confere só os imports do app
        errors: list[str] = []
        for mod, names in IMPORT_RE.findall(code):
            stmt = f"from {mod} import {names}"
            try:
                errors += _check_imports(ast.parse(stmt), {}, {})
            except SyntaxError:
                continue
        return errors, 0

    bound: dict = {}
    modules: dict = {}
    errors = _check_imports(tree, bound, modules)
    local_defs = {
        n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))
    }
    checked = 0
    for call in (n for n in ast.walk(tree) if isinstance(n, ast.Call)):
        func = call.func
        target = None
        if isinstance(func, ast.Name) and func.id not in local_defs:
            if func.id in bound:
                target = bound[func.id]
            else:
                target = _unique(func.id)
            label = func.id
        elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
            mod = modules.get(func.value.id)
            if mod is None:
                continue
            node = _top_names(mod).get(func.attr)
            if node is None:
                errors.append(f"{func.attr} não existe em {mod}")
                continue
            target = (mod, node)
            label = f"{func.value.id}.{func.attr}"
        if target is not None:
            checked += 1
            errors += _check_call(call, label, target[1])
    return errors, checked


def _check_inline(span: str) -> tuple[list[str], int]:
    try:
        expr = ast.parse(span, mode="eval").body
    except SyntaxError:
        # assinatura citada, ex. `f(X, y, algorithm, n_trials=50, balancing)`
        m = SIGNATURE_RE.match(span)
        hit = _unique(m.group(1)) if m else None
        sig = _signature(hit[1]) if hit else None
        if sig is None:
            return [], 0
        names, _, _, varkw = sig
        tokens = [t.split("=")[0].strip() for t in m.group(2).split(",") if t.strip()]
        if varkw or not all(t.isidentifier() for t in tokens):
            return [], 0
        bad = [t for t in tokens if t not in names]
        return [f"{m.group(1)}: parâmetro {t!r} não existe" for t in bad], 1
    if isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name):
        hit = _unique(expr.func.id)
        if hit is not None:
            return _check_call(expr, expr.func.id, hit[1]), 1
    return [], 0


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_doc_cita_so_o_que_existe(doc):
    text = doc.read_text(encoding="utf-8")
    errors = []

    for path in set(PATH_RE.findall(text)):
        if not (ROOT / path).exists():
            errors.append(f"arquivo {path} não existe")
    for mod in set(MOD_RE.findall(text)):
        if _module_file(mod) is None:
            errors.append(f"módulo {mod} não existe")
    for code in BLOCK_RE.findall(text):
        errs, _ = _check_block(code)
        errors += errs
    for span in INLINE_RE.findall(text):
        errs, _ = _check_inline(span)
        errors += errs
    errors += _check_names(text)

    assert not errors, f"{doc.relative_to(ROOT)}:\n" + "\n".join(sorted(set(errors)))


def test_skill_do_pipeline_tem_chamadas_conferidas():
    """Garante que o teste acima não passa por não achar nada para conferir."""
    text = (ROOT / ".claude" / "skills" / "datasus-pipeline" / "SKILL.md").read_text(
        encoding="utf-8"
    )
    checked = sum(_check_block(code)[1] for code in BLOCK_RE.findall(text))
    checked += sum(_check_inline(span)[1] for span in INLINE_RE.findall(text))
    assert checked >= 10


def test_verificador_pega_a_api_antiga():
    """As chamadas que a skill antiga citava precisam falhar aqui."""
    antigas = (
        "SentinelReplacer(sentinel_values=[9, 99])\n"
        "train_cv(X, y, pipeline, cv=StratifiedKFold(5, shuffle=True, random_state=42))\n"
    )
    errors, checked = _check_block(antigas)
    assert checked == 2
    assert any("sentinel_values" in e for e in errors)
    assert any("'cv'" in e for e in errors)
    assert _check_block("from core.models.pipeline import calibrar\n")[0]
    assert _check_names('`dbc_to_parquet("arquivo.dbc", "saida.parquet")`')
    assert _check_names("`_calibrar_modelo`")
    assert not _check_names("`calibrate_model(model, X, y)` e `_compute_metrics`")
