"""BenchLab runner: roda N algoritmos x M datasets com CV e retorna tabela de metricas.

Contrato de retorno de run_benchlab():
    {
        "results":  {bench_key: {algo_key: mean_metrics_dict}},
        "errors":   [(bench_key, algo_key, error_str), ...],
        "metadata": {"n_folds": int, "n_datasets": int, "n_algorithms": int},
    }
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

import pandas as pd

from core.models.pipeline import ALGORITHMS, train_cv

if TYPE_CHECKING:
    from core.data.benchmarks import Benchmark

BENCHLAB_CACHE_PATH = Path(__file__).parent.parent.parent / "benchlab_results.json"

# Exclui TabPFN — opcional e pesado (requer torch ~1GB)
BENCH_ALGORITHMS: dict[str, str] = {k: v for k, v in ALGORITHMS.items() if v != "tabpfn"}


def _load_bench(bench: "Benchmark") -> tuple[pd.DataFrame, pd.Series]:
    df = bench.loader()
    y = df[bench.target_col]
    X = df.drop(columns=[bench.target_col])
    return X, y


def run_benchlab(
    benchmarks: list["Benchmark"],
    algorithm_keys: Optional[list[str]] = None,
    n_folds: int = 5,
    progress_cb: Optional[Callable[[str, float], None]] = None,
) -> dict:
    """Roda benchmark: cada algoritmo em cada dataset com CV padrao.

    Usa params default, sem HPO, sem SMOTE — comparacao justa de base.
    """
    algo_keys = algorithm_keys or list(BENCH_ALGORITHMS.values())
    algo_display = {v: k for k, v in ALGORITHMS.items()}

    results: dict[str, dict[str, dict]] = {}
    errors: list[tuple[str, str, str]] = []
    total = len(benchmarks) * len(algo_keys)
    done = 0

    for bench in benchmarks:
        try:
            X, y = _load_bench(bench)
        except Exception as exc:
            for ak in algo_keys:
                errors.append((bench.key, ak, f"Erro ao carregar: {exc}"))
                done += 1
            continue

        bench_row: dict[str, dict] = {}
        for algo_key in algo_keys:
            algo_name = algo_display.get(algo_key, algo_key)
            if progress_cb:
                progress_cb(f"{bench.name}  x  {algo_name}", done / total)
            try:
                res = train_cv(X, y, algorithm=algo_key, n_folds=n_folds)
                bench_row[algo_key] = res["mean_metrics"]
            except Exception as exc:
                errors.append((bench.key, algo_key, str(exc)))
            done += 1

        results[bench.key] = bench_row

    if progress_cb:
        progress_cb("Concluido", 1.0)

    return {
        "results": results,
        "errors": errors,
        "metadata": {
            "n_folds": n_folds,
            "n_datasets": len(benchmarks),
            "n_algorithms": len(algo_keys),
        },
    }


def save_cache(bench_result: dict) -> None:
    out = dict(bench_result)
    out["errors"] = [list(e) for e in bench_result["errors"]]
    with open(BENCHLAB_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def load_cache() -> Optional[dict]:
    if not BENCHLAB_CACHE_PATH.exists():
        return None
    try:
        with open(BENCHLAB_CACHE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        data["errors"] = [tuple(e) for e in data.get("errors", [])]
        return data
    except Exception:
        return None
