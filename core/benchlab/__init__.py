"""BenchLab: comparativo global de algoritmos × datasets."""
from .runner import BENCH_ALGORITHMS, BENCHLAB_CACHE_PATH, load_cache, run_benchlab, save_cache

__all__ = [
    "BENCH_ALGORITHMS",
    "BENCHLAB_CACHE_PATH",
    "load_cache",
    "run_benchlab",
    "save_cache",
]
