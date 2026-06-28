"""Pytest config: garante o repo root no sys.path e registra marcadores."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "network: testes que baixam dados reais do DATASUS (rede). "
        "Pulados por padrão; rode com RUN_NETWORK_TESTS=1.",
    )
