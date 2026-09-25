"""Upload manual: load_from_csv lê CSV com vírgula ou ponto e vírgula."""
import pytest


@pytest.fixture
def downloader(tmp_path, monkeypatch):
    """core.data.downloader com o cache em tmp_path: nunca toca o RAW_DIR real."""
    pytest.importorskip("pyarrow")  # _save grava o cache em parquet
    mod = pytest.importorskip("core.data.downloader")  # exige requests
    monkeypatch.setattr(mod, "RAW_DIR", tmp_path)
    return mod


@pytest.mark.parametrize("sep", [";", ","])
def test_load_from_csv_le_csv_pequeno(downloader, tmp_path, sep):
    linhas = [
        sep.join(["SITUA_ENCE", "CS_SEXO", "NM_PACIENT"]),
        sep.join(["2", "M", "JOSÉ"]),
        sep.join(["1", "F", "MARIA"]),
        sep.join(["10", "F", "ANA"]),
    ]
    csv_bytes = ("\n".join(linhas) + "\n").encode("latin-1")

    df = downloader.load_from_csv(csv_bytes, "SINAN_TB", "GO", 2023)

    assert list(df.columns) == ["SITUA_ENCE", "CS_SEXO", "NM_PACIENT"]
    assert len(df) == 3
    assert df["NM_PACIENT"].tolist() == ["JOSÉ", "MARIA", "ANA"]
    assert (tmp_path / "sinan_tb_GO_2023.parquet").exists()
