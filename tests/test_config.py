import json
from pathlib import Path

from xpolyelec.config import Config


def test_default_loads():
    cfg = Config.default(verbose=False)
    assert cfg.get("physical.T_K") == 333.15
    assert cfg.get("strain_model.name") == "none"


def test_dotted_set_tracks_change(capsys):
    cfg = Config.default(verbose=True)
    cfg.set("strain_model.N_monomers_per_strand", 20)
    assert cfg.get("strain_model.N_monomers_per_strand") == 20
    out = capsys.readouterr().out
    assert "strain_model.N_monomers_per_strand" in out
    assert "16" in out and "20" in out


def test_save_non_interactive(tmp_path: Path):
    cfg = Config.default(verbose=False)
    cfg.set("physical.ravg", 0.12)
    path = tmp_path / "out.json"
    written = cfg.save(path)
    assert written == path
    data = json.loads(path.read_text())
    assert data["physical"]["ravg"] == 0.12
