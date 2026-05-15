from pathlib import Path

import numpy as np

from xpolyelec.io import load_csv, load_directory, load_combined

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "examples" / "sample_data"


def test_load_single():
    arr = load_csv(SAMPLE / "kappa.csv", "kappa")
    assert arr.shape[0] == 2
    assert arr.shape[1] > 5
    assert np.all(np.isfinite(arr))


def test_load_directory():
    data = load_directory(SAMPLE)
    for prop in ("kappa", "rho_plus", "D", "U", "rho_el"):
        assert prop in data
        assert data[prop].shape[0] == 2
