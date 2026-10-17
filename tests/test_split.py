import numpy as np
import pandas as pd
import pytest

from alglory.evolve.split import split_bars


def _bars(n):
    return pd.DataFrame({"time": np.arange(n), "close": np.arange(n, dtype=float)})


def test_split_is_chronological_60_20_20():
    breed, select, holdout = split_bars(_bars(1000))
    assert (len(breed), len(select), len(holdout)) == (600, 200, 200)
    assert list(holdout["time"]) == list(range(800, 1000))
    assert list(select["time"]) == list(range(600, 800))
    assert list(breed["time"]) == list(range(600))
    for part in (breed, select, holdout):
        assert list(part.index) == list(range(len(part)))


def test_split_rejects_oversized_fractions():
    with pytest.raises(ValueError):
        split_bars(_bars(1000), select_fraction=0.5, holdout_fraction=0.4)
