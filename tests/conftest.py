import pytest

from alglory.data.sample import generate_bars


@pytest.fixture(scope="session")
def bars_h1():
    return generate_bars("EURUSD", "H1", 5000, seed=7)
