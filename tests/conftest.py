from pathlib import Path

import pytest

from brickcore.catalog import load_catalog
from brickcore.schema import Build

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def catalog():
    return load_catalog()


@pytest.fixture(scope="session")
def two_brick() -> Build:
    return Build.from_json((ROOT / "fixtures" / "builds" / "two_brick.json").read_text())


@pytest.fixture(scope="session")
def fixture_meshes(tmp_path_factory) -> Path:
    from brickcore.fixtures import write_fixtures
    out = tmp_path_factory.mktemp("meshes")
    write_fixtures(out)
    return out
