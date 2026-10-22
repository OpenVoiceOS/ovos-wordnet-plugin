"""The built wheel must carry every locale file the engine loads at runtime."""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
PACKAGE = "ovos_wordnet_plugin"


@pytest.fixture(scope="module")
def wheel_names(tmp_path_factory):
    out = tmp_path_factory.mktemp("wheel")
    tree = out / "tree"
    tree.mkdir()
    shutil.copy(REPO / "pyproject.toml", tree)
    shutil.copy(REPO / "README.md", tree)
    shutil.copytree(REPO / PACKAGE, tree / PACKAGE,
                    ignore=shutil.ignore_patterns("__pycache__"))
    code = (
        "import setuptools.build_meta as b, sys; "
        "b.build_wheel(sys.argv[1])"
    )
    dist = out / "dist"
    subprocess.run([sys.executable, "-c", code, str(dist)], cwd=tree, check=True,
                   capture_output=True)
    wheels = list(dist.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as zf:
        return set(zf.namelist())


def test_wheel_ships_every_locale_file(wheel_names):
    source = {p.relative_to(REPO).as_posix()
              for p in (REPO / PACKAGE / "locale").rglob("*") if p.is_file()}
    assert len(source) > 100
    assert source - wheel_names == set()


def test_wheel_ships_readme_example_intent(wheel_names):
    assert f"{PACKAGE}/locale/en-US/hyponym.intent" in wheel_names
