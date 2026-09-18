"""DEV-1907: the pinned SLayer floor is 0.10.2 in every extra, and the
installed build is >= 0.10.2.

The surface / schema / reference-store contracts all assume 0.10.2; this is
the alarm that fires if the floor (or the synced build) drifts back.
"""

from __future__ import annotations

import pathlib
import tomllib
from importlib.metadata import version

from packaging.requirements import Requirement
from packaging.version import Version

_PYPROJECT = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
_FLOOR = Version("0.10.2")
_OLD = Version("0.9.6")
_JUST_BELOW = Version("0.10.1")


def _motley_slayer_requirements() -> list[Requirement]:
    data = tomllib.loads(_PYPROJECT.read_text(encoding="utf-8"))
    project = data["project"]
    reqs: list[str] = list(project.get("dependencies", []))
    for extra_reqs in project.get("optional-dependencies", {}).values():
        reqs.extend(extra_reqs)
    out = []
    for raw in reqs:
        req = Requirement(raw)
        if req.name == "motley-slayer":
            out.append(req)
    return out


def test_pyproject_pins_motley_slayer_at_least_twice():
    """Both the `slayer` extra and the `all` extra carry a pin (proposal Impact)."""
    reqs = _motley_slayer_requirements()
    assert len(reqs) >= 2, f"expected >=2 motley-slayer pins, found {len(reqs)}"


def test_every_pyproject_pin_floors_at_0_10_2():
    reqs = _motley_slayer_requirements()
    assert reqs, "no motley-slayer requirement found in pyproject"
    for req in reqs:
        spec = req.specifier
        assert _FLOOR in spec, f"{req} does not allow {_FLOOR}"
        assert _OLD not in spec, f"{req} still allows the old {_OLD} floor"
        # The floor is EXACTLY 0.10.2 — a `>=0.10.1` pin (which also excludes
        # 0.9.6 and allows 0.10.2) is still wrong.
        assert _JUST_BELOW not in spec, f"{req} allows {_JUST_BELOW}; floor must be {_FLOOR}"


def test_pins_carry_advanced_search_extra():
    for req in _motley_slayer_requirements():
        assert "advanced-search" in req.extras, f"{req} lost the advanced-search extra"


def test_installed_motley_slayer_is_at_least_0_10_2():
    assert Version(version("motley-slayer")) >= _FLOOR
