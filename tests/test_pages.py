"""Every page renders, with adjustments off and with all of them on.

This is the check that catches the breakages unit tests miss: a renamed helper, a column that no
longer exists, a matrix that comes back empty. It runs against the seeded database, so it works on a
fresh clone and cannot be fooled by whatever happens to be downloaded.
"""
import re
import warnings
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from market import adjust, sandbox
from tests.conftest import SEED

ROOT = Path(__file__).resolve().parent.parent  # AppTest resolves relative paths against this file

PAGES = ["pages/overview.py", "pages/asset_classes.py", "pages/screener.py", "pages/compare.py",
         "pages/portfolio.py", "pages/correlation.py", "pages/statistics.py", "pages/scenarios.py",
         "pages/data_manager.py"]
STATE_KEYS = ["range", "freq", "kind", "currency", "basket", "settings"]
TIMEOUT = 120


@pytest.fixture
def session(seeded_db):
    """Settings built the way the app builds them, ready to hand to a page."""
    warnings.simplefilter("ignore")

    def state(adjustments_on: bool) -> dict:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT).run()
        assert not app.exception, "the router itself must boot"
        for key in adjust.ADJUSTMENTS:
            app.session_state[f"adj_{key}"] = adjustments_on
        app.run()
        keys = STATE_KEYS + [f"adj_{k}" for k in adjust.ADJUSTMENTS]
        return {k: app.session_state[k] for k in keys}

    return state


def render(page: str, state: dict) -> AppTest:
    at = AppTest.from_file(str(ROOT / page), default_timeout=TIMEOUT)
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


@pytest.mark.slow
@pytest.mark.parametrize("page", PAGES)
@pytest.mark.parametrize("adjustments_on", [False, True], ids=["plain", "adjusted"])
def test_page_renders(session, page, adjustments_on):
    state = session(adjustments_on)
    assert state["settings"].adjust == (frozenset(adjust.ADJUSTMENTS) if adjustments_on else frozenset())
    at = render(page, state)
    assert not at.exception, f"{page}: {at.exception[0].value if at.exception else ''}"
    assert at.title or at.markdown, f"{page} rendered nothing"


def test_the_pages_read_the_seeded_database(seeded_db):
    """Isolation is the thing most likely to fail silently: a warm cache or an unpatched path would
    serve the real database and every page test would pass while proving nothing."""
    from market import ui

    assert len(ui.instruments()) == len(SEED), "the pages must see the seeded universe, not the real one"
    assert set(ui.instruments().ticker) == {row[0] for row in SEED}
    assert str(seeded_db.DB_PATH) != str(Path("data/market.duckdb").resolve())


def test_every_page_in_the_router_is_covered():
    """A page added to the navigation but not to PAGES would otherwise go untested."""
    routed = set(re.findall(r'"(pages/[\w_]+\.py)"', (ROOT / "app.py").read_text(encoding="utf-8")))
    assert routed - {"pages/code_lab.py"} == set(PAGES)


def test_code_lab_runs_its_own_examples(seeded_db):
    """The Code Lab page cannot be imported directly — pages/ shadows the standard library's
    `statistics` module — so exercise the engine behind it instead."""
    result = sandbox.run('print(prices(["^AXJO", "BHP.AX"]).shape)', sandbox.namespace(basket=["^AXJO"]))
    assert result.error is None, result.error
    assert "(" in result.stdout, result.stdout
