"""Static checks on the vault UI: what it calls evidence, and what it sorts by."""

import re
from pathlib import Path

from alglory.vault.db import _SORTABLE

UI = Path(__file__).resolve().parents[1] / "alglory" / "ui"
VAULT_JS = (UI / "js" / "vault.js").read_text(encoding="utf-8")
INDEX = (UI / "index.html").read_text(encoding="utf-8")


def test_select_figures_not_presented_as_forward_test():
    assert "OUT-OF-SAMPLE (forward test)" not in VAULT_JS
    assert "OOS EQUITY" not in VAULT_JS
    assert "SELECT (used for selection)" in VAULT_JS
    assert "SELECT EQUITY (used for selection)" in VAULT_JS


def test_detail_panel_shows_holdout_metrics():
    detail = VAULT_JS[VAULT_JS.index("async function showDetail"):]
    assert "holdoutDetail(row)" in detail
    fn = VAULT_JS[VAULT_JS.index("function holdoutDetail"):]
    fn = fn[: fn.index("\n}\n")]
    for label in ("WIN RATE", "PROFIT FACTOR", "TRADES", "NET PROFIT"):
        assert label in fn
    assert "legacy — no holdout" in fn


def test_reused_marker_in_table_and_detail():
    assert "holdout reused — results are weaker evidence" in VAULT_JS
    assert VAULT_JS.count("reusedMark(") >= 3  # definition + table + detail
    assert ">REUSED<" in VAULT_JS


def test_demo_line_marks_legacy_rows():
    detail = VAULT_JS[VAULT_JS.index("async function showDetail"):]
    assert "demoLine(row)" in detail
    fn = VAULT_JS[VAULT_JS.index("function demoLine"):]
    fn = fn[: fn.index("\n}\n")]
    assert "legacy — no holdout" in fn


def test_default_sort_is_holdout_and_server_allowed():
    select = INDEX[INDEX.index('<select id="vf-sort">'):]
    select = select[: select.index("</select>")]
    default = re.search(r'<option value="([a-z_]+)" selected>', select)
    assert default and default.group(1) == "ho_win_rate"
    for value in re.findall(r'<option value="([a-z_]+)"', select):
        assert value in _SORTABLE
