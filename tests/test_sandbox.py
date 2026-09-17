import matplotlib.figure
import pandas as pd
import plotly.graph_objects as go

from market import sandbox


def run(code: str) -> sandbox.Run:
    return sandbox.run(code, sandbox.namespace(["X"]))


def test_stdout_and_last_expression():
    result = run("print('hi')\nx = 2\nx * 21")
    assert result.stdout == "hi\n" and result.outputs == [42] and result.error is None


def test_show_collects_figures_and_frames():
    result = run("show(go.Figure())\nshow(pd.DataFrame({'a': [1]}))\nbasket")
    assert isinstance(result.outputs[0], go.Figure)
    assert isinstance(result.outputs[1], pd.DataFrame)
    assert result.outputs[2] == ["X"]


def test_matplotlib_figures_collected_once():
    result = run("fig, ax = plt.subplots()\nax.plot([1, 2])\nax")
    assert len(result.outputs) == 1 and isinstance(result.outputs[0], matplotlib.figure.Figure)


def test_runtime_error_reports_user_line():
    result = run("a = 1\nb = 0\nc = a / b")
    assert result.error.startswith("Line 3: c = a / b") and "ZeroDivisionError" in result.error


def test_syntax_error():
    result = run("x = (1,\ny = 2")
    assert "SyntaxError" in result.error and result.outputs == []


def test_snippets_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "SNIPPETS_DIR", tmp_path)
    sandbox.save_snippet("my test", "1 + 1")
    assert sandbox.snippets() == {"my test": "1 + 1"}
    sandbox.delete_snippet("my test")
    assert sandbox.snippets() == {}
