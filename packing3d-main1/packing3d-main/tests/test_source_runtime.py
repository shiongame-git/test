"""Source-only execution, code identity and supported record boundaries."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from packing import (
    InputFormatError, read_run_metadata_json, read_comparison_report_json,
    read_execution_records,
)
from packing.research_records import make_provenance
from packing.source import source_revision

ROOT = Path(__file__).resolve().parents[1]


def test_source_runner_uses_selected_checkout_with_real_spawn(tmp_path):
    script = tmp_path / "run_research.py"
    script.write_text('''from pathlib import Path
import sys
import packing
from packing import Box, Container, GAConfig, WorkersRequest
from packing import prepare_research_experiment, run_research_experiment, read_research_experiment

if __name__ == "__main__":
    plan = prepare_research_experiment(
        Container(2, 2, 2), (Box(1, 1, 1, 1),), record_type="PP-GA",
        config=GAConfig(2, 2), start_seed=0, run_count=1,
        workers_request=WorkersRequest("manual", 2), save_level="full",
    )
    result = run_research_experiment(plan, sys.argv[1])
    assert result["status"] == "completed", result["error"]
    directory = Path(sys.argv[1]) / result["experiment_id"]
    saved = read_research_experiment(directory)
    assert saved["provenance"]["distribution_version"] == "source"
    assert saved["provenance"]["git_head"]
    print(packing.__file__)
''', encoding="utf-8")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "run.py"), str(script), str(tmp_path / "results")],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert str(ROOT / "src" / "packing" / "__init__.py") in result.stdout


def test_provenance_uses_source_identity_without_distribution_lookup(monkeypatch):
    import importlib.metadata

    def forbidden(*args, **kwargs):
        raise AssertionError("Distribution metadata is not part of source execution")

    monkeypatch.setattr(importlib.metadata, "version", forbidden)
    provenance = make_provenance()
    assert provenance["distribution_version"] == "source"
    assert provenance["git_head"] == source_revision()
    assert not any("distribution" in reason for reason in provenance["unavailable"])


def test_source_runner_constructs_real_tk_gui(tmp_path):
    script = tmp_path / "gui_smoke.py"
    script.write_text(f'''import runpy
import sys
import tkinter as tk
sys.path.insert(0, {str(ROOT / "demo")!r})
namespace = runpy.run_path({str(ROOT / "demo/gui.pyw")!r})
root = tk.Tk()
root.withdraw()
try:
    app = namespace["PackingDemoApp"](root)
    root.update_idletasks()
    assert app.candidate_policy_var.get() == "Cartesian"
    assert app.min_support_percent_var.get() == "55"
    print("Tk GUI constructed")
finally:
    root.destroy()
''', encoding="utf-8")
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", str(ROOT / "run.py"), str(script)],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Tk GUI constructed" in result.stdout


@pytest.mark.parametrize("python_version,platform", [((3, 11), "win32"), ((3, 14), "win32"), ((3, 13), "linux")])
def test_source_runner_rejects_unsupported_runtime(monkeypatch, python_version, platform):
    spec = importlib.util.spec_from_file_location("source_runner", ROOT / "run.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    monkeypatch.setattr(runner.sys, "version_info", python_version)
    monkeypatch.setattr(runner.sys, "platform", platform)
    with pytest.raises(SystemExit, match="Windows and Python 3.13"):
        runner.main()


def test_retired_metadata_is_rejected_without_output(tmp_path):
    source = ROOT / "tests/fixtures/run_metadata_v2.json"
    data = json.loads(source.read_text())
    data["version"] = 1
    path = tmp_path / "unsupported.json"
    path.write_text(json.dumps(data))
    before = path.read_bytes()
    for reader in (read_run_metadata_json, read_execution_records):
        with pytest.raises(InputFormatError, match="unsupported run metadata version"):
            reader(path)
    assert path.read_bytes() == before


@pytest.mark.parametrize("version", [2, 3])
def test_retired_comparison_is_rejected(tmp_path, version):
    source = ROOT / "tests/fixtures/comparison_v4.json"
    data = json.loads(source.read_text())
    data["version"] = version
    path = tmp_path / "unsupported.json"
    path.write_text(json.dumps(data))
    for reader in (read_comparison_report_json, read_execution_records):
        with pytest.raises(InputFormatError, match="unsupported comparison version"):
            reader(path)
