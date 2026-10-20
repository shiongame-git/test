from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from packing import read_decoder_profile_json


def test_compare_ga_report_v4_contains_problem_identity_and_orders(tmp_path: Path):
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("{3,2,1}{1,1,1}{1,1,1}", encoding="utf-8")
    output = tmp_path / "report.json"
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks" / "compare_ga.py"),
            str(fixture),
            "--decoder",
            "BL",
            "--order",
            "depth_desc_stable",
            "--population",
            "1",
            "--generations",
            "1",
            "--crossover",
            "0",
            "--mutation",
            "0",
            "--output",
            str(output),
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["version"] == 4
    metadata = report["trials"][0]["run_metadata"]
    assert metadata["version"] == 2
    assert metadata["decoder"]["reference_version"] == "BL-ref-v2"
    trial = metadata["problem"]
    assert trial["source"] == str(fixture)
    assert trial["fixture_sha256"] == hashlib.sha256(fixture.read_bytes()).hexdigest()
    assert {k: trial[k] for k in ("container", "source_boxes")} == {
        "container": [3, 2, 1],
        "source_boxes": [[1, 1, 1, 1], [2, 1, 1, 1]],
    }
    assert trial["source_box_order"] == [1, 2]
    assert trial["effective_box_order"] == [1, 2]
    assert metadata["decoder"]["candidate_policy"] == "Cartesian"
    assert metadata["decoder"]["candidate_policy_version"] == 1
    assert metadata["decoder"]["min_support_percent"] == 55
    assert metadata["decoder"]["support_model_version"] == "support-union-v1"


def test_compare_ga_cli_runs_both_decoders_with_local_neighbors_and_writes_v4(
    tmp_path: Path,
):
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("{3,2,2}{1,1,1}{1,1,1}{1,1,1}", encoding="utf-8")
    output = tmp_path / "local-report.json"
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks" / "compare_ga.py"),
            str(fixture),
            "--decoder",
            "both",
            "--candidate-policy",
            "LocalNeighbors",
            "--min-support-percent",
            "55",
            "--priority",
            "BackFirst",
            "--population",
            "2",
            "--generations",
            "2",
            "--crossover",
            "0",
            "--mutation",
            "0",
            "--seeds",
            "13",
            "--output",
            str(output),
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["version"] == 4
    assert [trial["run_metadata"]["decoder"]["name"] for trial in report["trials"]] == [
        "BL",
        "PP",
    ]
    for trial in report["trials"]:
        decoder = trial["run_metadata"]["decoder"]
        assert decoder["candidate_policy"] == "LocalNeighbors"
        assert decoder["candidate_policy_version"] == 1
        assert decoder["min_support_percent"] == 55
        assert decoder["support_model_version"] == "support-union-v1"
        assert decoder["priority"] == "BackFirst"


def test_compare_ga_cli_preserves_explicit_zero_support(tmp_path: Path):
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("{2,1,1}{1,1,1}", encoding="utf-8")
    output = tmp_path / "zero-support-report.json"
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks" / "compare_ga.py"),
            str(fixture),
            "--decoder", "BL",
            "--min-support-percent", "0",
            "--population", "1",
            "--generations", "1",
            "--crossover", "0",
            "--mutation", "0",
            "--output", str(output),
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["trials"][0]["run_metadata"]["decoder"]["min_support_percent"] == 0


def test_compare_ga_cli_writes_run_linked_profiles_outside_report(tmp_path: Path):
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("{3,2,2}{1,1,1}{1,1,1}{1,1,1}", encoding="utf-8")
    output = tmp_path / "report.json"
    profile_dir = tmp_path / "profiles"
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks" / "compare_ga.py"),
            str(fixture),
            "--decoder",
            "both",
            "--candidate-policy",
            "LocalNeighbors",
            "--min-support-percent",
            "55",
            "--population",
            "2",
            "--generations",
            "2",
            "--crossover",
            "0.8",
            "--mutation",
            "0.1",
            "--output",
            str(output),
            "--profile-dir",
            str(profile_dir),
        ],
        cwd=root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output.read_text(encoding="utf-8"))
    assert set(report) == {"format", "version", "implementation_version", "trials"}
    assert len(report["trials"]) == 2
    for trial in report["trials"]:
        assert set(trial) == {"run_metadata", "best"}
        metadata = trial["run_metadata"]
        run_id = metadata["run_id"]
        profile_path = profile_dir / f"{run_id}.json"
        assert profile_path.is_file()
        profile = read_decoder_profile_json(profile_path)
        assert profile.run_id == run_id
        assert profile.decoder == metadata["decoder"]["name"]
        assert profile.decoder_config.candidate_policy == "LocalNeighbors"
        assert profile.decoder_config.min_support_percent == 55
        assert profile.requested_evaluations == 4


@pytest.mark.parametrize(
    ("option", "value"),
    [("--candidate-policy", "localneighbors"), ("--min-support-percent", "101")],
)
def test_compare_ga_cli_rejects_unknown_policy_and_support_values(
    tmp_path: Path, option: str, value: str
):
    fixture = tmp_path / "fixture.txt"
    fixture.write_text("{2,1,1}{1,1,1}", encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks" / "compare_ga.py"),
            str(fixture),
            option,
            value,
        ],
        cwd=root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "error:" in result.stderr
