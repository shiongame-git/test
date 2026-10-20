from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import bench_bl, bench_workers_acceptance


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("existing_suffix", [".json", ".md"])
def test_workers_benchmark_refuses_existing_output_pair(tmp_path, existing_suffix):
    output_json = tmp_path / "workers.json"
    output_markdown = tmp_path / "workers.md"
    existing = output_json if existing_suffix == ".json" else output_markdown
    existing.write_text("preserve existing artifact", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        bench_workers_acceptance._output_paths(output_json, output_markdown)

    assert existing.read_text(encoding="utf-8") == "preserve existing artifact"
    other = output_markdown if existing == output_json else output_json
    assert not other.exists()


def test_workers_benchmark_requires_explicit_output_paths():
    with pytest.raises(SystemExit, match="2"):
        bench_workers_acceptance._parse_args([])


def test_workers_benchmark_refuses_existing_output_before_running_trials(
    tmp_path, monkeypatch
):
    output_json = tmp_path / "workers.json"
    output_markdown = tmp_path / "workers.md"
    output_markdown.write_text("preserve", encoding="utf-8")

    def unexpected_trial(*_args, **_kwargs):
        pytest.fail("benchmark trial started before output preflight")

    monkeypatch.setattr(bench_workers_acceptance, "run_comparison_trial", unexpected_trial)
    with pytest.raises(SystemExit) as error:
        bench_workers_acceptance.main(
            [
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_markdown),
            ]
        )

    assert error.value.code == 2
    assert not output_json.exists()
    assert output_markdown.read_text(encoding="utf-8") == "preserve"


def test_workers_benchmark_writes_requested_new_artifact_pair(tmp_path):
    report = json.loads(
        (ROOT / "tests" / "fixtures" / "comparison_v4.json").read_text(
            encoding="utf-8"
        )
    )
    output_json = tmp_path / "measurements.json"
    output_markdown = tmp_path / "summary.md"

    result_json, result_markdown = bench_workers_acceptance._write_artifacts(
        report,
        [("small", 1, False, 1.0, 1.5, 2.0)],
        output_json,
        output_markdown,
    )

    assert result_json == output_json.resolve()
    assert result_markdown == output_markdown.resolve()
    assert json.loads(result_json.read_text(encoding="utf-8"))["version"] == 4
    markdown = result_markdown.read_text(encoding="utf-8")
    assert "|small|1|OFF|1.000000|1.500000|2.000000|" in markdown
    assert "[`measurements.json`](measurements.json)" in markdown


def test_workers_benchmark_preserves_artifact_created_after_preflight(
    tmp_path, monkeypatch
):
    report = json.loads(
        (ROOT / "tests" / "fixtures" / "comparison_v4.json").read_text(
            encoding="utf-8"
        )
    )
    output_json = tmp_path / "measurements.json"
    output_markdown = tmp_path / "summary.md"
    original_writer = bench_workers_acceptance.write_comparison_report_json

    def create_markdown_during_serialization(path, contents):
        original_writer(path, contents)
        output_markdown.write_text("concurrent existing artifact", encoding="utf-8")

    monkeypatch.setattr(
        bench_workers_acceptance,
        "write_comparison_report_json",
        create_markdown_during_serialization,
    )

    with pytest.raises(FileExistsError):
        bench_workers_acceptance._write_artifacts(
            report, [], output_json, output_markdown
        )

    assert not output_json.exists()
    assert output_markdown.read_text(encoding="utf-8") == "concurrent existing artifact"


def test_bench_bl_reports_effective_version_settings_and_validation_scope(
    tmp_path, capsys
):
    fixture = tmp_path / "small.txt"
    fixture.write_text("{2,2,2}{1,1,1}", encoding="utf-8")

    bench_bl.run(fixture, bench_bl.BLPriority.BACK_FIRST)

    output = capsys.readouterr().out
    assert "git_revision: " in output
    assert "distribution_version" not in output
    assert "priority: BackFirst" in output
    assert (
        "effective_decoder_config: candidate_policy=Cartesian, "
        "candidate_policy_version=1, min_support_percent=55, "
        "support_model_version=support-union-v1"
    ) in output
    assert "each of 3 timed intervals covers pack() only" in output
    assert "pairwise overlap, and result aggregates" in output
    assert "support-threshold validation is not performed" in output
    assert "placed_count: 1" in output
    assert "failed_count: 0" in output
