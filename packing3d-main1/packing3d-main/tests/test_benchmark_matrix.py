from __future__ import annotations

import pytest

from benchmarks import bench_matrix


@pytest.mark.parametrize("existing_suffix", [".json", ".md"])
def test_benchmark_matrix_refuses_existing_acceptance_artifact_pair(
    tmp_path, existing_suffix: str
):
    output_json = tmp_path / "acceptance.json"
    existing = output_json.with_suffix(existing_suffix)
    existing.write_text("preserve historical evidence", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        bench_matrix._output_paths(output_json)

    assert existing.read_text(encoding="utf-8") == "preserve historical evidence"
    assert not output_json.with_suffix(".md" if existing_suffix == ".json" else ".json").exists()


def test_benchmark_matrix_budget_expiration_writes_no_acceptance_artifacts(tmp_path, monkeypatch):
    output_json = tmp_path / "interrupted.json"
    monkeypatch.setattr(
        bench_matrix, "_source_commit", lambda: "test-source-commit"
    )

    def expire_budget(budget):
        budget.started -= budget.seconds + 1
        return []

    monkeypatch.setattr(bench_matrix, "_measure_bl", expire_budget)
    monkeypatch.setattr(bench_matrix, "_measure_workers", lambda _budget: [])

    with pytest.raises(TimeoutError, match="no acceptance artifacts were written"):
        bench_matrix.main(
            ["--output-json", str(output_json), "--time-budget-seconds", "1800"]
        )

    assert not output_json.exists()
    assert not output_json.with_suffix(".md").exists()
