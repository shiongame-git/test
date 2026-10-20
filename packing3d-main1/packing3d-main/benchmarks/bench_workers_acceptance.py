"""Measure workers/cache settings on two fixed synthetic PP-GA problems."""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from packing import (
    Box,
    ComparisonDecoder,
    Container,
    GAConfig,
    GAExecutionConfig,
    run_comparison_trial,
    write_comparison_report_json,
)


def _output_paths(output_json: Path, output_markdown: Path) -> tuple[Path, Path]:
    if output_json.suffix.lower() != ".json":
        raise ValueError("--output-json must use the .json extension")
    if output_markdown.suffix.lower() != ".md":
        raise ValueError("--output-markdown must use the .md extension")

    json_path = output_json.resolve()
    markdown_path = output_markdown.resolve()
    if os.path.normcase(str(json_path)) == os.path.normcase(str(markdown_path)):
        raise ValueError("JSON and Markdown output paths must be different")
    for path in (json_path, markdown_path):
        if not path.parent.is_dir():
            raise FileNotFoundError(f"output directory does not exist: {path.parent}")
    for path in (json_path, markdown_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing benchmark artifact: {path}")
    return json_path, markdown_path


def _markdown(
    groups: list[tuple[str, int, bool, float, float, float]],
    json_path: Path,
    markdown_path: Path,
) -> str:
    relative_json = os.path.relpath(json_path, markdown_path.parent).replace(os.sep, "/")
    json_link = quote(relative_json, safe="/-._~")
    return (
        f"# workers計測\n\n"
        f"{datetime.now(timezone(timedelta(hours=9))).date()}。"
        f"{platform.system()} {platform.python_implementation()} {platform.python_version()}。"
        "合成問題2種、PP-GA、BottomFirst、P=4、G=3、seed=17、交叉0.8・突然変異0.1。"
        "各条件3回。CPU検出・解決・spawn・転送・評価・shutdownを含むGA呼出し時間。\n\n"
        "|問題|workers|cache|min秒|median秒|max秒|\n"
        "|---|---:|---|---:|---:|---:|\n"
        + "\n".join(
            f"|{name}|{workers}|{'ON' if cache else 'OFF'}|{minimum:.6f}|"
            f"{median:.6f}|{maximum:.6f}|"
            for name, workers, cache, minimum, median, maximum in groups
        )
        + f"\n\nsmallは4箱・4³コンテナ、largerは48箱・12³コンテナ。"
        "元fixtureや過去記録は変更していない。この2問題・この環境の値から最適workersや"
        "Windows性能を一般化しない。全36実行の入力・環境・統計は"
        f"[`{json_path.name}`]({json_link})に保存。\n"
    )


def _write_artifacts(
    report: dict[str, object],
    groups: list[tuple[str, int, bool, float, float, float]],
    output_json: Path,
    output_markdown: Path,
) -> tuple[Path, Path]:
    json_path, markdown_path = _output_paths(output_json, output_markdown)
    markdown_data = _markdown(groups, json_path, markdown_path).encode("utf-8")

    # Use the public report writer for schema validation and canonical JSON,
    # then create the requested destination exclusively to prevent replacement.
    with tempfile.TemporaryDirectory(prefix="packing3d-workers-") as temporary_dir:
        temporary_json = Path(temporary_dir) / "report.json"
        write_comparison_report_json(temporary_json, report)
        json_data = temporary_json.read_bytes()

    created: list[Path] = []
    try:
        with json_path.open("xb") as json_file:
            created.append(json_path)
            json_file.write(json_data)
        with markdown_path.open("xb") as markdown_file:
            created.append(markdown_path)
            markdown_file.write(markdown_data)
    except Exception:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return json_path, markdown_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="new comparison-report JSON path; an existing file is never replaced",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        required=True,
        help="new summary Markdown path; an existing file is never replaced",
    )
    args = parser.parse_args(argv)
    try:
        args.output_json, args.output_markdown = _output_paths(
            args.output_json, args.output_markdown
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    trials = []
    groups: list[tuple[str, int, bool, float, float, float]] = []

    problems = (
        (
            "small",
            Container(4, 4, 4),
            tuple(Box(index, 1, 1, 1) for index in range(4)),
        ),
        (
            "larger",
            Container(12, 12, 12),
            tuple(
                Box(
                    index,
                    1 + index % 3,
                    1 + (index // 3) % 3,
                    1 + (index // 9) % 3,
                )
                for index in range(48)
            ),
        ),
    )

    for name, container, boxes in problems:
        for workers in (1, 2, 4):
            for cache in (False, True):
                times = []
                for _repeat in range(3):
                    trial = run_comparison_trial(
                        container,
                        boxes,
                        decoder=ComparisonDecoder.PP,
                        config=GAConfig(4, 3, 0.8, 0.1),
                        seed=17,
                        execution=GAExecutionConfig(workers, cache),
                        origin="benchmark",
                        problem_source=f"synthetic:{name}",
                    )
                    times.append(trial.elapsed_seconds)
                    trials.append(
                        {
                            "run_metadata": trial.run_metadata.to_dict(),
                            "best": {
                                "generation_index": trial.best_generation_index,
                                "population_index": trial.best_population_index,
                                "genes": list(trial.best_genes),
                                "fitness": trial.best_fitness,
                                "placed_count": trial.placed_count,
                                "failed_count": trial.failed_count,
                                "placed_volume": trial.placed_volume,
                                "density": trial.density,
                            },
                        }
                    )
                row = (
                    name,
                    workers,
                    cache,
                    min(times),
                    statistics.median(times),
                    max(times),
                )
                groups.append(row)
                print(row, flush=True)

    report = {
        "format": "packing3d-comparison-report",
        "version": 4,
        "implementation_version": "source",
        "trials": trials,
    }
    json_path, markdown_path = _write_artifacts(
        report, groups, args.output_json, args.output_markdown
    )
    print(f"wrote {json_path}", flush=True)
    print(f"wrote {markdown_path}", flush=True)


if __name__ == "__main__":
    main()
