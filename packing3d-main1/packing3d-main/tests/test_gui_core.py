from __future__ import annotations

import sys
from pathlib import Path

import pytest

from packing import (
    BLPriority,
    Box,
    Container,
    DecoderConfig,
    GAConfig,
    GAExecutionConfig,
    GAProgress,
    RunOptions,
    read_result_json,
    run_bl_ga,
    run_pp_ga,
    write_unity_result_data,
)

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
if str(DEMO) not in sys.path:
    sys.path.insert(0, str(DEMO))

from gui_core import parse_decoder_settings, render_previews, run_ga_core  # noqa: E402


def test_gui_decoder_settings_parser_normalizes_canonical_inputs():
    assert parse_decoder_settings("LocalNeighbors", "55") == DecoderConfig(
        candidate_policy="LocalNeighbors",
        candidate_policy_version=1,
        min_support_percent=55,
        support_model_version="support-union-v1",
    )
    assert parse_decoder_settings("Cartesian", "0") == DecoderConfig(min_support_percent=0)


@pytest.mark.parametrize(
    ("candidate_policy", "support_percent"),
    [
        ("localneighbors", "55"),
        ("Cartesian", "-1"),
        ("Cartesian", "101"),
        ("Cartesian", "55.0"),
    ],
)
def test_gui_decoder_settings_parser_rejects_noncanonical_values(
    candidate_policy: str, support_percent: str
):
    with pytest.raises(ValueError):
        parse_decoder_settings(candidate_policy, support_percent)


@pytest.mark.parametrize(
    ("decoder", "priority", "execution"),
    [
        ("BL", BLPriority.BOTTOM_FIRST, GAExecutionConfig(workers=1, cache=False)),
        ("BL", BLPriority.BACK_FIRST, GAExecutionConfig(workers=2, cache=True)),
        ("PP", BLPriority.BOTTOM_FIRST, GAExecutionConfig(workers=2, cache=False)),
        ("PP", BLPriority.BACK_FIRST, GAExecutionConfig(workers=1, cache=True)),
    ],
)
def test_headless_gui_ga_bridge_matches_public_core_and_common_io(
    tmp_path: Path,
    decoder: str,
    priority: BLPriority,
    execution: GAExecutionConfig,
):
    container = Container(3, 2, 1)
    boxes = (
        Box(1, 2, 1, 1),
        Box(2, 1, 2, 1),
        Box(3, 1, 1, 1),
    )
    config = GAConfig(population_size=3, generation_count=2, crossover_rate=0.8, mutation_rate=0.1)
    kwargs = dict(
        priority=priority,
        config=config,
        seed=19,
        execution=execution,
    )

    gui_run = run_ga_core(container, boxes, decoder=decoder, **kwargs)
    direct = (run_bl_ga if decoder == "BL" else run_pp_ga)(container, boxes, **kwargs)
    assert gui_run == direct
    assert gui_run.execution == execution

    json_text, unity_text = render_previews(gui_run.best.packing_result)
    json_path = tmp_path / "preview.json"
    json_path.write_text(json_text, encoding="utf-8", newline="\n")
    assert read_result_json(json_path) == gui_run.best.packing_result

    unity_path = tmp_path / "expected-unity.txt"
    write_unity_result_data(unity_path, gui_run.best.packing_result)
    assert unity_text == unity_path.read_text(encoding="utf-8")


def test_headless_gui_bridge_forwards_progress_callback():
    container = Container(3, 2, 1)
    boxes = (
        Box(1, 2, 1, 1),
        Box(2, 1, 2, 1),
        Box(3, 1, 1, 1),
    )
    config = GAConfig(population_size=3, generation_count=2, crossover_rate=0.8, mutation_rate=0.1)
    progress: list[GAProgress] = []

    run = run_ga_core(
        container,
        boxes,
        decoder="PP",
        priority=BLPriority.BOTTOM_FIRST,
        config=config,
        seed=5,
        execution=GAExecutionConfig(workers=1, cache=False),
        progress_callback=progress.append,
    )

    assert run.execution_stats.requested == 6
    assert len(progress) == 6
    assert progress[-1].completed_evaluations == 6
    assert progress[-1].decoder_name == "PP"


@pytest.mark.parametrize("decoder", ["BL", "PP"])
@pytest.mark.parametrize("candidate_policy", ["Cartesian", "LocalNeighbors"])
def test_headless_gui_ga_bridge_matches_core_with_explicit_decoder_settings(
    decoder: str, candidate_policy: str
):
    container = Container(3, 2, 1)
    boxes = (Box(1, 2, 1, 1), Box(2, 1, 2, 1), Box(3, 1, 1, 1))
    config = GAConfig(population_size=3, generation_count=2, crossover_rate=0.8, mutation_rate=0.1)
    decoder_config = DecoderConfig(candidate_policy=candidate_policy, min_support_percent=55)
    options = RunOptions(origin="gui")
    kwargs = dict(
        priority=BLPriority.BACK_FIRST,
        config=config,
        seed=19,
        execution=GAExecutionConfig(workers=1, cache=False),
        decoder_config=decoder_config,
        run_options=options,
    )

    gui_run = run_ga_core(container, boxes, decoder=decoder, **kwargs)
    direct = (run_bl_ga if decoder == "BL" else run_pp_ga)(container, boxes, **kwargs)

    assert gui_run == direct
    assert gui_run.decoder_config == decoder_config
