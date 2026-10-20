from __future__ import annotations

import time
from pathlib import Path

import pytest

from packing import (
    BLPriority,
    Box,
    Container,
    DecoderConfig,
    GAConfig,
    GAExecutionConfig,
    RunOptions,
    capture_ga_run,
    make_provenance,
    read_research_record,
    run_bl_ga,
    run_pp_ga,
    utc_now,
)
from packing.research_records import save_research_record


CONTAINER = Container(4, 3, 3)
BOXES = (Box(1, 1, 1, 1), Box(2, 2, 1, 3), Box(3, 3, 2, 2), Box(4, 3, 1, 3))
CONFIG = GAConfig(population_size=3, generation_count=3, crossover_rate=0.8, mutation_rate=0.1)


def _history(run):
    result = []
    for generation in run.generations:
        input_states = getattr(generation, "input_states", None)
        result.append((
            generation.generation_index,
            getattr(generation, "decoder_name", "BL"),
            generation.best_index,
            tuple((evaluation.individual.genes, evaluation.fitness, evaluation.packing_result)
                  for evaluation in generation.evaluations),
            None if input_states is None else tuple(input_states),
        ))
    return (tuple(result), run.best_generation_index, run.best_population_index,
            run.best.individual.genes, run.best.fitness, run.best.packing_result)


def _cases():
    for decoder in ("BL", "PP"):
        for policy in ("Cartesian", "LocalNeighbors"):
            for support in (0, 55, 100):
                yield decoder, policy, support, BLPriority.BOTTOM_FIRST
    yield "PP", "LocalNeighbors", 55, BLPriority.BACK_FIRST
    yield "BL", "Cartesian", 0, BLPriority.BACK_FIRST


@pytest.mark.slow
@pytest.mark.parametrize("decoder,policy,support,priority", list(_cases()))
def test_summary_full_profile_workers_cache_preserve_each_fixed_ga_history(
    tmp_path: Path, decoder: str, policy: str, support: int, priority: BLPriority
):
    runner = run_bl_ga if decoder == "BL" else run_pp_ga
    settings = DecoderConfig(candidate_policy=policy, min_support_percent=support)
    options = RunOptions(origin="api")
    baseline = runner(CONTAINER, BOXES, priority=priority, config=CONFIG, seed=17,
                      execution=GAExecutionConfig(1, False), decoder_config=settings,
                      run_options=options, profile=False)
    baseline_history = _history(baseline)
    assert any(item.position is None for generation in baseline.generations
               for evaluation in generation.evaluations for item in evaluation.packing_result.placements)

    provenance = make_provenance()
    for profile in (False, True):
        for workers in (1, 2):
            for cache in (False, True):
                run = runner(
                    CONTAINER, BOXES, priority=priority, config=CONFIG, seed=17,
                    execution=GAExecutionConfig(workers, cache), decoder_config=settings,
                    run_options=options, profile=profile,
                )
                assert _history(run) == baseline_history
                if decoder == "PP" and support in (55, 100) and profile and workers == 1 and not cache:
                    assert run.profile is not None
                    assert run.profile.metrics.pp_deferred_support > 0

                started = utc_now()
                stamp = time.perf_counter()
                capture = capture_ga_run(
                    run, execution_started_at=started, execution_finished_at=utc_now(),
                    call_wall_seconds=time.perf_counter() - stamp,
                    execution_provenance=provenance,
                )
                # Summary and Full must describe the very same GA execution.
                for level in ("summary", "full"):
                    key = f"{decoder}-{policy}-{support}-{priority.value}-{level}-{int(profile)}-{workers}-{int(cache)}"
                    saved = save_research_record(tmp_path / key, capture, run=run, save_level=level)
                    assert saved["status"] == "completed"
                    if level == "full":
                        assert saved["validation"]["source_run_match"] == "passed"
                        assert saved["validation"]["checked_evaluations"] == 9
                    loaded = read_research_record(tmp_path / key)
                    assert loaded["summary"]["best"]["genes"] == list(baseline.best.individual.genes)
                    assert loaded["summary"]["best"]["fitness"] == baseline.best.fitness
