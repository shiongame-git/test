from __future__ import annotations

import pickle

import pytest

from packing import (
    BLPriority,
    Box,
    Container,
    DecoderConfig,
    EvaluationError,
    GAConfig,
    GAExecutionConfig,
    Individual,
    PPState,
    Point,
    run_bl_ga,
    run_pp_ga,
)
from packing.execution import EvaluationRuntime


def _problem():
    return (
        Container(4, 3, 2),
        (
            Box(1, 2, 1, 1),
            Box(2, 1, 2, 1),
            Box(3, 1, 1, 2),
            Box(4, 2, 2, 1),
            Box(5, 1, 1, 1),
        ),
    )


def test_execution_config_validation():
    assert GAExecutionConfig() == GAExecutionConfig(workers=1, cache=False)
    with pytest.raises(TypeError, match="workers"):
        GAExecutionConfig(workers=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="workers"):
        GAExecutionConfig(workers=0)
    with pytest.raises(TypeError, match="cache"):
        GAExecutionConfig(cache=1)  # type: ignore[arg-type]


def test_pp_state_is_spawn_pickle_roundtrippable():
    state = PPState({10: Point(1, 2, 3), 20: None})
    assert pickle.loads(pickle.dumps(state)) == state


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("cache", [False, True])
def test_bl_ga_execution_modes_match_sequential_reference(workers: int, cache: bool):
    container, boxes = _problem()
    config = GAConfig(population_size=6, generation_count=3, crossover_rate=0.75, mutation_rate=0.35)
    baseline = run_bl_ga(
        container,
        boxes,
        priority=BLPriority.BACK_FIRST,
        config=config,
        seed=31415,
    )
    candidate = run_bl_ga(
        container,
        boxes,
        priority=BLPriority.BACK_FIRST,
        config=config,
        seed=31415,
        execution=GAExecutionConfig(workers=workers, cache=cache),
    )

    # Execution settings/counters are non-semantic fields of the run result.
    assert candidate == baseline
    assert candidate.execution.workers == workers
    assert candidate.execution.cache is cache
    assert candidate.execution_stats.requested == config.population_size * config.generation_count


@pytest.mark.parametrize("workers", [1, 2])
@pytest.mark.parametrize("cache", [False, True])
def test_pp_ga_execution_modes_match_sequential_reference(workers: int, cache: bool):
    container, boxes = _problem()
    config = GAConfig(population_size=6, generation_count=3, crossover_rate=0.75, mutation_rate=0.35)
    baseline = run_pp_ga(
        container,
        boxes,
        priority=BLPriority.BOTTOM_FIRST,
        config=config,
        seed=2718,
    )
    candidate = run_pp_ga(
        container,
        boxes,
        priority=BLPriority.BOTTOM_FIRST,
        config=config,
        seed=2718,
        execution=GAExecutionConfig(workers=workers, cache=cache),
    )

    assert candidate == baseline
    assert candidate.execution_stats.requested == config.population_size * config.generation_count


def test_cache_reuses_duplicate_bl_inputs_without_removing_population_records():
    config = GAConfig(population_size=4, generation_count=3, crossover_rate=1.0, mutation_rate=1.0)
    run = run_bl_ga(
        Container(2, 2, 2),
        (),
        config=config,
        seed=0,
        execution=GAExecutionConfig(workers=2, cache=True),
    )

    assert run.execution_stats.requested == 12
    assert run.execution_stats.computed == 1
    assert run.execution_stats.cache_hits == 11
    assert all(len(generation.evaluations) == 4 for generation in run.generations)


def test_pp_cache_key_distinguishes_same_permutation_with_different_states():
    container = Container(2, 1, 1)
    boxes = (Box(1, 1, 1, 1), Box(2, 1, 1, 1))
    individual = Individual((1, 2))
    state_a = PPState({1: Point(0, 0, 0), 2: Point(0, 0, 0)})  # deliberate conflict
    state_b = PPState({1: Point(1, 0, 0), 2: Point(0, 0, 0)})

    with EvaluationRuntime(GAExecutionConfig(workers=2, cache=True)) as runtime:
        evaluations = runtime.evaluate_pp(
            container,
            boxes,
            ((individual, state_a), (individual, state_b)),
            priority=BLPriority.BOTTOM_FIRST,
            generation_index=1,
        )
        stats = runtime.stats

    assert len(evaluations) == 2
    assert stats.requested == 2
    assert stats.computed == 2
    assert stats.cache_hits == 0
    assert evaluations[0].packing_result != evaluations[1].packing_result


def test_bl_cache_key_distinguishes_support_settings_within_one_runtime():
    container = Container(3, 2, 1)
    boxes = (Box(1, 1, 2, 1), Box(2, 1, 1, 1), Box(3, 2, 1, 1))
    individual = Individual((1, 2, 3))

    with EvaluationRuntime(GAExecutionConfig(workers=1, cache=True)) as runtime:
        legacy = runtime.evaluate_bl(
            container,
            boxes,
            (individual,),
            priority=BLPriority.BOTTOM_FIRST,
            generation_index=0,
            decoder_config=DecoderConfig(min_support_percent=0),
        )[0]
        supported = runtime.evaluate_bl(
            container,
            boxes,
            (individual,),
            priority=BLPriority.BOTTOM_FIRST,
            generation_index=0,
            decoder_config=DecoderConfig(min_support_percent=55),
        )[0]
        stats = runtime.stats

    assert legacy.packing_result != supported.packing_result
    assert stats.requested == 2
    assert stats.computed == 2
    assert stats.cache_hits == 0


def test_pp_cache_key_distinguishes_candidate_policies():
    container = Container(3, 1, 3)
    boxes = (
        Box(10, 1, 1, 2),
        Box(11, 1, 1, 1),
        Box(99, 2, 1, 2),
    )
    individual = Individual((10, 11, 99))
    state = PPState({10: Point(0, 0, 0), 11: Point(2, 0, 0), 99: None})

    with EvaluationRuntime(GAExecutionConfig(workers=1, cache=True)) as runtime:
        cartesian = runtime.evaluate_pp(
            container,
            boxes,
            ((individual, state),),
            priority=BLPriority.BOTTOM_FIRST,
            generation_index=0,
            decoder_config=DecoderConfig(
                candidate_policy="Cartesian", min_support_percent=55
            ),
        )[0]
        local = runtime.evaluate_pp(
            container,
            boxes,
            ((individual, state),),
            priority=BLPriority.BOTTOM_FIRST,
            generation_index=0,
            decoder_config=DecoderConfig(
                candidate_policy="LocalNeighbors", min_support_percent=55
            ),
        )[0]
        stats = runtime.stats

    assert cartesian.packing_result != local.packing_result
    assert stats.requested == 2
    assert stats.computed == 2
    assert stats.cache_hits == 0


@pytest.mark.parametrize("runner", [run_bl_ga, run_pp_ga])
@pytest.mark.parametrize("priority", list(BLPriority))
@pytest.mark.parametrize("min_support_percent", [0, 55, 100])
@pytest.mark.parametrize("candidate_policy", ["Cartesian", "LocalNeighbors"])
def test_ga_settings_match_across_workers_and_cache(
    runner, priority: BLPriority, min_support_percent: int, candidate_policy: str
):
    container = Container(3, 2, 2)
    boxes = (
        Box(1, 2, 1, 1),
        Box(2, 1, 2, 1),
        Box(3, 1, 1, 2),
        Box(4, 1, 1, 1),
    )
    settings = DecoderConfig(
        candidate_policy=candidate_policy, min_support_percent=min_support_percent
    )
    config = GAConfig(population_size=3, generation_count=2, crossover_rate=0.8, mutation_rate=0.1)
    baseline_progress = []
    baseline = runner(
        container,
        boxes,
        priority=priority,
        config=config,
        seed=912,
        execution=GAExecutionConfig(workers=1, cache=False),
        decoder_config=settings,
        progress_callback=baseline_progress.append,
    )

    assert baseline.run_metadata["version"] == 2
    decoder_metadata = baseline.run_metadata["decoder"]
    assert decoder_metadata["reference_version"] == f"{baseline.decoder_name}-ref-v2"
    assert tuple(decoder_metadata)[-4:] == (
        "candidate_policy",
        "candidate_policy_version",
        "min_support_percent",
        "support_model_version",
    )
    assert tuple(decoder_metadata.values())[-4:] == (
        settings.candidate_policy,
        settings.candidate_policy_version,
        settings.min_support_percent,
        settings.support_model_version,
    )
    assert "constraints" not in decoder_metadata
    assert baseline.run_metadata["execution"]["workers"]["resolved_workers"] == baseline.execution.workers
    assert baseline.run_metadata["execution"]["cache"] is baseline.execution.cache
    assert baseline.run_metadata["statistics"]["requested"] == baseline.execution_stats.requested
    assert baseline.run_metadata["statistics"]["computed"] == baseline.execution_stats.computed
    assert baseline.run_metadata["statistics"]["cache_hits"] == baseline.execution_stats.cache_hits

    # Cover worker=4 for both decoders and policies without repeating every support/priority case.
    workers_to_check = (1, 2)
    if priority == BLPriority.BOTTOM_FIRST and min_support_percent == 55:
        workers_to_check += (4,)
    for workers in workers_to_check:
        for cache in (False, True):
            progress = []
            candidate = runner(
                container,
                boxes,
                priority=priority,
                config=config,
                seed=912,
                execution=GAExecutionConfig(workers=workers, cache=cache),
                decoder_config=settings,
                progress_callback=progress.append,
            )
            assert candidate == baseline
            assert progress == baseline_progress
            assert candidate.execution_stats.requested == config.population_size * config.generation_count
            assert candidate.run_metadata["execution"]["workers"]["resolved_workers"] == workers
            assert candidate.run_metadata["execution"]["cache"] is cache
            assert candidate.run_metadata["statistics"]["computed"] == candidate.execution_stats.computed
            assert candidate.run_metadata["statistics"]["cache_hits"] == candidate.execution_stats.cache_hits


@pytest.mark.parametrize("runner", [run_bl_ga, run_pp_ga])
@pytest.mark.parametrize("decoder_config", [object()])
@pytest.mark.parametrize("boxes", [(), (Box(9, 4, 1, 1),)])
def test_invalid_decoder_settings_are_rejected_before_observation_rng_or_pool(
    monkeypatch, runner, decoder_config, boxes
):
    import packing.execution as execution_module
    import packing.ga as ga_module
    import packing.run_context as run_context

    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("invalid decoder settings must fail before this boundary")

    monkeypatch.setattr(run_context, "observe_environment", forbidden)
    monkeypatch.setattr(ga_module.random, "Random", forbidden)
    monkeypatch.setattr(execution_module, "ProcessPoolExecutor", forbidden)

    with pytest.raises(TypeError):
        runner(
            Container(2, 2, 2),
            boxes,
            config=GAConfig(population_size=2, generation_count=2),
            execution=GAExecutionConfig(workers=2, cache=True),
            decoder_config=decoder_config,
        )

    assert calls == []


def test_parallel_error_reports_lowest_population_index_not_completion_order():
    container = Container(2, 1, 1)
    boxes = (Box(1, 1, 1, 1), Box(2, 1, 1, 1))
    population = (
        Individual((1, 2)),
        Individual((1, 999)),
        Individual((888, 2)),
    )

    with pytest.raises(EvaluationError) as caught:
        with EvaluationRuntime(GAExecutionConfig(workers=2, cache=False)) as runtime:
            runtime.evaluate_bl(
                container,
                boxes,
                population,
                priority=BLPriority.BOTTOM_FIRST,
                generation_index=4,
                decoder_config=DecoderConfig(min_support_percent=55),
            )

    assert caught.value.generation_index == 4
    assert caught.value.population_index == 1
    assert isinstance(caught.value.__cause__, ValueError)


def test_all_unplaced_result_is_unchanged_by_parallel_and_cache():
    container = Container(1, 1, 1)
    boxes = (Box(1, 2, 1, 1), Box(2, 1, 2, 1), Box(3, 1, 1, 2))
    config = GAConfig(population_size=5, generation_count=3, crossover_rate=0.8, mutation_rate=0.2)
    baseline = run_bl_ga(container, boxes, config=config, seed=42)
    accelerated = run_bl_ga(
        container,
        boxes,
        config=config,
        seed=42,
        execution=GAExecutionConfig(workers=2, cache=True),
    )

    assert accelerated == baseline
    assert all(
        evaluation.packing_result.placed_count == 0
        for generation in accelerated.generations
        for evaluation in generation.evaluations
    )


def test_public_run_bl_ga_propagates_timeout_as_indexed_evaluation_error(monkeypatch):
    import packing.execution as execution_module

    def timeout_worker(container, boxes, individual, priority, decoder_config, *_):
        raise TimeoutError("public BL timeout")

    monkeypatch.setattr(execution_module, "_bl_worker", timeout_worker)
    with pytest.raises(EvaluationError) as caught:
        run_bl_ga(
            Container(2, 1, 1),
            (Box(1, 1, 1, 1),),
            config=GAConfig(population_size=2, generation_count=2, crossover_rate=0.0, mutation_rate=0.0),
            seed=0,
            execution=GAExecutionConfig(workers=1, cache=False),
            decoder_config=DecoderConfig(min_support_percent=55),
        )

    assert caught.value.generation_index == 0
    assert caught.value.population_index == 0
    assert isinstance(caught.value.__cause__, TimeoutError)


def test_public_run_pp_ga_propagates_pp_timeout_after_initial_bl_generation(monkeypatch):
    import packing.execution as execution_module

    def timeout_worker(container, boxes, individual, state, priority, decoder_config, *_):
        raise TimeoutError("public PP timeout")

    monkeypatch.setattr(execution_module, "_pp_worker", timeout_worker)
    with pytest.raises(EvaluationError) as caught:
        run_pp_ga(
            Container(2, 1, 1),
            (Box(1, 1, 1, 1),),
            config=GAConfig(population_size=1, generation_count=2, crossover_rate=0.0, mutation_rate=0.0),
            seed=0,
            execution=GAExecutionConfig(workers=1, cache=True),
            decoder_config=DecoderConfig(min_support_percent=55),
        )

    assert caught.value.generation_index == 1
    assert caught.value.population_index == 0
    assert isinstance(caught.value.__cause__, TimeoutError)


def test_failed_evaluation_is_not_cached_for_later_attempt(monkeypatch):
    import packing.execution as execution_module

    calls = 0

    def timeout_worker(container, boxes, individual, priority, decoder_config, *_):
        nonlocal calls
        calls += 1
        raise TimeoutError("do not cache me")

    monkeypatch.setattr(execution_module, "_bl_worker", timeout_worker)
    container = Container(2, 1, 1)
    boxes = (Box(1, 1, 1, 1),)
    population = (Individual((1,)),)

    with EvaluationRuntime(GAExecutionConfig(workers=1, cache=True)) as runtime:
        for generation_index in (0, 1):
            with pytest.raises(EvaluationError) as caught:
                runtime.evaluate_bl(
                    container,
                    boxes,
                    population,
                    priority=BLPriority.BOTTOM_FIRST,
                    generation_index=generation_index,
                    decoder_config=DecoderConfig(min_support_percent=55),
                )
            assert isinstance(caught.value.__cause__, TimeoutError)
        stats = runtime.stats

    assert calls == 2
    assert stats.requested == 2
    assert stats.computed == 2
    assert stats.cache_hits == 0


def test_real_spawn_reports_low_index_even_when_high_index_fails_first(monkeypatch):
    import packing.execution as execution_module
    from spawn_helpers import delayed_failure_bl_worker

    monkeypatch.setattr(execution_module, "_bl_worker", delayed_failure_bl_worker)
    container = Container(2, 1, 1)
    boxes = (Box(1, 1, 1, 1), Box(2, 1, 1, 1))
    population = (Individual((1, 2)), Individual((2, 1)))

    with pytest.raises(EvaluationError) as caught:
        with EvaluationRuntime(GAExecutionConfig(workers=2, cache=False)) as runtime:
            runtime.evaluate_bl(
                container,
                boxes,
                population,
                priority=BLPriority.BOTTOM_FIRST,
                generation_index=3,
                decoder_config=DecoderConfig(min_support_percent=55),
            )

    assert caught.value.generation_index == 3
    assert caught.value.population_index == 0
    assert isinstance(caught.value.__cause__, TimeoutError)
    assert "low-index delayed failure" in str(caught.value.__cause__)


@pytest.mark.parametrize("workers", [1, 2])
def test_progress_counts_requested_evaluations_with_cache_reuse(workers: int):
    config = GAConfig(population_size=4, generation_count=3, crossover_rate=1.0, mutation_rate=1.0)
    progress = []
    run = run_bl_ga(
        Container(2, 2, 2),
        (),
        config=config,
        seed=0,
        execution=GAExecutionConfig(workers=workers, cache=True),
        progress_callback=progress.append,
    )

    assert run.execution_stats.requested == 12
    assert run.execution_stats.computed == 1
    assert run.execution_stats.cache_hits == 11
    assert len(progress) == 12
    assert [item.completed_evaluations for item in progress] == list(range(1, 13))
    assert [(item.generation_index, item.population_index) for item in progress] == [
        (generation, population)
        for generation in range(3)
        for population in range(4)
    ]
