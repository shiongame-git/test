from __future__ import annotations

import pytest

from packing import (
    BLPriority,
    Box,
    BoxOrderMode,
    ComparisonDecoder,
    Container,
    DecoderConfig,
    GAConfig,
    GAExecutionConfig,
    order_boxes_for_experiment,
    run_comparison_trial,
)


def _boxes():
    return (
        Box(10, 1, 1, 2),
        Box(20, 1, 1, 3),
        Box(30, 1, 1, 3),
        Box(40, 1, 1, 1),
    )


def test_source_order_is_preserved():
    boxes = _boxes()
    assert order_boxes_for_experiment(boxes, BoxOrderMode.SOURCE) == boxes


def test_depth_desc_order_is_stable_for_ties():
    ordered = order_boxes_for_experiment(_boxes(), BoxOrderMode.DEPTH_DESC_STABLE)
    assert tuple(box.id for box in ordered) == (20, 30, 10, 40)


def test_order_mode_requires_enum():
    with pytest.raises(TypeError, match="BoxOrderMode"):
        order_boxes_for_experiment(_boxes(), "source")  # type: ignore[arg-type]


@pytest.mark.parametrize("decoder", [ComparisonDecoder.BL, ComparisonDecoder.PP])
def test_comparison_trial_records_budget_and_valid_best_result(decoder: ComparisonDecoder):
    container = Container(3, 2, 2)
    boxes = (
        Box(1, 2, 1, 1),
        Box(2, 1, 2, 1),
        Box(3, 1, 1, 2),
        Box(4, 1, 1, 1),
    )
    config = GAConfig(population_size=4, generation_count=3, crossover_rate=0.8, mutation_rate=0.1)
    trial = run_comparison_trial(
        container,
        boxes,
        decoder=decoder,
        priority=BLPriority.BOTTOM_FIRST,
        order_mode=BoxOrderMode.DEPTH_DESC_STABLE,
        config=config,
        seed=7,
        execution=GAExecutionConfig(workers=1, cache=False),
        implementation_version="test",
    )

    assert trial.requested_evaluation_budget == 12
    assert trial.execution_stats.requested == 12
    assert trial.execution_stats.computed == 12
    assert trial.execution_stats.cache_hits == 0
    assert trial.best_fitness == trial.density
    assert trial.placed_count + trial.failed_count == len(boxes)
    assert set(trial.best_genes) == {1, 2, 3, 4}
    assert trial.environment.implementation_version == "test"
    assert trial.elapsed_seconds >= 0.0
    assert trial.profile is None


@pytest.mark.parametrize("decoder", [ComparisonDecoder.BL, ComparisonDecoder.PP])
def test_comparison_trial_profile_is_linked_to_its_metadata(decoder: ComparisonDecoder):
    trial = run_comparison_trial(
        Container(3, 2, 2),
        _boxes(),
        decoder=decoder,
        config=GAConfig(population_size=2, generation_count=2, crossover_rate=0.8, mutation_rate=0.1),
        seed=7,
        execution=GAExecutionConfig(workers=1, cache=False),
        decoder_config=DecoderConfig(candidate_policy="LocalNeighbors", min_support_percent=55),
        implementation_version="test",
        profile=True,
    )

    assert trial.profile is not None
    assert trial.run_metadata is not None
    assert trial.profile.run_id == trial.run_metadata["run_id"]
    assert trial.profile.decoder == decoder.value
    assert trial.profile.decoder_config == trial.decoder_config
    assert trial.profile.requested_evaluations == trial.requested_evaluation_budget
    assert trial.profile.computed_evaluations == trial.execution_stats.computed
    assert trial.profile.cache_hits == trial.execution_stats.cache_hits


@pytest.mark.parametrize("decoder", [ComparisonDecoder.BL, ComparisonDecoder.PP])
@pytest.mark.parametrize("candidate_policy", ["Cartesian", "LocalNeighbors"])
@pytest.mark.parametrize("min_support_percent", [0, 55, 100])
def test_comparison_trial_records_and_checks_its_effective_decoder_settings(
    decoder: ComparisonDecoder,
    candidate_policy: str,
    min_support_percent: int,
    monkeypatch,
):
    import packing.experiment as experiment_module

    settings = DecoderConfig(
        candidate_policy=candidate_policy,
        min_support_percent=min_support_percent,
    )
    original_validate_support = experiment_module.validate_support
    observed_settings = []

    def validate_with_observation(result, *, settings):
        observed_settings.append(settings)
        return original_validate_support(result, settings=settings)

    monkeypatch.setattr(experiment_module, "validate_support", validate_with_observation)
    trial = run_comparison_trial(
        Container(3, 2, 2),
        _boxes(),
        decoder=decoder,
        priority=BLPriority.BACK_FIRST,
        config=GAConfig(population_size=3, generation_count=2, crossover_rate=0.8, mutation_rate=0.1),
        seed=15,
        decoder_config=settings,
    )

    assert trial.decoder_config is settings
    assert observed_settings == [settings]
    assert trial.run_metadata["version"] == 2
    assert trial.run_metadata["decoder"]["reference_version"] == f"{decoder.value}-ref-v2"
    assert trial.run_metadata["decoder"]["candidate_policy"] == settings.candidate_policy
    assert trial.run_metadata["decoder"]["candidate_policy_version"] == settings.candidate_policy_version
    assert trial.run_metadata["decoder"]["min_support_percent"] == settings.min_support_percent
    assert trial.run_metadata["decoder"]["support_model_version"] == settings.support_model_version


def test_comparison_trial_rejects_non_config_decoder_settings_before_run():
    with pytest.raises(TypeError, match="decoder_config"):
        run_comparison_trial(
            Container(2, 1, 1),
            (Box(1, 1, 1, 1),),
            decoder=ComparisonDecoder.BL,
            decoder_config=object(),  # type: ignore[arg-type]
        )


def test_comparison_trial_rejects_best_result_when_support_check_fails(monkeypatch):
    import packing.experiment as experiment_module

    monkeypatch.setattr(
        experiment_module,
        "validate_support",
        lambda _result, *, settings: ("configured support violation",),
    )
    with pytest.raises(ValueError, match="support validation.*configured support violation"):
        run_comparison_trial(
            Container(2, 1, 1),
            (Box(1, 1, 1, 1),),
            decoder=ComparisonDecoder.BL,
            config=GAConfig(population_size=1, generation_count=1, crossover_rate=0.0, mutation_rate=0.0),
            decoder_config=DecoderConfig(min_support_percent=55),
        )


def test_comparison_trial_checks_best_with_independent_support_after_ga(monkeypatch):
    import packing._support as decoder_support
    import packing.experiment as experiment_module

    original_validate_support = experiment_module.validate_support
    observed = []

    def fail_if_decoder_support_is_reused(*_args, **_kwargs):
        raise AssertionError("comparison validation called decoder support code")

    def validate_after_ga(result, *, settings):
        observed.append(settings)
        monkeypatch.setattr(
            decoder_support,
            "support_union_area",
            fail_if_decoder_support_is_reused,
        )
        return original_validate_support(result, settings=settings)

    monkeypatch.setattr(experiment_module, "validate_support", validate_after_ga)
    trial = run_comparison_trial(
        Container(3, 2, 2),
        _boxes(),
        decoder=ComparisonDecoder.BL,
        config=GAConfig(population_size=3, generation_count=2, crossover_rate=0.8, mutation_rate=0.1),
        seed=15,
    )

    assert observed == [trial.decoder_config]
    assert trial.best_fitness == trial.density


def test_comparison_semantics_do_not_depend_on_parallel_cache_mode():
    container = Container(3, 2, 2)
    boxes = (
        Box(1, 2, 1, 1),
        Box(2, 1, 2, 1),
        Box(3, 1, 1, 2),
        Box(4, 1, 1, 1),
    )
    config = GAConfig(population_size=4, generation_count=3, crossover_rate=0.8, mutation_rate=0.1)
    baseline = run_comparison_trial(
        container,
        boxes,
        decoder=ComparisonDecoder.PP,
        config=config,
        seed=11,
        execution=GAExecutionConfig(workers=1, cache=False),
        implementation_version="test",
    )
    accelerated = run_comparison_trial(
        container,
        boxes,
        decoder=ComparisonDecoder.PP,
        config=config,
        seed=11,
        execution=GAExecutionConfig(workers=2, cache=True),
        implementation_version="test",
    )

    # elapsed/environment are deliberately non-semantic comparison fields.
    assert accelerated.best_genes == baseline.best_genes
    assert accelerated.best_generation_index == baseline.best_generation_index
    assert accelerated.best_population_index == baseline.best_population_index
    assert accelerated.best_fitness == baseline.best_fitness
    assert accelerated.density == baseline.density
    assert accelerated.placed_count == baseline.placed_count


def test_trial_problem_signature_distinguishes_problem_values_and_source_order():
    config = GAConfig(population_size=1, generation_count=1, crossover_rate=0.0, mutation_rate=0.0)
    execution = GAExecutionConfig(workers=1, cache=False)

    a = run_comparison_trial(
        Container(2, 1, 1),
        (Box(1, 1, 1, 1), Box(2, 1, 1, 1)),
        decoder=ComparisonDecoder.BL,
        config=config,
        execution=execution,
        implementation_version="test",
    )
    different_container = run_comparison_trial(
        Container(3, 1, 1),
        (Box(1, 1, 1, 1), Box(2, 1, 1, 1)),
        decoder=ComparisonDecoder.BL,
        config=config,
        execution=execution,
        implementation_version="test",
    )
    reversed_source = run_comparison_trial(
        Container(2, 1, 1),
        (Box(2, 1, 1, 1), Box(1, 1, 1, 1)),
        decoder=ComparisonDecoder.BL,
        config=config,
        execution=execution,
        implementation_version="test",
    )

    assert a.problem_signature != different_container.problem_signature
    assert a.problem_signature != reversed_source.problem_signature
    assert a != different_container
    assert a != reversed_source
    assert a.source_box_order == (1, 2)
    assert reversed_source.source_box_order == (2, 1)


def test_trial_records_effective_order_and_optional_provenance():
    digest = "a" * 64
    trial = run_comparison_trial(
        Container(4, 1, 1),
        _boxes(),
        decoder=ComparisonDecoder.BL,
        order_mode=BoxOrderMode.DEPTH_DESC_STABLE,
        config=GAConfig(population_size=1, generation_count=1, crossover_rate=0.0, mutation_rate=0.0),
        execution=GAExecutionConfig(workers=1, cache=False),
        implementation_version="test",
        problem_source="fixture.txt",
        fixture_sha256=digest.upper(),
    )

    assert trial.source_box_order == (10, 20, 30, 40)
    assert trial.effective_box_order == (20, 30, 10, 40)
    assert trial.problem_source == "fixture.txt"
    assert trial.fixture_sha256 == digest
    assert trial.problem_signature.container == (4, 1, 1)
    assert trial.problem_signature.source_boxes[0] == (10, 1, 1, 2)


def test_same_complete_problem_has_same_problem_signature():
    container = Container(3, 2, 2)
    boxes = _boxes()
    kwargs = dict(
        decoder=ComparisonDecoder.BL,
        order_mode=BoxOrderMode.SOURCE,
        config=GAConfig(population_size=2, generation_count=1, crossover_rate=0.0, mutation_rate=0.0),
        seed=5,
        execution=GAExecutionConfig(workers=1, cache=False),
        implementation_version="test",
    )
    first = run_comparison_trial(container, boxes, **kwargs)
    second = run_comparison_trial(container, boxes, **kwargs)
    assert first.problem_signature == second.problem_signature
    assert first.source_box_order == second.source_box_order
    assert first.effective_box_order == second.effective_box_order


@pytest.mark.parametrize(
    ("decoder", "order_mode", "expected_genes"),
    [
        (ComparisonDecoder.BL, BoxOrderMode.SOURCE, (4, 2, 1, 3)),
        (ComparisonDecoder.BL, BoxOrderMode.DEPTH_DESC_STABLE, (4, 1, 3, 2)),
        (ComparisonDecoder.PP, BoxOrderMode.SOURCE, (4, 2, 1, 3)),
        (ComparisonDecoder.PP, BoxOrderMode.DEPTH_DESC_STABLE, (4, 1, 3, 2)),
    ],
)
def test_fixed_four_comparison_trials_keep_phase7_semantics(decoder, order_mode, expected_genes):
    container = Container(3, 2, 2)
    boxes = (
        Box(1, 2, 1, 1),
        Box(2, 1, 2, 1),
        Box(3, 1, 1, 2),
        Box(4, 1, 1, 1),
    )
    trial = run_comparison_trial(
        container,
        boxes,
        decoder=decoder,
        priority=BLPriority.BOTTOM_FIRST,
        order_mode=order_mode,
        config=GAConfig(population_size=4, generation_count=3, crossover_rate=0.8, mutation_rate=0.1),
        seed=7,
        execution=GAExecutionConfig(workers=1, cache=False),
        decoder_config=DecoderConfig(min_support_percent=0),
        implementation_version="test",
    )

    assert trial.best_generation_index == 0
    assert trial.best_population_index == 0
    assert trial.best_genes == expected_genes
    assert trial.best_fitness == 7 / 12
    assert (trial.placed_count, trial.failed_count, trial.placed_volume, trial.density) == (4, 0, 7, 7 / 12)
    assert trial.requested_evaluation_budget == 12
    assert trial.execution_stats.requested == 12
