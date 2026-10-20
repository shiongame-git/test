from __future__ import annotations

import random

import pytest

from packing import (
    BLPriority,
    Box,
    Container,
    EvaluationResult,
    GAConfig,
    GAProgress,
    Individual,
    Point,
    evaluate_bl_individual,
    run_bl_ga,
    validate_result,
)
from packing.ga import (
    _crossover_pair,
    _initial_population,
    _mutate_swap,
)


def _genes_by_generation(run):
    return tuple(
        tuple(e.individual.genes for e in generation.evaluations)
        for generation in run.generations
    )


def test_individual_contract_and_problem_permutation_validation():
    with pytest.raises(TypeError, match="tuple"):
        Individual([1, 2])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Python int"):
        Individual((True,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="duplicate"):
        Individual((1, 1))

    boxes = (Box(10, 1, 1, 1), Box(20, 1, 1, 1))
    with pytest.raises(ValueError, match="not a permutation"):
        evaluate_bl_individual(Container(2, 1, 1), boxes, Individual((10, 99)))


def test_density_evaluator_keeps_individual_order_and_uses_density_only():
    c = Container(2, 1, 1)
    boxes = (Box(10, 1, 1, 1), Box(20, 2, 1, 1))
    individual = Individual((20, 10))
    evaluation = evaluate_bl_individual(c, boxes, individual)

    assert tuple(p.box.id for p in evaluation.packing_result.placements) == (20, 10)
    assert tuple(p.position for p in evaluation.packing_result.placements) == (Point(0, 0, 0), None)
    assert evaluation.fitness == evaluation.packing_result.density == 1.0
    assert validate_result(
        evaluation.packing_result,
        expected_container=c,
        ordered_boxes=(boxes[1], boxes[0]),
    ) == ()


def test_initial_population_is_seeded_shuffle_and_allows_duplicates():
    base = (1, 2, 3, 4)
    p0 = _initial_population(base, 4, random.Random(0))
    p0_again = _initial_population(base, 4, random.Random(0))
    p1 = _initial_population(base, 4, random.Random(1))

    assert p0 == p0_again
    assert p0 != p1
    for individual in p0 + p1:
        assert sorted(individual.genes) == [1, 2, 3, 4]

    empty = _initial_population((), 3, random.Random(0))
    assert empty == (Individual(()), Individual(()), Individual(()))


def test_crossover_segment_moves_to_front_and_preserves_permutation():
    p1 = Individual((1, 2, 3, 4, 5))
    p2 = Individual((5, 4, 3, 2, 1))
    rng = random.Random(9)
    c1, c2 = _crossover_pair(p1, p2, 1.0, rng)

    # Freeze the selected boundaries/result for the reference RNG conditions.
    assert c1.genes == (3, 4, 5, 2, 1)
    assert c2.genes == (3, 2, 1, 4, 5)
    assert sorted(c1.genes) == sorted(p1.genes)
    assert sorted(c2.genes) == sorted(p1.genes)

    unchanged = _crossover_pair(p1, p2, 0.0, random.Random(9))
    assert unchanged == (p1, p2)


def test_swap_mutation_rate_endpoints_and_permutation():
    original = Individual((1, 2, 3, 4))
    assert _mutate_swap(original, 0.0, random.Random(4)) == original
    mutated = _mutate_swap(original, 1.0, random.Random(4))
    assert mutated.genes == (3, 2, 1, 4)
    assert sorted(mutated.genes) == sorted(original.genes)

    single = Individual((1,))
    assert _mutate_swap(single, 1.0, random.Random(4)) == single


def test_generation_count_includes_initial_and_elite_is_first_next_generation():
    c = Container(3, 2, 1)
    boxes = (
        Box(1, 2, 1, 1),
        Box(2, 1, 2, 1),
        Box(3, 1, 1, 1),
        Box(4, 1, 1, 1),
    )
    config = GAConfig(population_size=3, generation_count=2, crossover_rate=0.0, mutation_rate=0.0)
    run = run_bl_ga(c, boxes, config=config, seed=2)

    assert len(run.generations) == 2
    assert all(len(g.evaluations) == 3 for g in run.generations)
    assert run.generations[1].evaluations[0].individual == run.generations[0].best.individual


def test_odd_population_size_is_preserved_and_all_individuals_stay_permutations():
    c = Container(4, 2, 1)
    boxes = tuple(Box(i, 1, 1, 1) for i in range(5))
    config = GAConfig(population_size=5, generation_count=4, crossover_rate=1.0, mutation_rate=1.0)
    run = run_bl_ga(c, boxes, config=config, seed=13)
    expected_ids = sorted(box.id for box in boxes)

    assert [len(g.evaluations) for g in run.generations] == [5, 5, 5, 5]
    for generation in run.generations:
        for evaluation in generation.evaluations:
            assert sorted(evaluation.individual.genes) == expected_ids


def test_fitness_tie_keeps_first_population_and_earliest_global_best():
    # Eight unit cubes all fill this container regardless of order, so all fitness
    # values tie and no hidden failed-count/ID tie breaker can affect selection.
    c = Container(2, 2, 2)
    boxes = tuple(Box(i, 1, 1, 1) for i in range(8))
    run = run_bl_ga(
        c,
        boxes,
        config=GAConfig(population_size=4, generation_count=3, crossover_rate=1.0, mutation_rate=1.0),
        seed=5,
    )

    assert all(g.best_index == 0 for g in run.generations)
    assert run.best_generation_index == 0
    assert run.best_population_index == 0
    assert run.best.fitness == 1.0


def test_same_seed_reproduces_semantic_run_and_different_seed_changes_initial_population():
    c = Container(3, 3, 2)
    boxes = (
        Box(10, 2, 1, 1),
        Box(20, 1, 2, 1),
        Box(30, 1, 1, 2),
        Box(40, 2, 2, 1),
        Box(50, 1, 1, 1),
    )
    config = GAConfig(population_size=4, generation_count=3, crossover_rate=0.8, mutation_rate=0.1)
    a = run_bl_ga(c, boxes, priority=BLPriority.BACK_FIRST, config=config, seed=20260918)
    b = run_bl_ga(c, boxes, priority=BLPriority.BACK_FIRST, config=config, seed=20260918)
    other = run_bl_ga(c, boxes, priority=BLPriority.BACK_FIRST, config=config, seed=20260919)

    assert a == b
    assert _genes_by_generation(a)[0] != _genes_by_generation(other)[0]


def test_empty_and_single_box_problems_are_valid_ga_inputs():
    empty = run_bl_ga(
        Container(2, 2, 2),
        (),
        config=GAConfig(population_size=3, generation_count=2, crossover_rate=1.0, mutation_rate=1.0),
        seed=0,
    )
    assert empty.best.individual == Individual(())
    assert empty.best.fitness == 0.0
    assert empty.best.packing_result.placements == ()

    box = Box(77, 1, 1, 1)
    single = run_bl_ga(
        Container(2, 2, 2),
        (box,),
        config=GAConfig(population_size=3, generation_count=2, crossover_rate=1.0, mutation_rate=1.0),
        seed=0,
    )
    assert all(
        evaluation.individual == Individual((77,))
        for generation in single.generations
        for evaluation in generation.evaluations
    )


def test_ga_config_and_seed_validation():
    with pytest.raises(TypeError, match="population_size"):
        GAConfig(population_size=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="population_size"):
        GAConfig(population_size=0)
    with pytest.raises(ValueError, match="generation_count"):
        GAConfig(generation_count=0)
    with pytest.raises(TypeError, match="Python float"):
        GAConfig(crossover_rate=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="crossover_rate"):
        GAConfig(crossover_rate=1.01)
    with pytest.raises(ValueError, match="mutation_rate"):
        GAConfig(mutation_rate=-0.01)
    with pytest.raises(TypeError, match="seed"):
        run_bl_ga(Container(1, 1, 1), (), seed=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        run_bl_ga(Container(1, 1, 1), (), seed=-1)


def test_default_config_matches_paper_experiment_values():
    assert GAConfig() == GAConfig(
        population_size=100,
        generation_count=10,
        crossover_rate=1.0,
        mutation_rate=0.1,
    )


def test_equal_density_does_not_use_failed_or_placed_count_as_secondary_fitness():
    from packing.ga import _best_index
    from packing.model import PackingResult, Placement

    container = Container(4, 1, 1)
    more_failures = EvaluationResult(
        Individual((1, 2, 3)),
        PackingResult(
            container,
            (
                Placement(Box(1, 2, 1, 1), Point(0, 0, 0)),
                Placement(Box(2, 1, 1, 1), None),
                Placement(Box(3, 1, 1, 1), None),
            ),
        ),
        0.5,
    )
    fewer_failures = EvaluationResult(
        Individual((4, 5)),
        PackingResult(
            container,
            (
                Placement(Box(4, 1, 1, 1), Point(0, 0, 0)),
                Placement(Box(5, 1, 1, 1), Point(1, 0, 0)),
            ),
        ),
        0.5,
    )

    assert more_failures.packing_result.failed_count == 2
    assert fewer_failures.packing_result.failed_count == 0
    assert _best_index((more_failures, fewer_failures)) == 0
    assert _best_index((fewer_failures, more_failures)) == 0


def test_bl_ga_progress_reports_requested_evaluations_without_changing_result():
    c = Container(3, 2, 1)
    boxes = (
        Box(1, 2, 1, 1),
        Box(2, 1, 2, 1),
        Box(3, 1, 1, 1),
    )
    config = GAConfig(population_size=3, generation_count=2, crossover_rate=0.8, mutation_rate=0.1)
    progress: list[GAProgress] = []

    observed = run_bl_ga(c, boxes, config=config, seed=11, progress_callback=progress.append)
    baseline = run_bl_ga(c, boxes, config=config, seed=11)

    assert observed == baseline
    assert len(progress) == 6
    assert [item.completed_evaluations for item in progress] == list(range(1, 7))
    assert [(item.generation_index, item.population_index) for item in progress] == [
        (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)
    ]
    assert all(item.decoder_name == "BL" for item in progress)
    assert all(item.total_evaluations == 6 for item in progress)
    assert progress[-1].fraction == 1.0
