from __future__ import annotations

import random

from packing import (
    BLPriority,
    Box,
    Container,
    EvaluationResult,
    GAConfig,
    GAProgress,
    Individual,
    PPState,
    Point,
    evaluate_pp_individual,
    pp_state_from_result,
    run_bl_ga,
    run_pp_ga,
    validate_result,
)
from packing.model import PackingResult, Placement
from packing.pp_ga import _crossover_pair_with_states


def _parent_evaluation(
    container: Container,
    boxes_by_id: dict[int, Box],
    genes: tuple[int, ...],
    positions: dict[int, Point | None],
    fitness: float,
) -> EvaluationResult:
    individual = Individual(genes)
    result = PackingResult(
        container,
        tuple(Placement(boxes_by_id[box_id], positions[box_id]) for box_id in genes),
    )
    return EvaluationResult(individual, result, fitness)


def test_pp_state_from_result_is_id_keyed_and_preserves_none():
    c = Container(3, 1, 1)
    b1 = Box(10, 1, 1, 1)
    b2 = Box(20, 1, 1, 1)
    result = PackingResult(c, (Placement(b2, None), Placement(b1, Point(1, 0, 0))))

    state = pp_state_from_result(result)

    assert state == PPState({10: Point(1, 0, 0), 20: None})


def test_same_individual_with_different_pp_states_can_produce_different_packing():
    c = Container(2, 1, 1)
    boxes = (Box(1, 1, 1, 1), Box(2, 1, 1, 1))
    individual = Individual((1, 2))

    from_none = evaluate_pp_individual(c, boxes, individual, PPState({1: None, 2: None}))
    from_swapped = evaluate_pp_individual(
        c,
        boxes,
        individual,
        PPState({1: Point(1, 0, 0), 2: Point(0, 0, 0)}),
    )

    assert from_none.packing_result != from_swapped.packing_result
    assert from_none.fitness == from_swapped.fitness == 1.0


def test_crossover_inherits_each_gene_position_from_its_source_parent():
    c = Container(8, 4, 2)
    boxes = {i: Box(i, 1, 1, 1) for i in (1, 2, 3, 4)}
    p1_positions = {i: Point(i - 1, 0, 0) for i in boxes}
    p2_positions = {i: Point(i - 1, 2, 0) for i in boxes}
    p1 = _parent_evaluation(c, boxes, (1, 2, 3, 4), p1_positions, 0.5)
    p2 = _parent_evaluation(c, boxes, (4, 3, 2, 1), p2_positions, 0.5)

    seed = 1234
    control = random.Random(seed)
    assert control.random() < 1.0
    a, b = sorted(control.sample(range(5), 2))
    front1 = set(p1.individual.genes[a:b])
    front2 = set(p2.individual.genes[a:b])

    child1, child2 = _crossover_pair_with_states(p1, p2, 1.0, random.Random(seed))

    for gene in child1[0].genes:
        assert child1[1][gene] == (p1_positions[gene] if gene in front1 else p2_positions[gene])
    for gene in child2[0].genes:
        assert child2[1][gene] == (p2_positions[gene] if gene in front2 else p1_positions[gene])


def test_pp_ga_initial_generation_matches_bl_ga_initial_generation():
    c = Container(3, 2, 2)
    boxes = (
        Box(1, 1, 1, 1),
        Box(2, 2, 1, 1),
        Box(3, 1, 2, 1),
        Box(4, 1, 1, 2),
    )
    config = GAConfig(population_size=5, generation_count=3, crossover_rate=0.8, mutation_rate=0.2)

    bl = run_bl_ga(c, boxes, priority=BLPriority.BACK_FIRST, config=config, seed=77)
    pp = run_pp_ga(c, boxes, priority=BLPriority.BACK_FIRST, config=config, seed=77)

    assert pp.generations[0].decoder_name == "BL"
    assert pp.generations[0].input_states == (None,) * config.population_size
    assert pp.generations[0].evaluations == bl.generations[0].evaluations


def test_pp_ga_uses_pp_after_initial_generation_and_records_complete_input_states():
    c = Container(3, 2, 2)
    boxes = tuple(Box(i, 1, 1, 1) for i in range(1, 6))
    config = GAConfig(population_size=5, generation_count=4, crossover_rate=1.0, mutation_rate=1.0)

    run = run_pp_ga(c, boxes, config=config, seed=2026)
    ids = {box.id for box in boxes}

    assert tuple(g.decoder_name for g in run.generations) == ("BL", "PP", "PP", "PP")
    for generation in run.generations[1:]:
        for state, evaluation in zip(generation.input_states, generation.evaluations, strict=True):
            assert type(state) is PPState
            assert set(state) == ids
            ordered = tuple({box.id: box for box in boxes}[gene] for gene in evaluation.individual.genes)
            assert validate_result(
                evaluation.packing_result,
                expected_container=c,
                ordered_boxes=ordered,
            ) == ()


def test_pp_ga_mutation_does_not_rebind_positions_by_sequence_index():
    c = Container(4, 2, 1)
    boxes = tuple(Box(i, 1, 1, 1) for i in range(1, 5))
    config = GAConfig(population_size=6, generation_count=2, crossover_rate=0.0, mutation_rate=1.0)

    run = run_pp_ga(c, boxes, config=config, seed=9)
    parent_states = tuple(pp_state_from_result(e.packing_result) for e in run.generations[0].evaluations)

    for state in run.generations[1].input_states:
        assert type(state) is PPState
        assert any(state == parent_state for parent_state in parent_states)


def test_pp_ga_is_semantically_reproducible_for_same_seed():
    c = Container(4, 3, 2)
    boxes = (
        Box(1, 2, 1, 1),
        Box(2, 1, 2, 1),
        Box(3, 1, 1, 2),
        Box(4, 2, 2, 1),
        Box(5, 1, 1, 1),
    )
    config = GAConfig(population_size=7, generation_count=5, crossover_rate=0.75, mutation_rate=0.35)

    first = run_pp_ga(c, boxes, priority=BLPriority.BOTTOM_FIRST, config=config, seed=31415)
    second = run_pp_ga(c, boxes, priority=BLPriority.BOTTOM_FIRST, config=config, seed=31415)

    assert first == second


def test_pp_ga_generation_count_one_is_initial_bl_only():
    c = Container(2, 2, 2)
    boxes = (Box(1, 1, 1, 1), Box(2, 1, 1, 1))
    config = GAConfig(population_size=3, generation_count=1, crossover_rate=1.0, mutation_rate=1.0)

    run = run_pp_ga(c, boxes, config=config, seed=1)

    assert len(run.generations) == 1
    assert run.generations[0].decoder_name == "BL"
    assert run.decoder_name == "PP"


def test_pp_ga_empty_problem_is_supported():
    c = Container(2, 2, 2)
    config = GAConfig(population_size=3, generation_count=3, crossover_rate=1.0, mutation_rate=1.0)

    run = run_pp_ga(c, (), config=config, seed=4)

    assert tuple(g.decoder_name for g in run.generations) == ("BL", "PP", "PP")
    assert all(e.fitness == 0.0 for g in run.generations for e in g.evaluations)
    assert all(
        state == PPState({})
        for g in run.generations[1:]
        for state in g.input_states
    )


class _FixedRng:
    def __init__(self, *, random_values=(), samples=()):
        self.random_values = list(random_values)
        self.samples = [list(value) for value in samples]

    def random(self):
        return self.random_values.pop(0)

    def sample(self, population, count):
        value = self.samples.pop(0)
        assert len(value) == count
        return value


def test_fixed_parent_cut_and_swap_keep_positions_attached_to_source_box_ids():
    from packing.ga import _mutate_swap, _tournament_parent_index

    c = Container(8, 4, 2)
    boxes = {i: Box(i, 1, 1, 1) for i in (1, 2, 3, 4)}
    p1_positions = {i: Point(i - 1, 0, 0) for i in boxes}
    p2_positions = {i: Point(i - 1, 2, 0) for i in boxes}
    p1 = _parent_evaluation(c, boxes, (1, 2, 3, 4), p1_positions, 0.2)
    p2 = _parent_evaluation(c, boxes, (4, 3, 2, 1), p2_positions, 0.8)
    p3 = _parent_evaluation(c, boxes, (2, 1, 4, 3), p1_positions, 0.6)

    # Tournament candidates are explicitly indices 2 and 0; index 2 supplies
    # the selected parent because its fitness is higher. No RNG-derived oracle.
    selected = _tournament_parent_index((p1, p2, p3), _FixedRng(samples=([2, 0],)))
    assert selected == 2

    # Explicit cut [1:3]. child1 gets IDs 2,3 from p1 and the tail IDs 4,1
    # from p2, so every inherited position has an unambiguous source parent.
    child1, child2 = _crossover_pair_with_states(
        p1,
        p2,
        1.0,
        _FixedRng(random_values=(0.0,), samples=([1, 3],)),
    )
    assert child1[0].genes == (2, 3, 4, 1)
    assert child2[0].genes == (3, 2, 1, 4)
    assert child1[1][2] == p1_positions[2]
    assert child1[1][3] == p1_positions[3]
    assert child1[1][4] == p2_positions[4]
    assert child1[1][1] == p2_positions[1]
    assert child2[1][3] == p2_positions[3]
    assert child2[1][2] == p2_positions[2]
    assert child2[1][1] == p1_positions[1]
    assert child2[1][4] == p1_positions[4]

    # Explicit swap of sequence indices 0 and 3 changes only the permutation;
    # the ID-keyed inherited state is intentionally unchanged.
    mutated = _mutate_swap(
        child1[0],
        1.0,
        _FixedRng(random_values=(0.0,), samples=([0, 3],)),
    )
    assert mutated.genes == (1, 3, 4, 2)
    assert child1[1][1] == p2_positions[1]
    assert child1[1][2] == p1_positions[2]


def test_pp_ga_progress_switches_decoder_after_initial_generation_and_preserves_result():
    c = Container(3, 2, 1)
    boxes = (
        Box(1, 2, 1, 1),
        Box(2, 1, 2, 1),
        Box(3, 1, 1, 1),
    )
    config = GAConfig(population_size=3, generation_count=3, crossover_rate=0.8, mutation_rate=0.1)
    progress: list[GAProgress] = []

    observed = run_pp_ga(c, boxes, config=config, seed=17, progress_callback=progress.append)
    baseline = run_pp_ga(c, boxes, config=config, seed=17)

    assert observed == baseline
    assert len(progress) == 9
    assert [item.completed_evaluations for item in progress] == list(range(1, 10))
    assert [item.decoder_name for item in progress] == ["BL"] * 3 + ["PP"] * 6
    assert progress[-1].generation_index == 2
    assert progress[-1].population_index == 2
    assert progress[-1].fraction == 1.0
