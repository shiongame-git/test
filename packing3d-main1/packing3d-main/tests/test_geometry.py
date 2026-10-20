"""Tests for the decoder's shared AABB feasibility checks."""

from __future__ import annotations

import random

import packing.bl as bl_module
import packing.pp as pp_module
from packing import BLPriority, DecoderConfig, GAConfig, GAExecutionConfig, run_bl_ga, run_pp_ga
from packing.geometry import is_feasible_position
from packing.model import Box, Container, Placement, Point


def _reference_feasible_position(
    container: Container,
    box: Box,
    point: Point,
    placed: tuple[Placement, ...],
) -> bool:
    if (
        point.x + box.w > container.W
        or point.y + box.h > container.H
        or point.z + box.d > container.D
    ):
        return False

    for other in placed:
        if other.position is None:
            continue
        p = other.position
        b = other.box
        x_overlap = point.x < p.x + b.w and p.x < point.x + box.w
        y_overlap = point.y < p.y + b.h and p.y < point.y + box.h
        z_overlap = point.z < p.z + b.d and p.z < point.z + box.d
        if x_overlap and y_overlap and z_overlap:
            return False
    return True


def test_is_feasible_position_matches_independent_oracle_on_seeded_small_cases():
    rng = random.Random(20261008)
    for case_index in range(1000):
        container = Container(
            rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9)
        )
        box = Box(
            0, rng.randint(1, 10), rng.randint(1, 10), rng.randint(1, 10)
        )
        point = Point(rng.randint(0, 12), rng.randint(0, 12), rng.randint(0, 12))
        placed = tuple(
            Placement(
                Box(
                    index + 1,
                    rng.randint(1, 6),
                    rng.randint(1, 6),
                    rng.randint(1, 6),
                ),
                None
                if rng.randrange(8) == 0
                else Point(
                    rng.randint(0, 12),
                    rng.randint(0, 12),
                    rng.randint(0, 12),
                ),
            )
            for index in range(rng.randint(0, 12))
        )

        assert is_feasible_position(container, box, point, placed) == (
            _reference_feasible_position(container, box, point, placed)
        ), f"case={case_index} seed=20261008"


def test_is_feasible_position_allows_face_contact_but_rejects_positive_overlap():
    container = Container(4, 4, 4)
    box = Box(0, 2, 2, 2)
    touching = (
        Placement(Box(1, 2, 2, 2), Point(2, 0, 0)),
        Placement(Box(2, 2, 2, 2), Point(0, 2, 0)),
        Placement(Box(3, 2, 2, 2), Point(0, 0, 2)),
    )
    overlapping = (Placement(Box(4, 2, 2, 2), Point(1, 1, 1)),)

    assert is_feasible_position(container, box, Point(0, 0, 0), touching)
    assert not is_feasible_position(container, box, Point(0, 0, 0), overlapping)


def _point_signature(point: Point | None) -> tuple[int, int, int] | None:
    return None if point is None else (point.x, point.y, point.z)


def _run_semantics(run: object) -> tuple[object, ...]:
    generations = []
    for generation in run.generations:
        input_states = getattr(generation, "input_states", None)
        state_values = None if input_states is None else tuple(
            None
            if state is None
            else tuple(
                (box_id, _point_signature(position))
                for box_id, position in state.items_tuple()
            )
            for state in input_states
        )
        evaluations = tuple(
            (
                evaluation.individual.genes,
                evaluation.fitness,
                tuple(
                    (placement.box.id, _point_signature(placement.position))
                    for placement in evaluation.packing_result.placements
                ),
            )
            for evaluation in generation.evaluations
        )
        generations.append(
            (
                generation.best_index,
                getattr(generation, "decoder_name", "BL"),
                state_values,
                evaluations,
            )
        )
    return (
        run.best_generation_index,
        run.best_population_index,
        tuple(generations),
    )


def test_optimized_feasibility_matches_reference_full_ga_histories(monkeypatch):
    container = Container(7, 6, 5)
    boxes = (
        Box(9, 3, 2, 2),
        Box(2, 2, 3, 2),
        Box(7, 1, 1, 1),
        Box(4, 2, 1, 3),
        Box(12, 1, 2, 2),
        Box(3, 2, 2, 1),
        Box(1, 8, 1, 1),
    )
    config = GAConfig(4, 3, 0.8, 0.2)
    execution = GAExecutionConfig(1, False)

    for runner in (run_bl_ga, run_pp_ga):
        for policy in ("Cartesian", "LocalNeighbors"):
            for priority in BLPriority:
                for support_percent in (0, 55, 100):
                    decoder_config = DecoderConfig(
                        candidate_policy=policy,
                        min_support_percent=support_percent,
                    )
                    for seed in (0, 17, 31415):
                        optimized = runner(
                            container,
                            boxes,
                            priority=priority,
                            config=config,
                            seed=seed,
                            execution=execution,
                            decoder_config=decoder_config,
                        )
                        with monkeypatch.context() as reference_patch:
                            reference_patch.setattr(
                                bl_module,
                                "is_feasible_position",
                                _reference_feasible_position,
                            )
                            reference_patch.setattr(
                                pp_module,
                                "is_feasible_position",
                                _reference_feasible_position,
                            )
                            reference = runner(
                                container,
                                boxes,
                                priority=priority,
                                config=config,
                                seed=seed,
                                execution=execution,
                                decoder_config=decoder_config,
                            )

                        assert _run_semantics(optimized) == _run_semantics(reference), (
                            f"runner={runner.__name__} policy={policy} "
                            f"priority={priority.value} support={support_percent} "
                            f"seed={seed}"
                        )
