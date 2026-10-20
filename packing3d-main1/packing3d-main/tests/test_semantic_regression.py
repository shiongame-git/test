"""v1.1 Cartesian/0% GA behavior through the explicit v1.2 public settings.

This fixture is test data, not a supported full-history interchange format.
Keep all four decoder inputs explicit and do not regenerate the fixture just
to make an algorithm change pass.
"""

import json
from pathlib import Path

import pytest

from packing import (
    BLPriority,
    Box,
    Container,
    DecoderConfig,
    GAConfig,
    GAExecutionConfig,
    run_bl_ga,
    run_pp_ga,
)

FIXTURE = Path(__file__).parent / "fixtures" / "ga_semantics.json"
CONTAINER = Container(3, 2, 2)
BOXES = (Box(7, 2, 1, 1), Box(2, 1, 2, 1), Box(5, 1, 1, 2), Box(1, 2, 2, 1))
CONFIG = GAConfig(4, 3, 0.8, 0.2)
SEED = 31415
LEGACY_CARTESIAN_ZERO = DecoderConfig(
    candidate_policy="Cartesian",
    candidate_policy_version=1,
    min_support_percent=0,
    support_model_version="support-union-v1",
)


def point_value(point):
    return None if point is None else [point.x, point.y, point.z]


def semantic_value(run):
    """Exclude timing, UUID, CPU observations and execution/cache counters."""
    generations = []
    for generation in run.generations:
        states = getattr(generation, "input_states", None)
        generations.append({
            "best_index": generation.best_index,
            "decoder": getattr(generation, "decoder_name", "BL"),
            "input_states": None if states is None else [
                None if state is None else [
                    [box_id, point_value(position)] for box_id, position in state.items_tuple()
                ] for state in states
            ],
            "evaluations": [
                {
                    "genes": list(evaluation.individual.genes),
                    "fitness": evaluation.fitness,
                    "placements": [
                        [p.box.id, point_value(p.position)]
                        for p in evaluation.packing_result.placements
                    ],
                } for evaluation in generation.evaluations
            ],
        })
    return {
        "best_generation_index": run.best_generation_index,
        "best_population_index": run.best_population_index,
        "generations": generations,
    }


@pytest.mark.parametrize("runner", [run_bl_ga, run_pp_ga], ids=["BL", "PP"])
@pytest.mark.parametrize("priority", list(BLPriority), ids=lambda p: p.value)
@pytest.mark.parametrize("workers,cache", [(1, False), (1, True), (2, False), (2, True)])
def test_legacy_cartesian_zero_full_history_matches_reference(runner, priority, workers, cache):
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    run = runner(
        CONTAINER, BOXES, config=CONFIG, seed=SEED, priority=priority,
        execution=GAExecutionConfig(workers, cache),
        decoder_config=LEGACY_CARTESIAN_ZERO,
    )
    key = f"{run.decoder_name}/{priority.value}"
    assert semantic_value(run) == expected["runs"][key]
