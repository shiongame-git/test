"""Cross-module public-path persistence and replay acceptance."""
from __future__ import annotations

import pytest

from packing import (
    BLPriority,
    Box,
    ComparisonDecoder,
    Container,
    DecoderConfig,
    GAConfig,
    GAExecutionConfig,
    RunOptions,
    read_execution_records,
    run_bl_ga,
    run_comparison_trial,
    run_manual_replay,
    run_pp_ga,
    write_comparison_report_json,
    write_run_metadata_json,
)


CONTAINER = Container(3, 2, 2)
BOXES = (Box(5, 1, 1, 1), Box(2, 1, 1, 2), Box(9, 1, 2, 1))
CONFIG = GAConfig(population_size=4, generation_count=2, crossover_rate=0.8, mutation_rate=0.1)
EXECUTION = GAExecutionConfig(workers=1, cache=True)
DECODER_CONFIG = DecoderConfig(
    candidate_policy="Cartesian",
    candidate_policy_version=1,
    min_support_percent=55,
    support_model_version="support-union-v1",
)


def run_signature(run):
    generations = []
    for generation in run.generations:
        states = getattr(generation, "input_states", None)
        generations.append((
            generation.best_index,
            getattr(generation, "decoder_name", "BL"),
            None if states is None else tuple(
                None if state is None else state.items_tuple() for state in states
            ),
            tuple((
                tuple(evaluation.individual.genes),
                evaluation.fitness,
                tuple((placement.box.id, placement.position) for placement in evaluation.packing_result.placements),
            ) for evaluation in generation.evaluations),
        ))
    return run.best_generation_index, run.best_population_index, tuple(generations)


def decoder_settings(metadata):
    decoder = metadata["decoder"]
    return DecoderConfig(
        candidate_policy=decoder["candidate_policy"],
        candidate_policy_version=decoder["candidate_policy_version"],
        min_support_percent=decoder["min_support_percent"],
        support_model_version=decoder["support_model_version"],
    )


@pytest.mark.parametrize("decoder", ["BL", "PP"])
@pytest.mark.parametrize("record_format", ["metadata", "comparison"])
def test_cartesian_metadata_or_report_save_reload_replay_pipeline(tmp_path, decoder, record_format):
    priority = BLPriority.BACK_FIRST
    if record_format == "metadata":
        runner = run_bl_ga if decoder == "BL" else run_pp_ga
        original_run = runner(
            CONTAINER,
            BOXES,
            priority=priority,
            config=CONFIG,
            seed=29,
            execution=EXECUTION,
            decoder_config=DECODER_CONFIG,
            run_options=RunOptions(source_boxes=BOXES, origin="api"),
        )
        original_signature = run_signature(original_run)
        path = tmp_path / f"{decoder.lower()}.run.json"
        write_run_metadata_json(path, original_run.run_metadata)
    else:
        trial = run_comparison_trial(
            CONTAINER,
            BOXES,
            decoder=ComparisonDecoder(decoder),
            priority=priority,
            config=CONFIG,
            seed=29,
            execution=EXECUTION,
            decoder_config=DECODER_CONFIG,
        )
        metadata = trial.run_metadata.to_dict()
        path = tmp_path / f"{decoder.lower()}.comparison.json"
        write_comparison_report_json(path, {
            "format": "packing3d-comparison-report",
            "version": 4,
            "implementation_version": metadata["environment"]["implementation"]["version"],
            "trials": [{
                "run_metadata": metadata,
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
            }],
        })

    records = read_execution_records(path)
    assert len(records) == 1
    original = records[0]
    before = original.to_dict()
    assert original["version"] == 2
    assert original["decoder"]["reference_version"] == f"{decoder}-ref-v2"
    assert decoder_settings(original) == DECODER_CONFIG

    replay = run_manual_replay(original)

    assert original.to_dict() == before
    assert replay.decoder_config == DECODER_CONFIG
    assert replay.run_metadata["run_id"] != original["run_id"]
    assert replay.run_metadata["replay_of"] == original["run_id"]
    assert replay.run_metadata["version"] == 2
    assert replay.run_metadata["decoder"]["reference_version"] == f"{decoder}-ref-v2"
    assert decoder_settings(replay.run_metadata) == DECODER_CONFIG
    assert replay.run_metadata["problem"] == original["problem"]
    assert replay.execution == EXECUTION
    if record_format == "metadata":
        assert run_signature(replay) == original_signature


@pytest.mark.parametrize("decoder", ["BL", "PP"])
def test_local_neighbors_comparison_v4_save_reload_replay_preserves_settings(
    tmp_path, decoder
):
    settings = DecoderConfig(
        candidate_policy="LocalNeighbors",
        candidate_policy_version=1,
        min_support_percent=55,
        support_model_version="support-union-v1",
    )
    trial = run_comparison_trial(
        CONTAINER,
        BOXES,
        decoder=ComparisonDecoder(decoder),
        priority=BLPriority.BACK_FIRST,
        config=CONFIG,
        seed=47,
        execution=EXECUTION,
        decoder_config=settings,
    )
    metadata = trial.run_metadata.to_dict()
    path = tmp_path / f"{decoder.lower()}-local-report.json"
    write_comparison_report_json(
        path,
        {
            "format": "packing3d-comparison-report",
            "version": 4,
            "implementation_version": metadata["environment"]["implementation"][
                "version"
            ],
            "trials": [
                {
                    "run_metadata": metadata,
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
            ],
        },
    )

    original = read_execution_records(path)[0]
    original_data = original.to_dict()
    assert decoder_settings(original) == settings
    replay = run_manual_replay(original)

    assert original.to_dict() == original_data
    assert replay.decoder_config == settings
    assert replay.run_metadata["decoder"]["candidate_policy"] == "LocalNeighbors"
    assert replay.run_metadata["decoder"]["candidate_policy_version"] == 1
    assert replay.run_metadata["decoder"]["min_support_percent"] == 55
    assert replay.run_metadata["decoder"]["support_model_version"] == "support-union-v1"
    assert replay.run_metadata["replay_of"] == original["run_id"]
    assert replay.run_metadata["run_id"] != original["run_id"]
