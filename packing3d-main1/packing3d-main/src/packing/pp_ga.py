"""PP-GA using BL initialization and the PP decoder thereafter."""

from __future__ import annotations

import random
import math
from dataclasses import dataclass, field
from collections.abc import Callable, Iterable

from ._decoder_config import (
    DecoderConfig,
    DEFAULT_DECODER_CONFIG,
    resolve_decoder_config,
)
from .bl import BLPriority
from .evaluation import DecoderInput, apply_cog_penalty
from .center_of_gravity import with_unity_random_weights
from .execution import EvaluationRuntime, EvaluationStats, GAExecutionConfig
from .io_run_metadata import RunMetadata
from .profile import DecoderProfileCollector, DecoderProfileRecord
from .run_context import RunOptions, begin_run
from .ga import (
    EvaluationResult,
    GAConfig,
    GAProgress,
    Individual,
    _best_index,
    _freeze_problem_boxes,
    _initial_population,
    _mutate_swap,
    _require_seed,
    _tournament_parent_index,
    _validate_individual_for_problem,
)
from .model import Box, Container, PackingResult
from .pp import pack_pp
from .pp_state import PPState


@dataclass(frozen=True, slots=True)
class PPGenerationResult:
    """One evaluated PP-GA generation plus the decoder input states used."""

    generation_index: int
    decoder_name: str
    evaluations: tuple[EvaluationResult, ...]
    input_states: tuple[PPState | None, ...]
    best_index: int

    def __post_init__(self) -> None:
        if type(self.generation_index) is not int:
            raise TypeError("generation_index must be a Python int")
        if self.generation_index < 0:
            raise ValueError("generation_index must be >= 0")
        if self.decoder_name not in ("BL", "PP"):
            raise ValueError("decoder_name must be 'BL' or 'PP'")
        if type(self.evaluations) is not tuple:
            raise TypeError("evaluations must be a tuple")
        if not self.evaluations:
            raise ValueError("evaluations must not be empty")
        for index, evaluation in enumerate(self.evaluations):
            if type(evaluation) is not EvaluationResult:
                raise TypeError(f"evaluations[{index}] must be an EvaluationResult")
        if type(self.input_states) is not tuple:
            raise TypeError("input_states must be a tuple")
        if len(self.input_states) != len(self.evaluations):
            raise ValueError("input_states length must match evaluations length")
        if self.decoder_name == "BL":
            if any(state is not None for state in self.input_states):
                raise ValueError("BL generation input states must all be None")
        else:
            for index, state in enumerate(self.input_states):
                if type(state) is not PPState:
                    raise TypeError(f"input_states[{index}] must be a PPState for PP generations")
        if type(self.best_index) is not int:
            raise TypeError("best_index must be a Python int")
        if not 0 <= self.best_index < len(self.evaluations):
            raise ValueError("best_index is out of range")

    @property
    def best(self) -> EvaluationResult:
        return self.evaluations[self.best_index]


@dataclass(frozen=True, slots=True)
class PPGARunResult:
    container: Container
    boxes: tuple[Box, ...]
    priority: BLPriority
    config: GAConfig
    seed: int
    generations: tuple[PPGenerationResult, ...]
    best_generation_index: int
    best_population_index: int
    execution: GAExecutionConfig = field(default_factory=GAExecutionConfig, compare=False)
    execution_stats: EvaluationStats = field(default_factory=EvaluationStats, compare=False)
    run_metadata: RunMetadata | None = field(default=None, compare=False)
    decoder_config: DecoderConfig = DEFAULT_DECODER_CONFIG
    profile: DecoderProfileRecord | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if type(self.container) is not Container:
            raise TypeError("container must be a Container")
        if type(self.boxes) is not tuple:
            raise TypeError("boxes must be a tuple")
        if type(self.priority) is not BLPriority:
            raise TypeError("priority must be a BLPriority")
        if type(self.config) is not GAConfig:
            raise TypeError("config must be a GAConfig")
        _require_seed(self.seed)
        if type(self.generations) is not tuple:
            raise TypeError("generations must be a tuple")
        if len(self.generations) != self.config.generation_count:
            raise ValueError("generation result count does not match config")
        if type(self.best_generation_index) is not int:
            raise TypeError("best_generation_index must be a Python int")
        if not 0 <= self.best_generation_index < len(self.generations):
            raise ValueError("best_generation_index is out of range")
        generation = self.generations[self.best_generation_index]
        if type(self.best_population_index) is not int:
            raise TypeError("best_population_index must be a Python int")
        if not 0 <= self.best_population_index < len(generation.evaluations):
            raise ValueError("best_population_index is out of range")
        if type(self.execution) is not GAExecutionConfig:
            raise TypeError("execution must be a GAExecutionConfig")
        if type(self.execution_stats) is not EvaluationStats:
            raise TypeError("execution_stats must be an EvaluationStats")
        if type(self.decoder_config) is not DecoderConfig:
            raise TypeError("decoder_config must be a DecoderConfig")
        if self.profile is not None:
            if type(self.profile) is not DecoderProfileRecord:
                raise TypeError("profile must be a DecoderProfileRecord or None")
            if self.profile.decoder != "PP" or self.profile.decoder_config != self.decoder_config:
                raise ValueError("profile decoder identity does not match the run result")
            if self.run_metadata is None or self.profile.run_id != self.run_metadata["run_id"]:
                raise ValueError("profile run_id does not match run metadata")
            if (
                self.profile.requested_evaluations != self.execution_stats.requested
                or self.profile.computed_evaluations != self.execution_stats.computed
                or self.profile.cache_hits != self.execution_stats.cache_hits
            ):
                raise ValueError("profile evaluation counts do not match execution stats")

    @property
    def best(self) -> EvaluationResult:
        return self.generations[self.best_generation_index].evaluations[self.best_population_index]

    @property
    def decoder_name(self) -> str:
        return "PP"


def pp_state_from_result(result: PackingResult) -> PPState:
    """Create the ID-keyed inherited state represented by a completed result."""
    if type(result) is not PackingResult:
        raise TypeError("result must be a PackingResult")
    return PPState((placement.box.id, placement.position) for placement in result.placements)


def evaluate_pp_individual(
    container: Container,
    boxes: Iterable[Box],
    individual: Individual,
    state: PPState,
    *,
    priority: BLPriority = BLPriority.BOTTOM_FIRST,
    decoder_config: DecoderConfig | None = None,
    use_center_of_gravity: bool = False,
    center_of_gravity_lambda: float = 0.1,
    _profile: DecoderProfileCollector | None = None,
) -> EvaluationResult:
    """Evaluate one ID permutation with PP and optional CoG fitness penalty."""
    if type(container) is not Container:
        raise TypeError("container must be a Container")
    decoder_config = resolve_decoder_config(decoder_config)
    frozen_boxes = _freeze_problem_boxes(boxes)
    if type(individual) is not Individual:
        raise TypeError("individual must be an Individual")
    if type(state) is not PPState:
        raise TypeError("state must be a PPState")
    if type(priority) is not BLPriority:
        raise TypeError("priority must be a BLPriority")
    if type(use_center_of_gravity) is not bool:
        raise TypeError("use_center_of_gravity must be a Python bool")
    if type(center_of_gravity_lambda) is not float:
        raise TypeError("center_of_gravity_lambda must be a Python float")
    if not math.isfinite(center_of_gravity_lambda) or center_of_gravity_lambda < 0.0:
        raise ValueError("center_of_gravity_lambda must be finite and >= 0.0")
    _validate_individual_for_problem(individual, frozen_boxes)

    by_id = {box.id: box for box in frozen_boxes}
    ordered = tuple(by_id[box_id] for box_id in individual.genes)
    decoder_input = DecoderInput(container, ordered, state, decoder_config)
    if _profile is None:
        result = pack_pp(
            decoder_input.container,
            decoder_input.ordered_boxes,
            decoder_input.decoder_state,
            priority=priority,
            decoder_config=decoder_input.decoder_config,
        )
    else:
        result = pack_pp(
            decoder_input.container,
            decoder_input.ordered_boxes,
            decoder_input.decoder_state,
            priority=priority,
            decoder_config=decoder_input.decoder_config,
            _profile=_profile,
        )
    fitness = apply_cog_penalty(
        result, result.density, use_center_of_gravity, center_of_gravity_lambda
    )
    return EvaluationResult(individual, result, float(fitness))


def _crossover_pair_with_states(
    parent1: EvaluationResult,
    parent2: EvaluationResult,
    rate: float,
    rng: random.Random,
) -> tuple[tuple[Individual, PPState], tuple[Individual, PPState]]:
    """Normal crossover plus ID-keyed position inheritance from gene sources."""
    p1 = parent1.individual
    p2 = parent2.individual
    s1 = pp_state_from_result(parent1.packing_result)
    s2 = pp_state_from_result(parent2.packing_result)

    if rng.random() >= rate or len(p1.genes) < 2:
        return (Individual(p1.genes), s1), (Individual(p2.genes), s2)

    a, b = sorted(rng.sample(range(len(p1.genes) + 1), 2))

    def make_child(
        front_parent: Individual,
        fill_parent: Individual,
        front_state: PPState,
        fill_state: PPState,
    ) -> tuple[Individual, PPState]:
        front = front_parent.genes[a:b]
        used = set(front)
        tail = tuple(gene for gene in fill_parent.genes if gene not in used)
        genes = front + tail
        inherited = []
        for gene in genes:
            source = front_state if gene in used else fill_state
            inherited.append((gene, source[gene]))
        return Individual(genes), PPState(inherited)

    return (
        make_child(p1, p2, s1, s2),
        make_child(p2, p1, s2, s1),
    )


def _next_pp_population(
    generation: PPGenerationResult,
    config: GAConfig,
    rng: random.Random,
) -> tuple[tuple[Individual, PPState], ...]:
    next_population: list[tuple[Individual, PPState]] = [
        (
            generation.best.individual,
            pp_state_from_result(generation.best.packing_result),
        )
    ]
    while len(next_population) < config.population_size:
        p1_index = _tournament_parent_index(generation.evaluations, rng)
        p2_index = _tournament_parent_index(generation.evaluations, rng)
        child1, child2 = _crossover_pair_with_states(
            generation.evaluations[p1_index],
            generation.evaluations[p2_index],
            config.crossover_rate,
            rng,
        )
        child1_individual = _mutate_swap(child1[0], config.mutation_rate, rng)
        child2_individual = _mutate_swap(child2[0], config.mutation_rate, rng)
        # Mutation changes only the permutation.  Inherited positions remain
        # attached to box IDs rather than to sequence indices.
        next_population.append((child1_individual, child1[1]))
        if len(next_population) < config.population_size:
            next_population.append((child2_individual, child2[1]))
    return tuple(next_population)


def run_pp_ga(
    container: Container,
    boxes: Iterable[Box],
    *,
    priority: BLPriority = BLPriority.BOTTOM_FIRST,
    config: GAConfig = GAConfig(),
    seed: int = 0,
    execution: GAExecutionConfig = GAExecutionConfig(),
    progress_callback: Callable[[GAProgress], None] | None = None,
    run_options: RunOptions | None = None,
    decoder_config: DecoderConfig | None = None,
    profile: bool = False,
) -> PPGARunResult:
    """Run PP-GA with deterministic optional process evaluation and caching."""
    if type(container) is not Container:
        raise TypeError("container must be a Container")
    decoder_config = resolve_decoder_config(decoder_config)
    frozen_boxes = _freeze_problem_boxes(boxes)
    if type(priority) is not BLPriority:
        raise TypeError("priority must be a BLPriority")
    if type(config) is not GAConfig:
        raise TypeError("config must be a GAConfig")
    if type(execution) is not GAExecutionConfig:
        raise TypeError("execution must be a GAExecutionConfig")
    seed = _require_seed(seed)
    if type(profile) is not bool:
        raise TypeError("profile must be a Python bool")

    context = begin_run(
        container,
        frozen_boxes,
        "PP",
        priority,
        config,
        seed,
        execution,
        decoder_config,
        run_options,
    )
    execution = context.execution
    if config.use_center_of_gravity and config.unity_style_random_weights:
        frozen_boxes = with_unity_random_weights(frozen_boxes, seed)

    base_ids = tuple(box.id for box in frozen_boxes)
    rng = random.Random(seed)
    initial = _initial_population(base_ids, config.population_size, rng)
    population: tuple[tuple[Individual, PPState | None], ...] = tuple(
        (individual, None) for individual in initial
    )

    generations: list[PPGenerationResult] = []
    global_generation_index = 0
    global_population_index = 0
    global_fitness: float | None = None
    completed_evaluations = 0
    total_evaluations = config.population_size * config.generation_count

    with EvaluationRuntime(execution, profile_enabled=profile) as runtime:
        for generation_index in range(config.generation_count):
            decoder_name = "BL" if generation_index == 0 else "PP"
            input_states = tuple(state for _, state in population)

            def on_evaluated(population_index: int) -> None:
                nonlocal completed_evaluations
                completed_evaluations += 1
                if progress_callback is not None:
                    progress_callback(
                        GAProgress(
                            decoder_name=decoder_name,
                            generation_index=generation_index,
                            generation_count=config.generation_count,
                            population_index=population_index,
                            population_size=config.population_size,
                            completed_evaluations=completed_evaluations,
                            total_evaluations=total_evaluations,
                        )
                    )

            if generation_index == 0:
                if any(state is not None for state in input_states):
                    raise ValueError("initial BL generation must not have a PPState")
                individuals = tuple(individual for individual, _ in population)
                frozen_evaluations = runtime.evaluate_bl(
                    container,
                    frozen_boxes,
                    individuals,
                    priority=priority,
                    generation_index=generation_index,
                    decoder_config=decoder_config,
                    use_center_of_gravity=config.use_center_of_gravity,
                    center_of_gravity_lambda=config.center_of_gravity_lambda,
                    progress_callback=on_evaluated,
                )
            else:
                pp_population: list[tuple[Individual, PPState]] = []
                for individual, state in population:
                    if type(state) is not PPState:
                        raise TypeError("PP generation requires a PPState")
                    pp_population.append((individual, state))
                frozen_evaluations = runtime.evaluate_pp(
                    container,
                    frozen_boxes,
                    tuple(pp_population),
                    priority=priority,
                    generation_index=generation_index,
                    decoder_config=decoder_config,
                    use_center_of_gravity=config.use_center_of_gravity,
                    center_of_gravity_lambda=config.center_of_gravity_lambda,
                    progress_callback=on_evaluated,
                )

            best_index = _best_index(frozen_evaluations)
            generation = PPGenerationResult(
                generation_index=generation_index,
                decoder_name=decoder_name,
                evaluations=frozen_evaluations,
                input_states=input_states,
                best_index=best_index,
            )
            generations.append(generation)

            if global_fitness is None or generation.best.fitness > global_fitness:
                global_fitness = generation.best.fitness
                global_generation_index = generation_index
                global_population_index = best_index

            if generation_index + 1 < config.generation_count:
                population = _next_pp_population(generation, config, rng)

        execution_stats = runtime.stats

    run_metadata = context.complete(execution_stats)
    profile_record = None
    if profile:
        profile_metrics = runtime.profile_metrics
        assert profile_metrics is not None
        profile_record = DecoderProfileRecord(
            run_id=run_metadata["run_id"],
            decoder="PP",
            decoder_config=decoder_config,
            requested_evaluations=execution_stats.requested,
            computed_evaluations=execution_stats.computed,
            cache_hits=execution_stats.cache_hits,
            metrics=profile_metrics,
        )

    return PPGARunResult(
        container=container,
        boxes=frozen_boxes,
        priority=priority,
        config=config,
        seed=seed,
        generations=tuple(generations),
        best_generation_index=global_generation_index,
        best_population_index=global_population_index,
        execution=execution,
        execution_stats=execution_stats,
        run_metadata=run_metadata,
        decoder_config=decoder_config,
        profile=profile_record,
    )
