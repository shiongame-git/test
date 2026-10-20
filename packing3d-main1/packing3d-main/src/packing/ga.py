"""Reference genetic algorithm for box-order search."""

from __future__ import annotations

import random
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from ._decoder_config import (
    DecoderConfig,
    DEFAULT_DECODER_CONFIG,
    resolve_decoder_config,
)
from .bl import BLPriority, pack
from .evaluation import DecoderInput, apply_cog_penalty
from .center_of_gravity import with_unity_random_weights
from .execution import EvaluationRuntime, EvaluationStats, GAExecutionConfig
from .io_run_metadata import RunMetadata
from .profile import (
    DecoderProfileCollector,
    DecoderProfileRecord,
)
from .run_context import RunOptions, begin_run
from .model import Box, Container, PackingResult


class EvaluationError(RuntimeError):
    """Raised when one GA individual cannot be evaluated normally."""

    def __init__(self, generation_index: int, population_index: int, cause: Exception) -> None:
        self.generation_index = generation_index
        self.population_index = population_index
        self.cause = cause
        super().__init__(
            f"evaluation failed at generation {generation_index}, "
            f"population index {population_index}: {cause}"
        )


@dataclass(frozen=True, slots=True)
class Individual:
    """A box-ID permutation. Problem-specific completeness is checked by evaluators."""

    genes: tuple[int, ...]

    def __post_init__(self) -> None:
        if type(self.genes) is not tuple:
            raise TypeError("genes must be a tuple")
        seen: set[int] = set()
        for index, box_id in enumerate(self.genes):
            if type(box_id) is not int:
                raise TypeError(f"genes[{index}] must be a Python int")
            if box_id < 0:
                raise ValueError(f"genes[{index}] must be >= 0")
            if box_id in seen:
                raise ValueError(f"duplicate box id in individual: {box_id}")
            seen.add(box_id)


@dataclass(frozen=True, slots=True)
class GAConfig:
    population_size: int = 100
    generation_count: int = 10
    crossover_rate: float = 1.0
    mutation_rate: float = 0.1
    use_center_of_gravity: bool = False
    center_of_gravity_lambda: float = 0.1
    unity_style_random_weights: bool = True

    def __post_init__(self) -> None:
        for name, value in (
            ("population_size", self.population_size),
            ("generation_count", self.generation_count),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be a Python int")
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        for name, value in (
            ("crossover_rate", self.crossover_rate),
            ("mutation_rate", self.mutation_rate),
        ):
            if type(value) is not float:
                raise TypeError(f"{name} must be a Python float")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if type(self.use_center_of_gravity) is not bool:
            raise TypeError("use_center_of_gravity must be a Python bool")
        if type(self.center_of_gravity_lambda) is not float:
            raise TypeError("center_of_gravity_lambda must be a Python float")
        if not math.isfinite(self.center_of_gravity_lambda) or self.center_of_gravity_lambda < 0.0:
            raise ValueError("center_of_gravity_lambda must be finite and >= 0.0")
        if type(self.unity_style_random_weights) is not bool:
            raise TypeError("unity_style_random_weights must be a Python bool")


@dataclass(frozen=True, slots=True)
class GAProgress:
    """Non-semantic progress snapshot for one GA run.

    Indices are zero-based. ``completed_evaluations`` counts requested
    individual evaluations that have been integrated by the parent process,
    including cache reuses. Progress reporting must not consume GA RNG state
    or change placement/evaluation semantics.
    """

    decoder_name: str
    generation_index: int
    generation_count: int
    population_index: int
    population_size: int
    completed_evaluations: int
    total_evaluations: int

    def __post_init__(self) -> None:
        if self.decoder_name not in ("BL", "PP"):
            raise ValueError("decoder_name must be BL or PP")
        for name, value in (
            ("generation_index", self.generation_index),
            ("generation_count", self.generation_count),
            ("population_index", self.population_index),
            ("population_size", self.population_size),
            ("completed_evaluations", self.completed_evaluations),
            ("total_evaluations", self.total_evaluations),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be a Python int")
        if self.generation_count < 1 or self.population_size < 1:
            raise ValueError("generation_count and population_size must be >= 1")
        if not 0 <= self.generation_index < self.generation_count:
            raise ValueError("generation_index is out of range")
        if not 0 <= self.population_index < self.population_size:
            raise ValueError("population_index is out of range")
        if self.total_evaluations != self.generation_count * self.population_size:
            raise ValueError("total_evaluations does not match run dimensions")
        if not 1 <= self.completed_evaluations <= self.total_evaluations:
            raise ValueError("completed_evaluations is out of range")

    @property
    def fraction(self) -> float:
        return self.completed_evaluations / self.total_evaluations


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    individual: Individual
    packing_result: PackingResult
    fitness: float

    def __post_init__(self) -> None:
        if type(self.individual) is not Individual:
            raise TypeError("individual must be an Individual")
        if type(self.packing_result) is not PackingResult:
            raise TypeError("packing_result must be a PackingResult")
        if type(self.fitness) is not float:
            raise TypeError("fitness must be a Python float")


@dataclass(frozen=True, slots=True)
class GenerationResult:
    generation_index: int
    evaluations: tuple[EvaluationResult, ...]
    best_index: int

    def __post_init__(self) -> None:
        if type(self.generation_index) is not int:
            raise TypeError("generation_index must be a Python int")
        if self.generation_index < 0:
            raise ValueError("generation_index must be >= 0")
        if type(self.evaluations) is not tuple:
            raise TypeError("evaluations must be a tuple")
        if not self.evaluations:
            raise ValueError("evaluations must not be empty")
        for index, evaluation in enumerate(self.evaluations):
            if type(evaluation) is not EvaluationResult:
                raise TypeError(f"evaluations[{index}] must be an EvaluationResult")
        if type(self.best_index) is not int:
            raise TypeError("best_index must be a Python int")
        if not 0 <= self.best_index < len(self.evaluations):
            raise ValueError("best_index is out of range")

    @property
    def best(self) -> EvaluationResult:
        return self.evaluations[self.best_index]


@dataclass(frozen=True, slots=True)
class GARunResult:
    container: Container
    boxes: tuple[Box, ...]
    priority: BLPriority
    config: GAConfig
    seed: int
    generations: tuple[GenerationResult, ...]
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
            if self.profile.decoder != "BL" or self.profile.decoder_config != self.decoder_config:
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
        return "BL"


def _require_seed(seed: object) -> int:
    if type(seed) is not int:
        raise TypeError("seed must be a Python int")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    return seed


def _freeze_problem_boxes(boxes: Iterable[Box]) -> tuple[Box, ...]:
    frozen = tuple(boxes)
    seen: set[int] = set()
    for index, box in enumerate(frozen):
        if type(box) is not Box:
            raise TypeError(f"boxes[{index}] must be a Box")
        if box.id in seen:
            raise ValueError(f"duplicate box id in problem: {box.id}")
        seen.add(box.id)
    return frozen


def _validate_individual_for_problem(individual: Individual, boxes: tuple[Box, ...]) -> None:
    expected = {box.id for box in boxes}
    actual = set(individual.genes)
    if len(individual.genes) != len(boxes) or actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise ValueError(
            "individual is not a permutation of the problem box IDs: "
            f"missing={missing}, unknown={unknown}"
        )


def evaluate_bl_individual(
    container: Container,
    boxes: Iterable[Box],
    individual: Individual,
    *,
    priority: BLPriority = BLPriority.BOTTOM_FIRST,
    decoder_config: DecoderConfig | None = None,
    use_center_of_gravity: bool = False,
    center_of_gravity_lambda: float = 0.1,
    _profile: DecoderProfileCollector | None = None,
) -> EvaluationResult:
    """Evaluate one ID permutation with BL and optional CoG-penalized fitness."""
    if type(container) is not Container:
        raise TypeError("container must be a Container")
    decoder_config = resolve_decoder_config(decoder_config)
    frozen_boxes = _freeze_problem_boxes(boxes)
    if type(individual) is not Individual:
        raise TypeError("individual must be an Individual")
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
    decoder_input = DecoderInput(container, ordered, None, decoder_config)
    if _profile is None:
        result = pack(
            decoder_input.container,
            decoder_input.ordered_boxes,
            priority=priority,
            decoder_config=decoder_input.decoder_config,
        )
    else:
        result = pack(
            decoder_input.container,
            decoder_input.ordered_boxes,
            priority=priority,
            decoder_config=decoder_input.decoder_config,
            _profile=_profile,
        )
    fitness = apply_cog_penalty(
        result, result.density, use_center_of_gravity, center_of_gravity_lambda
    )
    return EvaluationResult(individual, result, float(fitness))


def _best_index(evaluations: tuple[EvaluationResult, ...]) -> int:
    # max() would also work with an index tie key, but this makes the explicit
    # "first index wins ties" rule visible.
    best = 0
    for index in range(1, len(evaluations)):
        if evaluations[index].fitness > evaluations[best].fitness:
            best = index
    return best


def _initial_population(
    base_ids: tuple[int, ...], population_size: int, rng: random.Random
) -> tuple[Individual, ...]:
    population: list[Individual] = []
    for _ in range(population_size):
        genes = list(base_ids)
        rng.shuffle(genes)
        population.append(Individual(tuple(genes)))
    return tuple(population)


def _tournament_parent_index(
    evaluations: tuple[EvaluationResult, ...], rng: random.Random
) -> int:
    if len(evaluations) == 1:
        return 0
    a, b = rng.sample(range(len(evaluations)), 2)
    if evaluations[a].fitness > evaluations[b].fitness:
        return a
    if evaluations[b].fitness > evaluations[a].fitness:
        return b
    return min(a, b)


def _crossover_pair(
    parent1: Individual,
    parent2: Individual,
    rate: float,
    rng: random.Random,
) -> tuple[Individual, Individual]:
    if rng.random() >= rate or len(parent1.genes) < 2:
        return Individual(parent1.genes), Individual(parent2.genes)

    a, b = sorted(rng.sample(range(len(parent1.genes) + 1), 2))

    def make_child(front_parent: Individual, fill_parent: Individual) -> Individual:
        front = front_parent.genes[a:b]
        used = set(front)
        tail = tuple(gene for gene in fill_parent.genes if gene not in used)
        return Individual(front + tail)

    return make_child(parent1, parent2), make_child(parent2, parent1)


def _mutate_swap(individual: Individual, rate: float, rng: random.Random) -> Individual:
    if rng.random() >= rate or len(individual.genes) < 2:
        return Individual(individual.genes)
    i, j = rng.sample(range(len(individual.genes)), 2)
    genes = list(individual.genes)
    genes[i], genes[j] = genes[j], genes[i]
    return Individual(tuple(genes))


def _next_population(
    generation: GenerationResult,
    config: GAConfig,
    rng: random.Random,
) -> tuple[Individual, ...]:
    next_population: list[Individual] = [generation.best.individual]
    while len(next_population) < config.population_size:
        p1_index = _tournament_parent_index(generation.evaluations, rng)
        p2_index = _tournament_parent_index(generation.evaluations, rng)
        child1, child2 = _crossover_pair(
            generation.evaluations[p1_index].individual,
            generation.evaluations[p2_index].individual,
            config.crossover_rate,
            rng,
        )
        child1 = _mutate_swap(child1, config.mutation_rate, rng)
        child2 = _mutate_swap(child2, config.mutation_rate, rng)
        next_population.append(child1)
        if len(next_population) < config.population_size:
            next_population.append(child2)
    return tuple(next_population)


def run_bl_ga(
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
) -> GARunResult:
    """Run BL-GA with deterministic optional process evaluation and caching."""
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
        "BL",
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
    population = _initial_population(base_ids, config.population_size, rng)
    generations: list[GenerationResult] = []
    global_generation_index = 0
    global_population_index = 0
    global_fitness: float | None = None
    completed_evaluations = 0
    total_evaluations = config.population_size * config.generation_count

    with EvaluationRuntime(execution, profile_enabled=profile) as runtime:
        for generation_index in range(config.generation_count):
            def on_evaluated(population_index: int) -> None:
                nonlocal completed_evaluations
                completed_evaluations += 1
                if progress_callback is not None:
                    progress_callback(
                        GAProgress(
                            decoder_name="BL",
                            generation_index=generation_index,
                            generation_count=config.generation_count,
                            population_index=population_index,
                            population_size=config.population_size,
                            completed_evaluations=completed_evaluations,
                            total_evaluations=total_evaluations,
                        )
                    )

            frozen = runtime.evaluate_bl(
                container,
                frozen_boxes,
                population,
                priority=priority,
                generation_index=generation_index,
                decoder_config=decoder_config,
                use_center_of_gravity=config.use_center_of_gravity,
                center_of_gravity_lambda=config.center_of_gravity_lambda,
                progress_callback=on_evaluated,
            )
            best_index = _best_index(frozen)
            generation = GenerationResult(generation_index, frozen, best_index)
            generations.append(generation)

            if global_fitness is None or generation.best.fitness > global_fitness:
                global_fitness = generation.best.fitness
                global_generation_index = generation_index
                global_population_index = best_index

            if generation_index + 1 < config.generation_count:
                population = _next_population(generation, config, rng)

        execution_stats = runtime.stats

    run_metadata = context.complete(execution_stats)
    profile_record = None
    if profile:
        profile_metrics = runtime.profile_metrics
        assert profile_metrics is not None
        profile_record = DecoderProfileRecord(
            run_id=run_metadata["run_id"],
            decoder="BL",
            decoder_config=decoder_config,
            requested_evaluations=execution_stats.requested,
            computed_evaluations=execution_stats.computed,
            cache_hits=execution_stats.cache_hits,
            metrics=profile_metrics,
        )

    return GARunResult(
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
