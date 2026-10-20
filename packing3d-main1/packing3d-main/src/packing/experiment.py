"""Comparison helpers for reproducible BL-GA / PP-GA experiments.

This module is deliberately separate from decoder/evaluator semantics.  It fixes
experimental ordering and records the complete trial settings used for a
comparison; it does not change packing rules or fitness.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

from ._decoder_config import (
    DecoderConfig,
    DEFAULT_DECODER_CONFIG,
    resolve_decoder_config,
)
from .bl import BLPriority
from .execution import EvaluationStats, GAExecutionConfig
from .io_run_metadata import RunMetadata
from .run_context import RunOptions
from .source import SOURCE_LABEL
from .ga import GAConfig, GARunResult, _freeze_problem_boxes, run_bl_ga
from .model import Box, Container
from .pp_ga import PPGARunResult, run_pp_ga
from .profile import DecoderProfileRecord
from .validate import validate_result, validate_support


class ComparisonDecoder(Enum):
    BL = "BL"
    PP = "PP"


class BoxOrderMode(Enum):
    SOURCE = "source"
    DEPTH_DESC_STABLE = "depth_desc_stable"


@dataclass(frozen=True, slots=True)
class ExperimentEnvironment:
    implementation_version: str
    python: str
    cpu: str
    os: str

    def __post_init__(self) -> None:
        for name, value in (
            ("implementation_version", self.implementation_version),
            ("python", self.python),
            ("cpu", self.cpu),
            ("os", self.os),
        ):
            if type(value) is not str:
                raise TypeError(f"{name} must be a str")
            if not value:
                raise ValueError(f"{name} must not be empty")


@dataclass(frozen=True, slots=True)
class ComparisonProblemSignature:
    container: tuple[int, int, int]
    source_boxes: tuple[tuple[int, int, int, int], ...]

    def __post_init__(self) -> None:
        if type(self.container) is not tuple or len(self.container) != 3:
            raise TypeError("container must be a 3-tuple")
        for index, value in enumerate(self.container):
            if type(value) is not int:
                raise TypeError(f"container[{index}] must be a Python int")
            if value < 1:
                raise ValueError(f"container[{index}] must be >= 1")
        if type(self.source_boxes) is not tuple:
            raise TypeError("source_boxes must be a tuple")
        seen: set[int] = set()
        for index, box in enumerate(self.source_boxes):
            if type(box) is not tuple or len(box) != 4:
                raise TypeError(f"source_boxes[{index}] must be a 4-tuple")
            box_id, w, h, d = box
            for name, value, minimum in (("id", box_id, 0), ("w", w, 1), ("h", h, 1), ("d", d, 1)):
                if type(value) is not int:
                    raise TypeError(f"source_boxes[{index}].{name} must be a Python int")
                if value < minimum:
                    raise ValueError(f"source_boxes[{index}].{name} must be >= {minimum}")
            if box_id in seen:
                raise ValueError(f"duplicate box id in source_boxes: {box_id}")
            seen.add(box_id)


@dataclass(frozen=True, slots=True)
class ComparisonTrialResult:
    decoder: ComparisonDecoder
    priority: BLPriority
    order_mode: BoxOrderMode
    problem_signature: ComparisonProblemSignature
    effective_box_order: tuple[int, ...]
    problem_source: str | None
    fixture_sha256: str | None
    config: GAConfig
    seed: int
    execution: GAExecutionConfig
    best_generation_index: int
    best_population_index: int
    best_genes: tuple[int, ...]
    best_fitness: float
    placed_count: int
    failed_count: int
    placed_volume: int
    density: float
    execution_stats: EvaluationStats
    elapsed_seconds: float = field(compare=False)
    environment: ExperimentEnvironment = field(compare=False)
    run_metadata: RunMetadata | None = field(default=None, compare=False)
    decoder_config: DecoderConfig = DEFAULT_DECODER_CONFIG
    profile: DecoderProfileRecord | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        if type(self.decoder) is not ComparisonDecoder:
            raise TypeError("decoder must be a ComparisonDecoder")
        if type(self.priority) is not BLPriority:
            raise TypeError("priority must be a BLPriority")
        if type(self.order_mode) is not BoxOrderMode:
            raise TypeError("order_mode must be a BoxOrderMode")
        if type(self.decoder_config) is not DecoderConfig:
            raise TypeError("decoder_config must be a DecoderConfig")
        if type(self.problem_signature) is not ComparisonProblemSignature:
            raise TypeError("problem_signature must be a ComparisonProblemSignature")
        if type(self.effective_box_order) is not tuple or any(type(value) is not int for value in self.effective_box_order):
            raise TypeError("effective_box_order must be a tuple of Python ints")
        source_order = tuple(box[0] for box in self.problem_signature.source_boxes)
        if len(self.effective_box_order) != len(source_order) or set(self.effective_box_order) != set(source_order):
            raise ValueError("effective_box_order must be a permutation of source box IDs")
        if self.problem_source is not None:
            if type(self.problem_source) is not str:
                raise TypeError("problem_source must be a str or None")
            if not self.problem_source:
                raise ValueError("problem_source must not be empty")
        if self.fixture_sha256 is not None:
            if type(self.fixture_sha256) is not str:
                raise TypeError("fixture_sha256 must be a str or None")
            digest = self.fixture_sha256.lower()
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("fixture_sha256 must be a 64-character hexadecimal SHA-256 digest")
        if type(self.config) is not GAConfig:
            raise TypeError("config must be a GAConfig")
        if type(self.seed) is not int:
            raise TypeError("seed must be a Python int")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if type(self.execution) is not GAExecutionConfig:
            raise TypeError("execution must be a GAExecutionConfig")
        if type(self.best_genes) is not tuple:
            raise TypeError("best_genes must be a tuple")
        if type(self.best_fitness) is not float:
            raise TypeError("best_fitness must be a Python float")
        if type(self.density) is not float:
            raise TypeError("density must be a Python float")
        if type(self.execution_stats) is not EvaluationStats:
            raise TypeError("execution_stats must be an EvaluationStats")
        if type(self.elapsed_seconds) is not float:
            raise TypeError("elapsed_seconds must be a Python float")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be >= 0")
        if type(self.environment) is not ExperimentEnvironment:
            raise TypeError("environment must be an ExperimentEnvironment")
        if self.profile is not None:
            if type(self.profile) is not DecoderProfileRecord:
                raise TypeError("profile must be a DecoderProfileRecord or None")
            if self.profile.decoder != self.decoder.value:
                raise ValueError("profile decoder does not match comparison decoder")
            if self.profile.decoder_config != self.decoder_config:
                raise ValueError("profile decoder config does not match comparison config")
            if self.run_metadata is None or self.profile.run_id != self.run_metadata["run_id"]:
                raise ValueError("profile run_id does not match comparison run metadata")

    @property
    def source_box_order(self) -> tuple[int, ...]:
        return tuple(box[0] for box in self.problem_signature.source_boxes)

    @property
    def requested_evaluation_budget(self) -> int:
        return self.config.population_size * self.config.generation_count


def _cpu_name() -> str:
    name = platform.processor().strip()
    if name:
        return name
    return os.environ.get("PROCESSOR_IDENTIFIER", "unknown") or "unknown"


def current_environment(*, implementation_version: str) -> ExperimentEnvironment:
    if type(implementation_version) is not str:
        raise TypeError("implementation_version must be a str")
    if not implementation_version:
        raise ValueError("implementation_version must not be empty")
    return ExperimentEnvironment(
        implementation_version=implementation_version,
        python=f"{platform.python_implementation()} {platform.python_version()}",
        cpu=_cpu_name(),
        os=platform.platform(),
    )


def order_boxes_for_experiment(
    boxes: Iterable[Box],
    mode: BoxOrderMode,
) -> tuple[Box, ...]:
    """Return the explicit problem order used to seed GA initialization.

    ``DEPTH_DESC_STABLE`` means descending box depth ``d`` with source order
    preserved for ties.  It is an experiment factor only: neither BL nor PP
    performs an internal box sort.
    """
    frozen = _freeze_problem_boxes(boxes)
    if type(mode) is not BoxOrderMode:
        raise TypeError("mode must be a BoxOrderMode")
    if mode is BoxOrderMode.SOURCE:
        return frozen
    return tuple(sorted(frozen, key=lambda box: -box.d))


def _problem_signature(container: Container, boxes: tuple[Box, ...]) -> ComparisonProblemSignature:
    return ComparisonProblemSignature(
        container=(container.W, container.H, container.D),
        source_boxes=tuple((box.id, box.w, box.h, box.d) for box in boxes),
    )


def _validate_best_result(
    container: Container,
    boxes: tuple[Box, ...],
    run: GARunResult | PPGARunResult,
) -> None:
    by_id = {box.id: box for box in boxes}
    ordered = tuple(by_id[box_id] for box_id in run.best.individual.genes)
    errors = validate_result(
        run.best.packing_result,
        expected_container=container,
        ordered_boxes=ordered,
    )
    if errors:
        raise ValueError("best GA result failed validation: " + "; ".join(errors))
    support_errors = validate_support(run.best.packing_result, settings=run.decoder_config)
    if support_errors:
        raise ValueError(
            "best GA result failed support validation: " + "; ".join(support_errors)
        )


def run_comparison_trial(
    container: Container,
    boxes: Iterable[Box],
    *,
    decoder: ComparisonDecoder,
    priority: BLPriority = BLPriority.BOTTOM_FIRST,
    order_mode: BoxOrderMode = BoxOrderMode.SOURCE,
    config: GAConfig = GAConfig(),
    seed: int = 0,
    execution: GAExecutionConfig = GAExecutionConfig(),
    implementation_version: str = SOURCE_LABEL,
    origin: str = "api",
    problem_source: str | None = None,
    fixture_sha256: str | None = None,
    decoder_config: DecoderConfig | None = None,
    profile: bool = False,
) -> ComparisonTrialResult:
    """Run one fully specified comparison trial and validate its best result.

    For algorithm comparisons, callers should normally use the same ``config``,
    ``seed``, ``priority``, ``order_mode`` and ``execution`` for BL and PP.  A
    cache-disabled execution makes requested and computed evaluation budgets
    identical and is the reference comparison condition.
    """
    if type(container) is not Container:
        raise TypeError("container must be a Container")
    decoder_config = resolve_decoder_config(decoder_config)
    if type(decoder) is not ComparisonDecoder:
        raise TypeError("decoder must be a ComparisonDecoder")
    if type(priority) is not BLPriority:
        raise TypeError("priority must be a BLPriority")
    if type(config) is not GAConfig:
        raise TypeError("config must be a GAConfig")
    if type(execution) is not GAExecutionConfig:
        raise TypeError("execution must be a GAExecutionConfig")
    if type(profile) is not bool:
        raise TypeError("profile must be a bool")
    if type(seed) is not int:
        raise TypeError("seed must be a Python int")
    if seed < 0:
        raise ValueError("seed must be >= 0")
    if problem_source is not None:
        if type(problem_source) is not str:
            raise TypeError("problem_source must be a str or None")
        if not problem_source:
            raise ValueError("problem_source must not be empty")
    if fixture_sha256 is not None:
        if type(fixture_sha256) is not str:
            raise TypeError("fixture_sha256 must be a str or None")
        digest = fixture_sha256.lower()
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError("fixture_sha256 must be a 64-character hexadecimal SHA-256 digest")

    source_boxes = _freeze_problem_boxes(boxes)
    ordered_boxes = order_boxes_for_experiment(source_boxes, order_mode)
    problem_signature = _problem_signature(container, source_boxes)
    options = RunOptions(origin=origin, source_boxes=source_boxes, order_mode=order_mode.value,
                         source=problem_source, fixture_sha256=None if fixture_sha256 is None else fixture_sha256.lower(),
                         implementation_version=implementation_version)
    if decoder is ComparisonDecoder.BL:
        run = run_bl_ga(
            container,
            ordered_boxes,
            priority=priority,
            config=config,
            seed=seed,
            execution=execution,
            run_options=options,
            decoder_config=decoder_config,
            profile=profile,
        )
    else:
        run = run_pp_ga(
            container,
            ordered_boxes,
            priority=priority,
            config=config,
            seed=seed,
            execution=execution,
            run_options=options,
            decoder_config=decoder_config,
            profile=profile,
        )
    metadata = run.run_metadata
    elapsed = metadata["statistics"]["elapsed_seconds"]
    env = metadata["environment"]

    _validate_best_result(container, ordered_boxes, run)
    best = run.best
    result = best.packing_result
    return ComparisonTrialResult(
        decoder=decoder,
        priority=priority,
        order_mode=order_mode,
        decoder_config=decoder_config,
        problem_signature=problem_signature,
        effective_box_order=tuple(box.id for box in ordered_boxes),
        problem_source=problem_source,
        fixture_sha256=None if fixture_sha256 is None else fixture_sha256.lower(),
        config=config,
        seed=seed,
        execution=execution,
        best_generation_index=run.best_generation_index,
        best_population_index=run.best_population_index,
        best_genes=best.individual.genes,
        best_fitness=best.fitness,
        placed_count=result.placed_count,
        failed_count=result.failed_count,
        placed_volume=result.placed_volume,
        density=result.density,
        execution_stats=run.execution_stats,
        elapsed_seconds=float(elapsed),
        environment=ExperimentEnvironment(implementation_version,
            env["python"]["implementation"] + " " + env["python"]["version"],
            env["cpu"]["label"] or "unknown", env["os"]["system"]),
        run_metadata=metadata,
        profile=run.profile,
    )
