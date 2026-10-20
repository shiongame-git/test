"""Deterministic evaluation runtime for GA decoding.

The GA owns all randomness.  This module only evaluates already-fixed decoder
inputs, optionally using process workers and an in-run exact-value cache.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

from ._decoder_config import (
    DecoderConfig,
    resolve_decoder_config,
)
from .workers import preflight_workers
from .bl import BLPriority
from .model import Box, Container, Point
from .pp_state import PPState
from .profile import DecoderProfileCollector, DecoderProfileMetrics


@dataclass(frozen=True, slots=True)
class GAExecutionConfig:
    """Execution-only settings that must not change GA semantics."""

    workers: int = 1
    cache: bool = False

    def __post_init__(self) -> None:
        if type(self.workers) is not int:
            raise TypeError("workers must be a Python int")
        if self.workers < 1:
            raise ValueError("workers must be >= 1")
        preflight_workers(self.workers)
        if type(self.cache) is not bool:
            raise TypeError("cache must be a Python bool")


@dataclass(frozen=True, slots=True)
class EvaluationStats:
    """Non-semantic execution counters for one GA run."""

    requested: int = 0
    computed: int = 0
    cache_hits: int = 0

    def __post_init__(self) -> None:
        for name, value in (
            ("requested", self.requested),
            ("computed", self.computed),
            ("cache_hits", self.cache_hits),
        ):
            if type(value) is not int:
                raise TypeError(f"{name} must be a Python int")
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.computed + self.cache_hits != self.requested:
            raise ValueError("computed + cache_hits must equal requested")


@dataclass(frozen=True, slots=True)
class _ProfiledEvaluation:
    result: Any
    metrics: DecoderProfileMetrics


def _bl_worker(
    container: Container,
    boxes: tuple[Box, ...],
    individual: Any,
    priority: BLPriority,
    decoder_config: DecoderConfig,
    use_center_of_gravity: bool = False,
    center_of_gravity_lambda: float = 0.1,
    profile_enabled: bool = False,
) -> Any:
    # Local import keeps this module independent of ga.py at import time and
    # makes the worker function safe to import under multiprocessing spawn.
    from .ga import evaluate_bl_individual

    if not profile_enabled:
        return evaluate_bl_individual(
            container,
            boxes,
            individual,
            priority=priority,
            decoder_config=decoder_config,
            use_center_of_gravity=use_center_of_gravity,
            center_of_gravity_lambda=center_of_gravity_lambda,
        )
    profile = DecoderProfileCollector()
    result = evaluate_bl_individual(
        container,
        boxes,
        individual,
        priority=priority,
        decoder_config=decoder_config,
        use_center_of_gravity=use_center_of_gravity,
        center_of_gravity_lambda=center_of_gravity_lambda,
        _profile=profile,
    )
    return _ProfiledEvaluation(result, profile.snapshot())


def _pp_worker(
    container: Container,
    boxes: tuple[Box, ...],
    individual: Any,
    state: PPState,
    priority: BLPriority,
    decoder_config: DecoderConfig,
    use_center_of_gravity: bool = False,
    center_of_gravity_lambda: float = 0.1,
    profile_enabled: bool = False,
) -> Any:
    from .pp_ga import evaluate_pp_individual

    if not profile_enabled:
        return evaluate_pp_individual(
            container,
            boxes,
            individual,
            state,
            priority=priority,
            decoder_config=decoder_config,
            use_center_of_gravity=use_center_of_gravity,
            center_of_gravity_lambda=center_of_gravity_lambda,
        )
    profile = DecoderProfileCollector()
    result = evaluate_pp_individual(
        container,
        boxes,
        individual,
        state,
        priority=priority,
        decoder_config=decoder_config,
        use_center_of_gravity=use_center_of_gravity,
        center_of_gravity_lambda=center_of_gravity_lambda,
        _profile=profile,
    )
    return _ProfiledEvaluation(result, profile.snapshot())


def _point_signature(point: Point | None) -> tuple[int, int, int] | None:
    if point is None:
        return None
    return (point.x, point.y, point.z)


def _problem_signature(
    container: Container,
    boxes: tuple[Box, ...],
    priority: BLPriority,
    decoder_config: DecoderConfig,
) -> tuple[object, ...]:
    return (
        "packing3d-evaluation-v2",
        (container.W, container.H, container.D),
        tuple((box.id, box.w, box.h, box.d, box.weight) for box in boxes),
        priority.value,
        decoder_config.candidate_policy,
        decoder_config.candidate_policy_version,
        decoder_config.min_support_percent,
        decoder_config.support_model_version,
        "density-v1",
    )


def _state_signature(state: PPState) -> tuple[tuple[int, tuple[int, int, int] | None], ...]:
    return tuple((box_id, _point_signature(position)) for box_id, position in state.items_tuple())


class EvaluationRuntime:
    """Evaluate populations without changing population order or GA RNG use.

    The cache lives only for one GA run.  It stores successful evaluations by
    exact immutable input values; no hashes are serialized or persisted.
    """

    __slots__ = (
        "config",
        "profile_enabled",
        "_profile_collector",
        "_executor",
        "_cache",
        "_requested",
        "_computed",
        "_cache_hits",
    )

    def __init__(self, config: GAExecutionConfig, *, profile_enabled: bool = False) -> None:
        if type(config) is not GAExecutionConfig:
            raise TypeError("config must be a GAExecutionConfig")
        if type(profile_enabled) is not bool:
            raise TypeError("profile_enabled must be a Python bool")
        self.config = config
        self.profile_enabled = profile_enabled
        self._profile_collector = DecoderProfileCollector() if profile_enabled else None
        self._executor: ProcessPoolExecutor | None = None
        self._cache: dict[tuple[object, ...], Any] = {}
        self._requested = 0
        self._computed = 0
        self._cache_hits = 0

    def __enter__(self) -> EvaluationRuntime:
        if self.config.workers > 1:
            # Explicit spawn mirrors Windows behavior and avoids inheriting GUI
            # or mutable parent-process state into workers.
            self._executor = ProcessPoolExecutor(
                max_workers=self.config.workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None

    @property
    def stats(self) -> EvaluationStats:
        return EvaluationStats(self._requested, self._computed, self._cache_hits)

    @property
    def profile_metrics(self) -> DecoderProfileMetrics | None:
        if self._profile_collector is None:
            return None
        return self._profile_collector.snapshot()

    def evaluate_bl(
        self,
        container: Container,
        boxes: tuple[Box, ...],
        population: tuple[Any, ...],
        *,
        priority: BLPriority,
        generation_index: int,
        decoder_config: DecoderConfig | None = None,
        use_center_of_gravity: bool = False,
        center_of_gravity_lambda: float = 0.1,
        progress_callback: Callable[[int], None] | None = None,
    ) -> tuple[Any, ...]:
        decoder_config = resolve_decoder_config(decoder_config)
        namespace = _problem_signature(container, boxes, priority, decoder_config) + (
            "BL-ref-v2", use_center_of_gravity, center_of_gravity_lambda
        )
        keys = tuple(namespace + (individual.genes,) for individual in population)
        calls = tuple(
            (container, boxes, individual, priority, decoder_config, use_center_of_gravity, center_of_gravity_lambda)
            + ((True,) if self.profile_enabled else ())
            for individual in population
        )
        return self._evaluate_population(
            population,
            keys,
            calls,
            _bl_worker,
            generation_index,
            progress_callback,
        )

    def evaluate_pp(
        self,
        container: Container,
        boxes: tuple[Box, ...],
        population: tuple[tuple[Any, PPState], ...],
        *,
        priority: BLPriority,
        generation_index: int,
        decoder_config: DecoderConfig | None = None,
        use_center_of_gravity: bool = False,
        center_of_gravity_lambda: float = 0.1,
        progress_callback: Callable[[int], None] | None = None,
    ) -> tuple[Any, ...]:
        decoder_config = resolve_decoder_config(decoder_config)
        namespace = _problem_signature(container, boxes, priority, decoder_config) + (
            "PP-ref-v2", use_center_of_gravity, center_of_gravity_lambda
        )
        individuals = tuple(individual for individual, _ in population)
        keys = tuple(
            namespace + (individual.genes, _state_signature(state))
            for individual, state in population
        )
        calls = tuple(
            (container, boxes, individual, state, priority, decoder_config, use_center_of_gravity, center_of_gravity_lambda)
            + ((True,) if self.profile_enabled else ())
            for individual, state in population
        )
        return self._evaluate_population(
            individuals,
            keys,
            calls,
            _pp_worker,
            generation_index,
            progress_callback,
        )

    def _evaluate_population(
        self,
        individuals: tuple[Any, ...],
        keys: tuple[tuple[object, ...], ...],
        calls: tuple[tuple[Any, ...], ...],
        worker: Callable[..., Any],
        generation_index: int,
        progress_callback: Callable[[int], None] | None,
    ) -> tuple[Any, ...]:
        from .ga import EvaluationError

        if len(individuals) != len(keys) or len(keys) != len(calls):
            raise ValueError("population evaluation inputs must have equal lengths")

        count = len(individuals)
        self._requested += count
        results: list[Any | None] = [None] * count

        # A pending entry allows equal inputs in the same parallel batch to
        # share one computation.  It is counted as a cache hit/reuse.
        pending: dict[tuple[object, ...], Future[Any]] = {}
        pending_direct: dict[tuple[object, ...], Any] = {}
        future_by_index: dict[int, Future[Any]] = {}

        for index, (key, call) in enumerate(zip(keys, calls, strict=True)):
            if self.config.cache and key in self._cache:
                results[index] = self._cache[key]
                self._cache_hits += 1
                if self._executor is None and progress_callback is not None:
                    progress_callback(index)
                continue

            if self.config.cache and key in pending_direct:
                results[index] = pending_direct[key]
                self._cache_hits += 1
                if progress_callback is not None:
                    progress_callback(index)
                continue

            if self.config.cache and key in pending:
                future_by_index[index] = pending[key]
                self._cache_hits += 1
                continue

            self._computed += 1
            if self._executor is None:
                try:
                    result = worker(*call)
                except Exception as exc:
                    raise EvaluationError(generation_index, index, exc) from exc
                if self.profile_enabled:
                    result = self._unwrap_profile_result(result)
                self._validate_result(result, individuals[index], generation_index, index)
                results[index] = result
                if self.config.cache:
                    self._cache[key] = result
                    pending_direct[key] = result
                if progress_callback is not None:
                    progress_callback(index)
            else:
                future = self._executor.submit(worker, *call)
                future_by_index[index] = future
                if self.config.cache:
                    pending[key] = future

        if self._executor is not None:
            profiled_futures: set[Future[Any]] = set()
            # Resolve in population order, not completion order.  If several
            # workers fail, this preserves the sequential rule for which
            # population index is reported.
            for index in range(count):
                if results[index] is not None:
                    if progress_callback is not None:
                        progress_callback(index)
                    continue
                future = future_by_index[index]
                try:
                    result = future.result()
                except Exception as exc:
                    for other in set(future_by_index.values()):
                        if other is not future:
                            other.cancel()
                    raise EvaluationError(generation_index, index, exc) from exc
                if self.profile_enabled:
                    if future not in profiled_futures:
                        profiled_futures.add(future)
                        result = self._unwrap_profile_result(result)
                    else:
                        result = self._unwrap_profile_result(result, collect=False)
                self._validate_result(result, individuals[index], generation_index, index)
                results[index] = result
                if self.config.cache:
                    self._cache[keys[index]] = result
                if progress_callback is not None:
                    progress_callback(index)

        # Validation above guarantees every slot was populated.
        return tuple(results)  # type: ignore[return-value]

    def _unwrap_profile_result(self, result: Any, *, collect: bool = True) -> Any:
        if not self.profile_enabled:
            return result
        if type(result) is not _ProfiledEvaluation:
            raise TypeError("profiled worker must return _ProfiledEvaluation")
        if collect:
            assert self._profile_collector is not None
            self._profile_collector.add(result.metrics)
        return result.result

    @staticmethod
    def _validate_result(result: Any, individual: Any, generation_index: int, index: int) -> None:
        from .ga import EvaluationError, EvaluationResult

        if type(result) is not EvaluationResult:
            exc = TypeError("evaluator must return an EvaluationResult")
            raise EvaluationError(generation_index, index, exc) from exc
        if result.individual != individual:
            exc = ValueError("evaluator returned a result for a different individual")
            raise EvaluationError(generation_index, index, exc) from exc
