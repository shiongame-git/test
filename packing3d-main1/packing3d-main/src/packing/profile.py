"""Optional decoder workload metrics and their independent run-linked record."""

from __future__ import annotations

from dataclasses import dataclass, fields
from math import isfinite

from ._decoder_config import DecoderConfig

_COUNTER_FIELDS = (
    "raw_candidates",
    "bounds_eligible_candidates",
    "visited_candidates",
    "aabb_checks",
    "broad_phase_collision_candidates",
    "support_searches",
    "support_intersection_rectangles",
    "pp_inherited_accepted",
    "pp_deferred_overlap",
    "pp_deferred_support",
    "pp_repack_successes",
    "pp_repack_failures",
)
_TIME_FIELDS = (
    "support_union_seconds",
    "candidate_construction_seconds",
    "common_legality_seconds",
)
_METRIC_FIELDS = _COUNTER_FIELDS + _TIME_FIELDS


@dataclass(frozen=True, slots=True)
class DecoderProfileMetrics:
    """Aggregated decoder counters and elapsed seconds for computed evaluations."""

    raw_candidates: int = 0
    bounds_eligible_candidates: int = 0
    visited_candidates: int = 0
    aabb_checks: int = 0
    broad_phase_collision_candidates: int = 0
    support_searches: int = 0
    support_intersection_rectangles: int = 0
    pp_inherited_accepted: int = 0
    pp_deferred_overlap: int = 0
    pp_deferred_support: int = 0
    pp_repack_successes: int = 0
    pp_repack_failures: int = 0
    support_union_seconds: float = 0.0
    candidate_construction_seconds: float = 0.0
    common_legality_seconds: float = 0.0

    def __post_init__(self) -> None:
        for name in _COUNTER_FIELDS:
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be a Python int")
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in _TIME_FIELDS:
            value = getattr(self, name)
            if type(value) is not float:
                raise TypeError(f"{name} must be a Python float")
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0.0")

    def to_dict(self) -> dict[str, int | float]:
        return {item.name: getattr(self, item.name) for item in fields(self)}


class DecoderProfileCollector:
    """Mutable, evaluation-local collector; never shared across worker tasks."""

    __slots__ = _METRIC_FIELDS

    def __init__(self) -> None:
        for name in _COUNTER_FIELDS:
            setattr(self, name, 0)
        for name in _TIME_FIELDS:
            setattr(self, name, 0.0)

    def add(self, metrics: DecoderProfileMetrics) -> None:
        if type(metrics) is not DecoderProfileMetrics:
            raise TypeError("metrics must be DecoderProfileMetrics")
        for name in _METRIC_FIELDS:
            setattr(self, name, getattr(self, name) + getattr(metrics, name))

    def snapshot(self) -> DecoderProfileMetrics:
        return DecoderProfileMetrics(**{name: getattr(self, name) for name in _METRIC_FIELDS})


@dataclass(frozen=True, slots=True)
class DecoderProfileRecord:
    """A schema-versioned profile artifact linked to one GA run by run ID."""

    run_id: str
    decoder: str
    decoder_config: DecoderConfig
    requested_evaluations: int
    computed_evaluations: int
    cache_hits: int
    metrics: DecoderProfileMetrics

    def __post_init__(self) -> None:
        if type(self.run_id) is not str or not self.run_id:
            raise TypeError("run_id must be a non-empty string")
        if type(self.decoder) is not str or self.decoder not in ("BL", "PP"):
            raise ValueError("decoder must be 'BL' or 'PP'")
        if type(self.decoder_config) is not DecoderConfig:
            raise TypeError("decoder_config must be a DecoderConfig")
        for name in ("requested_evaluations", "computed_evaluations", "cache_hits"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError(f"{name} must be a Python int")
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.computed_evaluations + self.cache_hits != self.requested_evaluations:
            raise ValueError("computed_evaluations + cache_hits must equal requested_evaluations")
        if type(self.metrics) is not DecoderProfileMetrics:
            raise TypeError("metrics must be DecoderProfileMetrics")

    def to_dict(self) -> dict[str, object]:
        config = self.decoder_config
        return {
            "format": "packing3d-decoder-profile",
            "version": 1,
            "run_id": self.run_id,
            "decoder": self.decoder,
            "decoder_config": {
                "candidate_policy": config.candidate_policy,
                "candidate_policy_version": config.candidate_policy_version,
                "min_support_percent": config.min_support_percent,
                "support_model_version": config.support_model_version,
            },
            "evaluations": {
                "requested": self.requested_evaluations,
                "computed": self.computed_evaluations,
                "cache_hits": self.cache_hits,
            },
            "metrics": self.metrics.to_dict(),
        }
