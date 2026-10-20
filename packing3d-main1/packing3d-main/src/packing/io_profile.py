"""Strict JSON v1 reader and atomic writer for separate decoder profiles."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ._decoder_config import DecoderConfig
from .errors import InputFormatError
from .io_run_metadata import atomic_write
from .profile import DecoderProfileMetrics, DecoderProfileRecord

_ROOT_KEYS = ("format", "version", "run_id", "decoder", "decoder_config", "evaluations", "metrics")
_DECODER_CONFIG_KEYS = (
    "candidate_policy",
    "candidate_policy_version",
    "min_support_percent",
    "support_model_version",
)
_EVALUATION_KEYS = ("requested", "computed", "cache_hits")
_METRIC_KEYS = tuple(DecoderProfileMetrics().to_dict())
_COUNT_KEYS = (
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


class _DuplicateKey(ValueError):
    pass


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey(f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-standard numeric constant {token!r} is not allowed")


def _require_object(value: Any, keys: tuple[str, ...], path: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError(f"{path} must be a JSON object")
    missing = set(keys) - set(value)
    extra = set(value) - set(keys)
    if missing:
        raise ValueError(f"{path} is missing key(s): {', '.join(sorted(missing))}")
    if extra:
        raise ValueError(f"{path} has unknown key(s): {', '.join(sorted(extra))}")
    return value


def _require_int(value: Any, path: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise ValueError(f"{path} must be a JSON integer")
    if value < minimum:
        raise ValueError(f"{path} must be >= {minimum}")
    return value


def _path(path: str | os.PathLike[str]) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError("path must be str or os.PathLike")
    return Path(path)


def read_decoder_profile_json(path: str | os.PathLike[str]) -> DecoderProfileRecord:
    """Read a strict standalone profile artifact without changing run metadata."""
    filename = _path(path)
    try:
        text = filename.read_bytes().decode("utf-8-sig")
        data = json.loads(
            text,
            object_pairs_hook=_pairs_no_duplicates,
            parse_constant=_reject_constant,
        )
        root = _require_object(data, _ROOT_KEYS, "$")
        if root["format"] != "packing3d-decoder-profile":
            raise ValueError("$.format must be exactly 'packing3d-decoder-profile'")
        if _require_int(root["version"], "$.version") != 1:
            raise ValueError("$.version must be 1")
        if type(root["run_id"]) is not str or not root["run_id"]:
            raise ValueError("$.run_id must be a non-empty string")
        if type(root["decoder"]) is not str or root["decoder"] not in ("BL", "PP"):
            raise ValueError("$.decoder must be 'BL' or 'PP'")

        raw_config = _require_object(root["decoder_config"], _DECODER_CONFIG_KEYS, "$.decoder_config")
        config = DecoderConfig(
            candidate_policy=raw_config["candidate_policy"],
            candidate_policy_version=_require_int(
                raw_config["candidate_policy_version"], "$.decoder_config.candidate_policy_version", minimum=1
            ),
            min_support_percent=_require_int(
                raw_config["min_support_percent"], "$.decoder_config.min_support_percent"
            ),
            support_model_version=raw_config["support_model_version"],
        )

        evaluations = _require_object(root["evaluations"], _EVALUATION_KEYS, "$.evaluations")
        metrics_data = _require_object(root["metrics"], _METRIC_KEYS, "$.metrics")
        metrics_values: dict[str, int | float] = {}
        for key in _METRIC_KEYS:
            value = metrics_data[key]
            if key in _COUNT_KEYS:
                metrics_values[key] = _require_int(value, f"$.metrics.{key}")
            else:
                if type(value) not in (int, float):
                    raise ValueError(f"$.metrics.{key} must be a JSON number")
                metrics_values[key] = float(value)

        return DecoderProfileRecord(
            run_id=root["run_id"],
            decoder=root["decoder"],
            decoder_config=config,
            requested_evaluations=_require_int(evaluations["requested"], "$.evaluations.requested"),
            computed_evaluations=_require_int(evaluations["computed"], "$.evaluations.computed"),
            cache_hits=_require_int(evaluations["cache_hits"], "$.evaluations.cache_hits"),
            metrics=DecoderProfileMetrics(**metrics_values),
        )
    except (UnicodeError, json.JSONDecodeError, _DuplicateKey, TypeError, ValueError) as exc:
        if isinstance(exc, InputFormatError):
            raise
        raise InputFormatError(f"{filename}: {exc}") from exc


def write_decoder_profile_json(
    path: str | os.PathLike[str], profile: DecoderProfileRecord
) -> None:
    """Atomically write a standalone decoder profile JSON v1 record."""
    if type(profile) is not DecoderProfileRecord:
        raise TypeError("profile must be a DecoderProfileRecord")
    atomic_write(_path(path), profile.to_dict())
