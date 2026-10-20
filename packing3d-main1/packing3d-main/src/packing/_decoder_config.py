"""Canonical immutable decoder configuration and defaults."""

from __future__ import annotations

from dataclasses import dataclass


_CANDIDATE_POLICIES = frozenset({"Cartesian", "LocalNeighbors"})
_SUPPORT_MODEL_VERSIONS = frozenset({"support-union-v1"})


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    """Validated semantic inputs for one decoder evaluation or run."""

    candidate_policy: str = "Cartesian"
    candidate_policy_version: int = 1
    min_support_percent: int = 55
    support_model_version: str = "support-union-v1"

    def __post_init__(self) -> None:
        if type(self.candidate_policy) is not str:
            raise TypeError("candidate_policy must be a canonical string")
        if self.candidate_policy not in _CANDIDATE_POLICIES:
            raise ValueError(f"unknown candidate_policy: {self.candidate_policy!r}")

        if type(self.candidate_policy_version) is not int:
            raise TypeError("candidate_policy_version must be a Python int")
        if self.candidate_policy_version != 1:
            raise ValueError(
                f"unsupported candidate_policy_version: {self.candidate_policy_version}"
            )

        if type(self.min_support_percent) is not int:
            raise TypeError("min_support_percent must be a Python int")
        if not 0 <= self.min_support_percent <= 100:
            raise ValueError("min_support_percent must be between 0 and 100")

        if type(self.support_model_version) is not str:
            raise TypeError("support_model_version must be a canonical string")
        if self.support_model_version not in _SUPPORT_MODEL_VERSIONS:
            raise ValueError(
                f"unknown support_model_version: {self.support_model_version!r}"
            )


# These constants distinguish normal new runs from explicit historical replay.
DEFAULT_DECODER_CONFIG = DecoderConfig()
LEGACY_DECODER_CONFIG = DecoderConfig(min_support_percent=0)


def resolve_decoder_config(settings: DecoderConfig | None) -> DecoderConfig:
    """Resolve omitted settings to the standard Cartesian/55% contract."""
    if settings is None:
        return DEFAULT_DECODER_CONFIG
    if type(settings) is not DecoderConfig:
        raise TypeError("decoder_config must be a DecoderConfig or None")
    return settings
