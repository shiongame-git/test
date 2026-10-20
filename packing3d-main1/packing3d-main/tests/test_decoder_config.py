"""Validation tests for the internal v1.2 decoder settings value."""

from dataclasses import FrozenInstanceError

import pytest

from packing import DecoderConfig


class IntSubclass(int):
    pass


class StrSubclass(str):
    pass


def test_v12_defaults_are_canonical_and_immutable():
    settings = DecoderConfig()

    assert settings.candidate_policy == "Cartesian"
    assert settings.candidate_policy_version == 1
    assert settings.min_support_percent == 55
    assert settings.support_model_version == "support-union-v1"
    with pytest.raises(FrozenInstanceError):
        settings.min_support_percent = 0


@pytest.mark.parametrize("candidate_policy", ["Cartesian", "LocalNeighbors"])
@pytest.mark.parametrize("support_percent", [0, 55, 100])
def test_accepts_known_policies_and_support_percentages(
    candidate_policy, support_percent
):
    settings = DecoderConfig(
        candidate_policy=candidate_policy,
        min_support_percent=support_percent,
    )

    assert settings.candidate_policy == candidate_policy
    assert settings.min_support_percent == support_percent


@pytest.mark.parametrize("candidate_policy", [None, 1, StrSubclass("Cartesian"), []])
def test_candidate_policy_requires_a_canonical_string(candidate_policy):
    with pytest.raises(TypeError, match="candidate_policy"):
        DecoderConfig(candidate_policy=candidate_policy)


@pytest.mark.parametrize("candidate_policy", ["cartesian", "LocalNeighbors ", ""])
def test_candidate_policy_rejects_unknown_names(candidate_policy):
    with pytest.raises(ValueError, match="unknown candidate_policy"):
        DecoderConfig(candidate_policy=candidate_policy)


@pytest.mark.parametrize(
    ("version", "error"),
    [
        (True, TypeError),
        (1.0, TypeError),
        ("1", TypeError),
        (IntSubclass(1), TypeError),
        (0, ValueError),
        (2, ValueError),
    ],
)
def test_candidate_policy_version_has_a_strict_boundary(version, error):
    with pytest.raises(error, match="candidate_policy_version"):
        DecoderConfig(candidate_policy_version=version)


@pytest.mark.parametrize("support_percent", [True, False, 55.0, "55", IntSubclass(55)])
def test_support_percent_requires_a_builtin_int(support_percent):
    with pytest.raises(TypeError, match="min_support_percent"):
        DecoderConfig(min_support_percent=support_percent)


@pytest.mark.parametrize("support_percent", [-1, 101])
def test_support_percent_is_inclusive_zero_to_one_hundred(support_percent):
    with pytest.raises(ValueError, match="between 0 and 100"):
        DecoderConfig(min_support_percent=support_percent)


@pytest.mark.parametrize("support_model_version", [None, 1, StrSubclass("support-union-v1")])
def test_support_model_version_requires_a_canonical_string(support_model_version):
    with pytest.raises(TypeError, match="support_model_version"):
        DecoderConfig(support_model_version=support_model_version)


@pytest.mark.parametrize("support_model_version", ["support-union-v2", "Support-Union-v1", ""])
def test_support_model_version_rejects_unknown_names(support_model_version):
    with pytest.raises(ValueError, match="unknown support_model_version"):
        DecoderConfig(support_model_version=support_model_version)
