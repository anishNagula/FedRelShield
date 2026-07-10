from dataclasses import replace
from typing import Dict, Mapping

from fedrelshield.data.profiles import (
    BENIGN_RELATIONS,
    EnterpriseProfile,
)


HETEROGENEITY_LEVELS = {
    "h0": 0.0,
    "h1": 0.25,
    "h2": 0.50,
    "h3": 0.75,
    "h4": 1.0,
}


CLIENT_PREFERENCE_WEIGHTS = {
    "enterprise_a": {
        "authenticates_to": 0.55,
        "connects_to": 0.15,
        "spawns": 0.10,
        "accesses": 0.10,
        "invokes": 0.10,
    },
    "enterprise_b": {
        "authenticates_to": 0.10,
        "connects_to": 0.55,
        "spawns": 0.15,
        "accesses": 0.10,
        "invokes": 0.10,
    },
    "enterprise_c": {
        "authenticates_to": 0.10,
        "connects_to": 0.10,
        "spawns": 0.55,
        "accesses": 0.15,
        "invokes": 0.10,
    },
    "enterprise_d": {
        "authenticates_to": 0.10,
        "connects_to": 0.10,
        "spawns": 0.10,
        "accesses": 0.55,
        "invokes": 0.15,
    },
    "enterprise_e": {
        "authenticates_to": 0.15,
        "connects_to": 0.10,
        "spawns": 0.10,
        "accesses": 0.10,
        "invokes": 0.55,
    },
}


def _normalize(
    weights: Mapping[str, float],
) -> Dict[str, float]:
    relation_set = set(BENIGN_RELATIONS)

    if set(weights) != relation_set:
        raise ValueError(
            "Relation weight keys must exactly match "
            "BENIGN_RELATIONS"
        )

    if any(
        weight < 0.0
        for weight in weights.values()
    ):
        raise ValueError(
            "Relation weights must be non-negative"
        )

    total = sum(weights.values())

    if total <= 0.0:
        raise ValueError(
            "Relation weights must have positive total mass"
        )

    return {
        relation: (
            float(weights[relation])
            / total
        )
        for relation in BENIGN_RELATIONS
    }


def build_common_base_distribution(
    profiles: Mapping[
        str,
        EnterpriseProfile,
    ],
) -> Dict[str, float]:
    if not profiles:
        raise ValueError(
            "profiles must be non-empty"
        )

    accumulated = {
        relation: 0.0
        for relation in BENIGN_RELATIONS
    }

    for profile in profiles.values():
        normalized = _normalize(
            profile.benign_relation_weights
        )

        for relation in BENIGN_RELATIONS:
            accumulated[relation] += (
                normalized[relation]
            )

    count = len(profiles)

    return _normalize(
        {
            relation: (
                accumulated[relation]
                / count
            )
            for relation in BENIGN_RELATIONS
        }
    )


def build_skewed_distribution(
    base_weights: Mapping[str, float],
    client_id: str,
    alpha: float,
) -> Dict[str, float]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(
            "alpha must be in [0, 1]"
        )

    if client_id not in CLIENT_PREFERENCE_WEIGHTS:
        raise ValueError(
            f"Unknown client_id: {client_id}"
        )

    base = _normalize(base_weights)

    preference = _normalize(
        CLIENT_PREFERENCE_WEIGHTS[
            client_id
        ]
    )

    mixed = {
        relation: (
            (1.0 - alpha)
            * base[relation]
            + alpha
            * preference[relation]
        )
        for relation in BENIGN_RELATIONS
    }

    return _normalize(mixed)


def build_heterogeneous_profile(
    profile: EnterpriseProfile,
    base_weights: Mapping[str, float],
    level: str,
) -> EnterpriseProfile:
    if level not in HETEROGENEITY_LEVELS:
        raise ValueError(
            f"Unknown heterogeneity level: {level}"
        )

    alpha = HETEROGENEITY_LEVELS[level]

    skewed_weights = build_skewed_distribution(
        base_weights=base_weights,
        client_id=profile.enterprise_id,
        alpha=alpha,
    )

    heterogeneous_profile = replace(
        profile,
        benign_relation_weights=skewed_weights,
    )

    heterogeneous_profile.validate()

    return heterogeneous_profile
