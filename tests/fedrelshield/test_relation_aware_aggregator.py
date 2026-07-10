from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

from fedrelshield.federation import (
    RelationAwareAggregator,
)


def make_result(
    client_id,
    value,
    num_examples,
):
    return SimpleNamespace(
        client_id=client_id,
        state=OrderedDict({
            "weight": torch.tensor(
                [value],
                dtype=torch.float32,
            ),
        }),
        num_examples=num_examples,
    )


def test_identical_distributions_reduce_to_fedavg():
    aggregator = RelationAwareAggregator(
        relation_distributions={
            "client_a": {
                "r1": 0.5,
                "r2": 0.5,
            },
            "client_b": {
                "r1": 0.5,
                "r2": 0.5,
            },
        },
        divergence_strength=1.0,
    )

    results = (
        make_result(
            "client_a",
            value=1.0,
            num_examples=1,
        ),
        make_result(
            "client_b",
            value=3.0,
            num_examples=3,
        ),
    )

    weights = aggregator.aggregation_weights(
        results
    )

    assert weights["client_a"] == pytest.approx(
        0.25
    )

    assert weights["client_b"] == pytest.approx(
        0.75
    )

    state = aggregator.aggregate(results)

    assert state["weight"].item() == pytest.approx(
        2.5
    )


def test_divergent_client_is_downweighted():
    aggregator = RelationAwareAggregator(
        relation_distributions={
            "client_a": {
                "r1": 0.5,
                "r2": 0.5,
            },
            "client_b": {
                "r1": 1.0,
                "r2": 0.0,
            },
            "client_c": {
                "r1": 0.5,
                "r2": 0.5,
            },
        },
        divergence_strength=5.0,
    )

    results = (
        make_result(
            "client_a",
            value=1.0,
            num_examples=10,
        ),
        make_result(
            "client_b",
            value=2.0,
            num_examples=10,
        ),
        make_result(
            "client_c",
            value=3.0,
            num_examples=10,
        ),
    )

    weights = aggregator.aggregation_weights(
        results
    )

    assert (
        weights["client_b"]
        < weights["client_a"]
    )

    assert (
        weights["client_b"]
        < weights["client_c"]
    )

    assert sum(
        weights.values()
    ) == pytest.approx(1.0)


def test_alpha_zero_matches_fedavg_weights():
    aggregator = RelationAwareAggregator(
        relation_distributions={
            "client_a": {
                "r1": 0.9,
                "r2": 0.1,
            },
            "client_b": {
                "r1": 0.2,
                "r2": 0.8,
            },
        },
        divergence_strength=5.0,
        alpha=0.0,
    )

    results = (
        make_result(
            "client_a",
            25,
            1.0,
        ),
        make_result(
            "client_b",
            75,
            3.0,
        ),
    )

    weights = aggregator.aggregation_weights(
        results
    )

    assert weights["client_a"] == pytest.approx(0.25)
    assert weights["client_b"] == pytest.approx(0.75)


def test_alpha_one_matches_relation_aware_weights():
    aggregator = RelationAwareAggregator(
        relation_distributions={
            "client_a": {
                "r1": 0.9,
                "r2": 0.1,
            },
            "client_b": {
                "r1": 0.2,
                "r2": 0.8,
            },
        },
        divergence_strength=5.0,
        alpha=1.0,
    )

    results = (
        make_result(
            "client_a",
            25,
            1.0,
        ),
        make_result(
            "client_b",
            75,
            3.0,
        ),
    )

    weights = aggregator.aggregation_weights(
        results
    )

    relation_aware_weights = (
        aggregator.relation_aware_weights(
            results
        )
    )

    assert weights == pytest.approx(
        relation_aware_weights
    )


def test_alpha_interpolates_weights():
    relation_distributions = {
        "client_a": {
            "r1": 0.9,
            "r2": 0.1,
        },
        "client_b": {
            "r1": 0.2,
            "r2": 0.8,
        },
    }

    results = (
        make_result(
            "client_a",
            25,
            1.0,
        ),
        make_result(
            "client_b",
            75,
            3.0,
        ),
    )

    fedavg = RelationAwareAggregator(
        relation_distributions=
            relation_distributions,
        divergence_strength=5.0,
        alpha=0.0,
    )

    relation_aware = RelationAwareAggregator(
        relation_distributions=
            relation_distributions,
        divergence_strength=5.0,
        alpha=1.0,
    )

    interpolated = RelationAwareAggregator(
        relation_distributions=
            relation_distributions,
        divergence_strength=5.0,
        alpha=0.5,
    )

    fedavg_weights = fedavg.aggregation_weights(
        results
    )

    relation_aware_weights = (
        relation_aware.aggregation_weights(
            results
        )
    )

    interpolated_weights = (
        interpolated.aggregation_weights(
            results
        )
    )

    for client_id in (
        "client_a",
        "client_b",
    ):
        expected = (
            0.5 * fedavg_weights[client_id]
            + 0.5
            * relation_aware_weights[client_id]
        )

        assert (
            interpolated_weights[client_id]
            == pytest.approx(expected)
        )

    assert sum(
        interpolated_weights.values()
    ) == pytest.approx(1.0)
