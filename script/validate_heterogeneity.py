import argparse
import json
import math
from itertools import combinations
from pathlib import Path


DEFAULT_ENTERPRISE_IDS = (
    "enterprise_a",
    "enterprise_b",
    "enterprise_c",
    "enterprise_d",
    "enterprise_e",
)


DISTRIBUTION_FIELDS = (
    "entity_type_distribution",
    "relation_distribution",
    "degree_distribution",
)


JSD_FIELDS = (
    "relation_distribution_jsd",
    "entity_type_distribution_jsd",
    "degree_distribution_jsd",
)


NON_NEGATIVE_FIELDS = (
    "graph_density_difference",
    "attack_prevalence_difference",
    "attack_occurrence_rate_difference",
    "aggregate_heterogeneity_score",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate a FedRelShield "
            "heterogeneity analysis report."
        )
    )

    parser.add_argument(
        "--report",
        default=(
            "data/fedrelshield/"
            "heterogeneity_report.json"
        ),
    )

    parser.add_argument(
        "--enterprise-ids",
        nargs="+",
        default=list(DEFAULT_ENTERPRISE_IDS),
    )

    return parser.parse_args()


def assert_finite(
    value,
    field_name,
):
    if not isinstance(value, (int, float)):
        raise ValueError(
            f"{field_name} must be numeric"
        )

    if not math.isfinite(value):
        raise ValueError(
            f"{field_name} must be finite"
        )


def validate_distribution(
    distribution,
    field_name,
):
    if not isinstance(distribution, dict):
        raise ValueError(
            f"{field_name} must be a dictionary"
        )

    if not distribution:
        raise ValueError(
            f"{field_name} must be non-empty"
        )

    for key, value in distribution.items():
        assert_finite(
            value,
            f"{field_name}.{key}",
        )

        if value < 0.0:
            raise ValueError(
                f"{field_name} contains negative probability"
            )

    total = sum(distribution.values())

    if not math.isclose(
        total,
        1.0,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"{field_name} does not sum to one: {total}"
        )


def main():
    args = parse_args()

    report_path = Path(args.report)

    with open(
        report_path,
        "r",
        encoding="utf-8",
    ) as file:
        report = json.load(file)

    expected_enterprise_ids = tuple(
        sorted(args.enterprise_ids)
    )

    actual_enterprise_ids = tuple(
        report["enterprise_ids"]
    )

    if actual_enterprise_ids != expected_enterprise_ids:
        raise ValueError(
            "Enterprise ID set mismatch"
        )

    enterprise_metrics = report[
        "enterprise_metrics"
    ]

    if set(enterprise_metrics) != set(
        expected_enterprise_ids
    ):
        raise ValueError(
            "Enterprise metrics set mismatch"
        )

    for enterprise_id, metrics in (
        enterprise_metrics.items()
    ):
        if metrics["enterprise_id"] != enterprise_id:
            raise ValueError(
                "Enterprise metric ID mismatch"
            )

        if metrics["num_entities"] <= 0:
            raise ValueError(
                f"{enterprise_id} has no entities"
            )

        if metrics["num_triples"] <= 0:
            raise ValueError(
                f"{enterprise_id} has no triples"
            )

        if metrics["num_relations"] <= 0:
            raise ValueError(
                f"{enterprise_id} has no relations"
            )

        assert_finite(
            metrics["graph_density"],
            f"{enterprise_id}.graph_density",
        )

        if not 0.0 <= metrics["graph_density"] <= 1.0:
            raise ValueError(
                f"{enterprise_id} graph density out of range"
            )

        assert_finite(
            metrics["attack_triple_prevalence"],
            (
                f"{enterprise_id}."
                "attack_triple_prevalence"
            ),
        )

        if not (
            0.0
            <= metrics["attack_triple_prevalence"]
            <= 1.0
        ):
            raise ValueError(
                f"{enterprise_id} attack prevalence out of range"
            )

        for field_name in DISTRIBUTION_FIELDS:
            validate_distribution(
                metrics[field_name],
                f"{enterprise_id}.{field_name}",
            )

    expected_pairs = {
        f"{enterprise_a}__{enterprise_b}"
        for enterprise_a, enterprise_b
        in combinations(
            expected_enterprise_ids,
            2,
        )
    }

    pairwise_metrics = report[
        "pairwise_metrics"
    ]

    if set(pairwise_metrics) != expected_pairs:
        raise ValueError(
            "Pairwise metric set mismatch"
        )

    for pair_key, metrics in (
        pairwise_metrics.items()
    ):
        for field_name in JSD_FIELDS:
            value = metrics[field_name]

            assert_finite(
                value,
                f"{pair_key}.{field_name}",
            )

            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{pair_key}.{field_name} "
                    "must be in [0, 1]"
                )

        for field_name in NON_NEGATIVE_FIELDS:
            value = metrics[field_name]

            assert_finite(
                value,
                f"{pair_key}.{field_name}",
            )

            if value < 0.0:
                raise ValueError(
                    f"{pair_key}.{field_name} "
                    "must be non-negative"
                )

    summary = report["summary"]

    if summary["num_enterprises"] != len(
        expected_enterprise_ids
    ):
        raise ValueError(
            "Summary enterprise count mismatch"
        )

    if summary["num_pairs"] != len(
        expected_pairs
    ):
        raise ValueError(
            "Summary pair count mismatch"
        )

    if (
        summary["most_heterogeneous_pair"]
        not in expected_pairs
    ):
        raise ValueError(
            "Invalid most heterogeneous pair"
        )

    if (
        summary["least_heterogeneous_pair"]
        not in expected_pairs
    ):
        raise ValueError(
            "Invalid least heterogeneous pair"
        )

    print("Report structure: PASS")
    print("Enterprise metrics: PASS")
    print("Distribution normalization: PASS")
    print("Pairwise coverage: PASS")
    print("JSD bounds: PASS")
    print("Finite metric validation: PASS")
    print()
    print("Heterogeneity validation: PASS")


if __name__ == "__main__":
    main()
