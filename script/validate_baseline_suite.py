import argparse
import hashlib
import json
import math
import os
from pathlib import Path


METHOD_ORDER = (
    "zero_shot",
    "local_only",
    "centralized",
    "fedavg",
    "relation_aware",
)


FEDERATED_METHODS = {
    "fedavg",
    "relation_aware",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate the FedRelShield multi-seed "
            "baseline suite artifact."
        )
    )

    parser.add_argument(
        "--suite-dir",
        default=(
            "output/fedrelshield/"
            "baseline_suite"
        ),
    )

    return parser.parse_args()


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def file_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def compute_mean(values):
    if not values:
        raise RuntimeError(
            "Cannot compute mean of empty values"
        )

    return sum(values) / len(values)


def compute_sample_std(values):
    if not values:
        raise RuntimeError(
            "Cannot compute standard deviation "
            "of empty values"
        )

    if len(values) == 1:
        return 0.0

    mean = compute_mean(values)

    variance = sum(
        (value - mean) ** 2
        for value in values
    ) / (len(values) - 1)

    return math.sqrt(variance)


def summarize_values(values):
    return {
        "mean": compute_mean(values),
        "std": compute_sample_std(values),
        "min": min(values),
        "max": max(values),
        "num_runs": len(values),
    }


def values_close(left, right):
    return math.isclose(
        left,
        right,
        rel_tol=1e-12,
        abs_tol=1e-15,
    )


def validate_summary(
    actual,
    expected,
    context,
):
    if set(actual) != {
        "mean",
        "std",
        "min",
        "max",
        "num_runs",
    }:
        raise RuntimeError(
            f"Summary contract mismatch: {context}"
        )

    if actual["num_runs"] != expected["num_runs"]:
        raise RuntimeError(
            f"Summary run count mismatch: {context}"
        )

    for field in (
        "mean",
        "std",
        "min",
        "max",
    ):
        if not values_close(
            actual[field],
            expected[field],
        ):
            raise RuntimeError(
                f"Summary value mismatch: "
                f"{context}.{field}"
            )


def expected_run_artifact_path(
    suite_dir,
    method,
    seed,
):
    run_dir = (
        suite_dir
        / "runs"
        / method
        / f"seed_{seed}"
    )

    if method in FEDERATED_METHODS:
        return run_dir / "round_metrics.json"

    return run_dir / "baseline_metrics.json"


def extract_evaluation(
    method,
    artifact,
):
    if method in FEDERATED_METHODS:
        rounds = artifact["rounds"]

        if not rounds:
            raise RuntimeError(
                f"{method} source artifact "
                "has no rounds"
            )

        return rounds[-1][
            "global_evaluation"
        ]

    return artifact["evaluation"]


def validate_suite_contract(artifact):
    required_keys = {
        "format_version",
        "seeds",
        "clients",
        "evaluation_split",
        "methods",
        "aggregates",
    }

    if set(artifact) != required_keys:
        raise RuntimeError(
            "Suite artifact keys do not match "
            "the required contract"
        )

    if artifact["format_version"] != "1.0":
        raise RuntimeError(
            "Unexpected suite artifact "
            "format version"
        )

    seeds = artifact["seeds"]

    if not seeds:
        raise RuntimeError(
            "Suite artifact has no seeds"
        )

    if len(seeds) != len(set(seeds)):
        raise RuntimeError(
            "Suite artifact contains duplicate seeds"
        )

    clients = artifact["clients"]

    if not clients:
        raise RuntimeError(
            "Suite artifact has no clients"
        )

    if list(artifact["methods"]) != sorted(
        artifact["methods"]
    ):
        # JSON is serialized with sort_keys=True.
        pass

    if set(artifact["methods"]) != set(METHOD_ORDER):
        raise RuntimeError(
            "Suite methods do not match contract"
        )

    if set(artifact["aggregates"]) != set(
        METHOD_ORDER
    ):
        raise RuntimeError(
            "Suite aggregates do not match contract"
        )


def validate_source_artifact(
    method,
    artifact,
    seed,
    expected_clients,
    expected_split,
):
    if artifact["seed"] != seed:
        raise RuntimeError(
            f"Source seed mismatch: {method}/{seed}"
        )

    if artifact["clients"] != expected_clients:
        raise RuntimeError(
            "Source client order mismatch: "
            f"{method}/{seed}"
        )

    if (
        artifact["evaluation_split"]
        != expected_split
    ):
        raise RuntimeError(
            "Source evaluation split mismatch: "
            f"{method}/{seed}"
        )

    if method in FEDERATED_METHODS:
        if artifact["format_version"] != "1.0":
            raise RuntimeError(
                f"Unexpected {method} metrics version"
            )
    
        if not artifact["rounds"]:
            raise RuntimeError(
                f"{method} source artifact "
                "contains no rounds"
            )
    else:
        if artifact["format_version"] != "1.0":
            raise RuntimeError(
                f"Unexpected {method} version"
            )

        if artifact["baseline"] != method:
            raise RuntimeError(
                f"Source baseline mismatch: {method}"
            )


def validate_stored_run(
    stored_run,
    source_evaluation,
    method,
    seed,
):
    required_keys = {
        "macro_metrics",
        "weighted_metrics",
        "enterprises",
    }

    if set(stored_run) != required_keys:
        raise RuntimeError(
            "Stored run contract mismatch: "
            f"{method}/{seed}"
        )

    if stored_run != {
        "macro_metrics": (
            source_evaluation["macro_metrics"]
        ),
        "weighted_metrics": (
            source_evaluation["weighted_metrics"]
        ),
        "enterprises": (
            source_evaluation["enterprises"]
        ),
    }:
        raise RuntimeError(
            "Stored suite run differs from "
            f"source artifact: {method}/{seed}"
        )


def validate_cross_method_comparability(
    source_evaluations,
    seeds,
    clients,
):
    reference_method = METHOD_ORDER[0]

    for seed in seeds:
        reference = source_evaluations[
            reference_method
        ][seed]

        reference_enterprises = (
            reference["enterprises"]
        )

        reference_macro_names = set(
            reference["macro_metrics"]
        )

        reference_weighted_names = set(
            reference["weighted_metrics"]
        )

        for method in METHOD_ORDER[1:]:
            evaluation = source_evaluations[
                method
            ][seed]

            if set(
                evaluation["enterprises"]
            ) != set(clients):
                raise RuntimeError(
                    "Cross-method enterprise "
                    f"coverage mismatch: {method}/{seed}"
                )

            if set(
                evaluation["macro_metrics"]
            ) != reference_macro_names:
                raise RuntimeError(
                    "Cross-method macro metric "
                    f"mismatch: {method}/{seed}"
                )

            if set(
                evaluation["weighted_metrics"]
            ) != reference_weighted_names:
                raise RuntimeError(
                    "Cross-method weighted metric "
                    f"mismatch: {method}/{seed}"
                )

            for enterprise_id in clients:
                reference_result = (
                    reference_enterprises[
                        enterprise_id
                    ]
                )

                result = evaluation[
                    "enterprises"
                ][enterprise_id]

                if (
                    result["num_examples"]
                    != reference_result[
                        "num_examples"
                    ]
                ):
                    raise RuntimeError(
                        "Cross-method example count "
                        "mismatch: "
                        f"{method}/{seed}/"
                        f"{enterprise_id}"
                    )

                if set(
                    result["metrics"]
                ) != set(
                    reference_result["metrics"]
                ):
                    raise RuntimeError(
                        "Cross-method enterprise "
                        "metric mismatch: "
                        f"{method}/{seed}/"
                        f"{enterprise_id}"
                    )


def recompute_aggregates(
    source_evaluations,
    seeds,
    clients,
):
    aggregates = {}

    for method in METHOD_ORDER:
        evaluations = source_evaluations[
            method
        ]

        first = evaluations[seeds[0]]

        macro_metrics = {}

        for metric_name in first[
            "macro_metrics"
        ]:
            values = [
                evaluations[seed][
                    "macro_metrics"
                ][metric_name]
                for seed in seeds
            ]

            macro_metrics[
                metric_name
            ] = summarize_values(values)

        weighted_metrics = {}

        for metric_name in first[
            "weighted_metrics"
        ]:
            values = [
                evaluations[seed][
                    "weighted_metrics"
                ][metric_name]
                for seed in seeds
            ]

            weighted_metrics[
                metric_name
            ] = summarize_values(values)

        enterprises = {}

        for enterprise_id in clients:
            first_result = first[
                "enterprises"
            ][enterprise_id]

            metrics = {}

            for metric_name in first_result[
                "metrics"
            ]:
                values = [
                    evaluations[seed][
                        "enterprises"
                    ][enterprise_id][
                        "metrics"
                    ][metric_name]
                    for seed in seeds
                ]

                metrics[
                    metric_name
                ] = summarize_values(values)

            num_examples = (
                first_result["num_examples"]
            )

            for seed in seeds:
                if (
                    evaluations[seed][
                        "enterprises"
                    ][enterprise_id][
                        "num_examples"
                    ]
                    != num_examples
                ):
                    raise RuntimeError(
                        "Example count changed across "
                        f"seeds: {method}/"
                        f"{enterprise_id}"
                    )

            enterprises[
                enterprise_id
            ] = {
                "num_examples": num_examples,
                "metrics": metrics,
            }

        aggregates[method] = {
            "macro_metrics": macro_metrics,
            "weighted_metrics": weighted_metrics,
            "enterprises": enterprises,
        }

    return aggregates


def validate_aggregates(
    actual,
    expected,
    clients,
):
    for method in METHOD_ORDER:
        for aggregate_name in (
            "macro_metrics",
            "weighted_metrics",
        ):
            if set(
                actual[method][aggregate_name]
            ) != set(
                expected[method][aggregate_name]
            ):
                raise RuntimeError(
                    "Aggregate metric coverage "
                    f"mismatch: {method}/"
                    f"{aggregate_name}"
                )

            for metric_name in expected[
                method
            ][aggregate_name]:
                validate_summary(
                    actual[method][
                        aggregate_name
                    ][metric_name],
                    expected[method][
                        aggregate_name
                    ][metric_name],
                    (
                        f"{method}."
                        f"{aggregate_name}."
                        f"{metric_name}"
                    ),
                )

        if set(
            actual[method]["enterprises"]
        ) != set(clients):
            raise RuntimeError(
                "Aggregate enterprise coverage "
                f"mismatch: {method}"
            )

        for enterprise_id in clients:
            actual_result = actual[
                method
            ]["enterprises"][enterprise_id]

            expected_result = expected[
                method
            ]["enterprises"][enterprise_id]

            if (
                actual_result["num_examples"]
                != expected_result["num_examples"]
            ):
                raise RuntimeError(
                    "Aggregate example count mismatch: "
                    f"{method}/{enterprise_id}"
                )

            if set(
                actual_result["metrics"]
            ) != set(
                expected_result["metrics"]
            ):
                raise RuntimeError(
                    "Aggregate enterprise metric "
                    "coverage mismatch: "
                    f"{method}/{enterprise_id}"
                )

            for metric_name in expected_result[
                "metrics"
            ]:
                validate_summary(
                    actual_result[
                        "metrics"
                    ][metric_name],
                    expected_result[
                        "metrics"
                    ][metric_name],
                    (
                        f"{method}."
                        f"{enterprise_id}."
                        f"{metric_name}"
                    ),
                )


def main():
    args = parse_args()

    suite_dir = Path(
        os.path.expanduser(args.suite_dir)
    )

    suite_path = (
        suite_dir
        / "baseline_suite.json"
    )

    if not suite_path.is_file():
        raise RuntimeError(
            "Baseline suite artifact does not exist"
        )

    artifact = load_json(suite_path)

    validate_suite_contract(artifact)

    seeds = artifact["seeds"]
    clients = artifact["clients"]
    evaluation_split = artifact[
        "evaluation_split"
    ]

    source_evaluations = {
        method: {}
        for method in METHOD_ORDER
    }

    print()
    print("Validating raw per-seed artifacts")

    for method in METHOD_ORDER:
        for seed in seeds:
            source_path = (
                expected_run_artifact_path(
                    suite_dir,
                    method,
                    seed,
                )
            )

            if not source_path.is_file():
                raise RuntimeError(
                    "Missing source artifact: "
                    f"{source_path}"
                )

            source_artifact = load_json(
                source_path
            )

            validate_source_artifact(
                method=method,
                artifact=source_artifact,
                seed=seed,
                expected_clients=clients,
                expected_split=(
                    evaluation_split
                ),
            )

            evaluation = extract_evaluation(
                method,
                source_artifact,
            )

            source_evaluations[
                method
            ][seed] = evaluation

            stored_run = artifact[
                "methods"
            ][method][str(seed)]

            validate_stored_run(
                stored_run=stored_run,
                source_evaluation=evaluation,
                method=method,
                seed=seed,
            )

    validate_cross_method_comparability(
        source_evaluations=source_evaluations,
        seeds=seeds,
        clients=clients,
    )

    recomputed_aggregates = (
        recompute_aggregates(
            source_evaluations=(
                source_evaluations
            ),
            seeds=seeds,
            clients=clients,
        )
    )

    validate_aggregates(
        actual=artifact["aggregates"],
        expected=recomputed_aggregates,
        clients=clients,
    )

    before_hash = file_sha256(
        suite_path
    )

    temporary_path = (
        suite_dir
        / "baseline_suite.regenerated.json"
    )

    regenerated_artifact = {
        "format_version": (
            artifact["format_version"]
        ),
        "seeds": seeds,
        "clients": clients,
        "evaluation_split": evaluation_split,
        "methods": artifact["methods"],
        "aggregates": recomputed_aggregates,
    }

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            regenerated_artifact,
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")

    after_hash = file_sha256(
        temporary_path
    )

    temporary_path.unlink()

    if before_hash != after_hash:
        raise RuntimeError(
            "Suite artifact is not byte-identical "
            "to independent regeneration"
        )

    print()
    print("Suite validation diagnostics")
    print("----------------------------")
    print(
        "Source artifacts checked: "
        f"{len(seeds) * len(METHOD_ORDER)}"
    )
    print(f"Suite SHA-256: {before_hash}")
    print("Suite artifact contract: PASS")
    print("Source artifact contracts: PASS")
    print("Stored raw runs: IDENTICAL TO SOURCES")
    print("Cross-method comparability: PASS")
    print("Cross-seed example counts: PASS")
    print("Sample standard deviations: VERIFIED")
    print("Aggregate statistics: VERIFIED")
    print("Independent regeneration: BYTE-IDENTICAL")
    print()
    print("Baseline suite validation: PASS")


if __name__ == "__main__":
    main()

