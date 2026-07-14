import argparse
import hashlib
import statistics
import json
import math
from pathlib import Path


DEFAULT_SUITE_DIR = (
    "output/fedrelshield/"
    "relation_aware_ablation"
)

BASELINE_SUITE_DIR = (
    Path("output/fedrelshield")
    / "baseline_suite"
)

METHOD = "relation_aware"

METRIC_NAMES = (
    "hits@1",
    "hits@10",
    "hits@3",
    "mrr",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate the FedRelShield "
            "relation-aware alpha ablation suite."
        )
    )

    parser.add_argument(
        "--suite-dir",
        default=DEFAULT_SUITE_DIR,
        help=(
            "relation-aware ablation suite directory"
        ),
    )

    parser.add_argument(
        "--baseline-suite-dir",
        default=str(BASELINE_SUITE_DIR),
        help=(
            "validated baseline suite directory "
            "used for endpoint comparisons"
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


def canonical_json_bytes(value):
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def assert_close(
    actual,
    expected,
    *,
    name,
    tolerance=1e-12,
):
    if not math.isclose(
        float(actual),
        float(expected),
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise RuntimeError(
            f"{name} mismatch: "
            f"{actual} != {expected}"
        )


def normalize_alpha(alpha):
    return f"{float(alpha):.2f}"

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

    squared_deviations = [
        (value - mean) ** 2
        for value in values
    ]

    variance = (
        sum(squared_deviations)
        / (len(values) - 1)
    )

    return math.sqrt(variance)


def summarize_values(values):
    return {
        "mean": compute_mean(values),
        "std": compute_sample_std(values),
        "min": min(values),
        "max": max(values),
        "num_runs": len(values),
    }

def validate_metric_mapping(
    metrics,
    *,
    context,
):
    if not isinstance(metrics, dict):
        raise RuntimeError(
            f"{context} metrics must be a mapping"
        )

    if set(metrics) != set(METRIC_NAMES):
        raise RuntimeError(
            f"{context} metric names do not "
            "match the required contract"
        )

    for metric_name in METRIC_NAMES:
        value = metrics[metric_name]

        if not isinstance(
            value,
            (int, float),
        ):
            raise RuntimeError(
                f"{context}.{metric_name} "
                "must be numeric"
            )

        if not math.isfinite(float(value)):
            raise RuntimeError(
                f"{context}.{metric_name} "
                "must be finite"
            )


def extract_final_evaluation(
    metrics_artifact,
):
    if not isinstance(metrics_artifact, dict):
        raise RuntimeError(
            "Federated metrics artifact "
            "must be a mapping"
        )

    rounds = metrics_artifact.get("rounds")

    if not isinstance(rounds, list):
        raise RuntimeError(
            "Federated metrics artifact "
            "must contain rounds"
        )

    if not rounds:
        raise RuntimeError(
            "Federated metrics artifact "
            "contains no rounds"
        )

    final_round = rounds[-1]

    if not isinstance(final_round, dict):
        raise RuntimeError(
            "Final round must be a mapping"
        )

    evaluation = final_round.get(
        "global_evaluation"
    )

    if not isinstance(evaluation, dict):
        raise RuntimeError(
            "Final round must contain "
            "global_evaluation"
        )

    return evaluation


def validate_evaluation_contract(
    evaluation,
    *,
    client_ids,
    context,
):
    if not isinstance(evaluation, dict):
        raise RuntimeError(
            f"{context} evaluation must "
            "be a mapping"
        )

    expected_keys = {
        "enterprises",
        "macro_metrics",
        "weighted_metrics",
    }

    if set(evaluation) != expected_keys:
        raise RuntimeError(
            f"{context} evaluation keys "
            "do not match the required contract"
        )

    enterprises = evaluation["enterprises"]

    if not isinstance(enterprises, dict):
        raise RuntimeError(
            f"{context} enterprises must "
            "be a mapping"
        )

    if list(enterprises) != list(client_ids):
        raise RuntimeError(
            f"{context} client order mismatch"
        )

    for client_id in client_ids:
        entry = enterprises[client_id]

        if not isinstance(entry, dict):
            raise RuntimeError(
                f"{context}.{client_id} "
                "must be a mapping"
            )

        if set(entry) != {
            "metrics",
            "num_examples",
        }:
            raise RuntimeError(
                f"{context}.{client_id} "
                "keys do not match contract"
            )

        num_examples = entry["num_examples"]

        if (
            not isinstance(num_examples, int)
            or num_examples < 0
        ):
            raise RuntimeError(
                f"{context}.{client_id} "
                "num_examples must be "
                "a non-negative integer"
            )

        validate_metric_mapping(
            entry["metrics"],
            context=(
                f"{context}."
                f"{client_id}"
            ),
        )

    validate_metric_mapping(
        evaluation["macro_metrics"],
        context=f"{context}.macro_metrics",
    )

    validate_metric_mapping(
        evaluation["weighted_metrics"],
        context=(
            f"{context}.weighted_metrics"
        ),
    )


def validate_raw_metrics_artifact(
    artifact,
    *,
    expected_seed,
    expected_clients,
    expected_split,
):
    required_keys = {
        "clients",
        "evaluation_split",
        "federation",
        "format_version",
        "rounds",
        "seed",
    }

    if set(artifact) != required_keys:
        raise RuntimeError(
            "Raw federated metrics artifact "
            "keys do not match contract"
        )

    if artifact["seed"] != expected_seed:
        raise RuntimeError(
            "Raw artifact seed mismatch"
        )

    if artifact["clients"] != expected_clients:
        raise RuntimeError(
            "Raw artifact client order mismatch"
        )

    if (
        artifact["evaluation_split"]
        != expected_split
    ):
        raise RuntimeError(
            "Raw artifact evaluation split mismatch"
        )

    federation = artifact["federation"]

    if not isinstance(federation, dict):
        raise RuntimeError(
            "Raw artifact federation "
            "configuration must be a mapping"
        )

    if set(federation) != {
        "num_rounds",
        "local_epochs",
        "batch_per_epoch",
    }:
        raise RuntimeError(
            "Raw artifact federation keys "
            "do not match contract"
        )

    rounds = artifact["rounds"]

    if not isinstance(rounds, list):
        raise RuntimeError(
            "Raw artifact rounds must be a list"
        )

    if len(rounds) != federation["num_rounds"]:
        raise RuntimeError(
            "Raw artifact round count mismatch"
        )

    for expected_round_id, round_entry in enumerate(
        rounds
    ):
        if not isinstance(round_entry, dict):
            raise RuntimeError(
                "Raw round entry must be a mapping"
            )

        if set(round_entry) != {
            "global_evaluation",
            "local_training",
            "round_id",
        }:
            raise RuntimeError(
                "Raw round entry keys do not "
                "match contract"
            )

        if (
            round_entry["round_id"]
            != expected_round_id
        ):
            raise RuntimeError(
                "Raw round IDs must be "
                "contiguous from zero"
            )

        local_training = round_entry[
            "local_training"
        ]

        if list(local_training) != expected_clients:
            raise RuntimeError(
                "Raw local-training "
                "client order mismatch"
            )

        validate_evaluation_contract(
            round_entry["global_evaluation"],
            client_ids=expected_clients,
            context=(
                f"round_{expected_round_id}"
            ),
        )


def validate_statistic(
    statistic,
    *,
    values,
    context,
):
    expected_keys = {
        "max",
        "mean",
        "min",
        "num_runs",
        "std",
    }

    if set(statistic) != expected_keys:
        raise RuntimeError(
            f"{context} statistic keys "
            "do not match contract"
        )

    expected_mean = compute_mean(values)
    expected_std = compute_sample_std(values)

    if len(values) > 1:
        expected_std = statistics.stdev(values)
    else:
        expected_std = 0.0

    assert_close(
        statistic["min"],
        min(values),
        name=f"{context}.min",
    )

    assert_close(
        statistic["max"],
        max(values),
        name=f"{context}.max",
    )

    assert_close(
        statistic["mean"],
        expected_mean,
        name=f"{context}.mean",
    )

    assert_close(
        statistic["std"],
        expected_std,
        name=f"{context}.std",
    )

    if statistic["num_runs"] != len(values):
        raise RuntimeError(
            f"{context}.num_runs mismatch"
        )


def validate_aggregate(
    aggregate,
    *,
    evaluations,
    context,
):
    expected_aggregate_keys = {
        "alpha",
        "enterprises",
        "macro_metrics",
        "weighted_metrics",
    }

    if set(aggregate) != expected_aggregate_keys:
        raise RuntimeError(
            f"{context} aggregate keys "
            "do not match contract"
        )

    for metric_group in (
        "macro_metrics",
        "weighted_metrics",
    ):
        stored_group = aggregate[metric_group]

        if set(stored_group) != set(METRIC_NAMES):
            raise RuntimeError(
                f"{context}.{metric_group} "
                "metric names mismatch"
            )

        for metric_name in METRIC_NAMES:
            values = [
                evaluation[
                    metric_group
                ][metric_name]
                for evaluation in evaluations
            ]

            validate_statistic(
                stored_group[metric_name],
                values=values,
                context=(
                    f"{context}."
                    f"{metric_group}."
                    f"{metric_name}"
                ),
            )


def validate_cross_seed_example_counts(
    raw_evaluations,
    *,
    alphas,
    seeds,
    clients,
):
    reference = None

    for alpha in alphas:
        alpha_key = normalize_alpha(alpha)

        for seed in seeds:
            evaluation = raw_evaluations[
                alpha_key
            ][str(seed)]

            counts = tuple(
                evaluation["enterprises"][
                    client_id
                ]["num_examples"]
                for client_id in clients
            )

            if reference is None:
                reference = counts

            elif counts != reference:
                raise RuntimeError(
                    "Cross-seed or cross-alpha "
                    "example counts mismatch"
                )


def compare_evaluations(
    actual,
    expected,
    *,
    clients,
    context,
):
    for client_id in clients:
        actual_entry = actual[
            "enterprises"
        ][client_id]

        expected_entry = expected[
            "enterprises"
        ][client_id]

        if (
            actual_entry["num_examples"]
            != expected_entry["num_examples"]
        ):
            raise RuntimeError(
                f"{context}.{client_id} "
                "example count mismatch"
            )

        for metric_name in METRIC_NAMES:
            assert_close(
                actual_entry[
                    "metrics"
                ][metric_name],
                expected_entry[
                    "metrics"
                ][metric_name],
                name=(
                    f"{context}."
                    f"{client_id}."
                    f"{metric_name}"
                ),
            )

    for metric_group in (
        "macro_metrics",
        "weighted_metrics",
    ):
        for metric_name in METRIC_NAMES:
            assert_close(
                actual[
                    metric_group
                ][metric_name],
                expected[
                    metric_group
                ][metric_name],
                name=(
                    f"{context}."
                    f"{metric_group}."
                    f"{metric_name}"
                ),
            )


def load_baseline_evaluation(
    baseline_artifact,
    *,
    method,
    seed,
):
    methods = baseline_artifact.get("methods")

    if not isinstance(methods, dict):
        raise RuntimeError(
            "Baseline suite artifact "
            "must contain methods"
        )

    if method not in methods:
        raise RuntimeError(
            "Required baseline method missing: "
            f"{method}"
        )

    seed_key = str(seed)

    if seed_key not in methods[method]:
        raise RuntimeError(
            "Required baseline seed missing: "
            f"{method}/{seed}"
        )

    return methods[method][seed_key]


def main():
    args = parse_args()

    suite_dir = Path(args.suite_dir)

    artifact_path = (
        suite_dir
        / "relation_aware_ablation.json"
    )

    baseline_artifact_path = (
        Path(args.baseline_suite_dir)
        / "baseline_suite.json"
    )

    if not artifact_path.is_file():
        raise RuntimeError(
            "Relation-aware ablation artifact "
            "does not exist"
        )

    if not baseline_artifact_path.is_file():
        raise RuntimeError(
            "Baseline suite artifact "
            "does not exist"
        )

    artifact = load_json(artifact_path)

    baseline_artifact = load_json(
        baseline_artifact_path
    )

    expected_top_level_keys = {
        "aggregates",
        "alphas",
        "clients",
        "evaluation_split",
        "format_version",
        "runs",
        "seeds",
        "source_config",
    }

    if set(artifact) != expected_top_level_keys:
        raise RuntimeError(
            "Ablation suite artifact keys "
            "do not match contract"
        )

    alphas = artifact["alphas"]
    seeds = artifact["seeds"]
    clients = artifact["clients"]
    evaluation_split = artifact[
        "evaluation_split"
    ]

    if not isinstance(alphas, list) or not alphas:
        raise RuntimeError(
            "Ablation alphas must be "
            "a non-empty list"
        )

    if not isinstance(seeds, list) or not seeds:
        raise RuntimeError(
            "Ablation seeds must be "
            "a non-empty list"
        )

    if len(set(alphas)) != len(alphas):
        raise RuntimeError(
            "Ablation alphas must be unique"
        )

    if len(set(seeds)) != len(seeds):
        raise RuntimeError(
            "Ablation seeds must be unique"
        )

    if not isinstance(clients, list) or not clients:
        raise RuntimeError(
            "Ablation clients must be "
            "a non-empty list"
        )

    expected_alpha_keys = [
        normalize_alpha(alpha)
        for alpha in alphas
    ]

    if (
        list(artifact["runs"])
        != expected_alpha_keys
    ):
        raise RuntimeError(
            "Stored alpha run order mismatch"
        )

    if (
        list(artifact["aggregates"])
        != expected_alpha_keys
    ):
        raise RuntimeError(
            "Stored alpha aggregate order mismatch"
        )

    print()
    print("Validating raw per-alpha artifacts")
    print()

    raw_evaluations = {}

    source_artifacts_checked = 0

    regenerated_runs = {}

    for alpha in alphas:
        alpha_key = normalize_alpha(alpha)

        stored_seed_runs = artifact[
            "runs"
        ][alpha_key]

        expected_seed_keys = [
            str(seed)
            for seed in seeds
        ]

        if (
            list(stored_seed_runs)
            != expected_seed_keys
        ):
            raise RuntimeError(
                "Stored seed run order mismatch "
                f"for alpha={alpha_key}"
            )

        raw_evaluations[alpha_key] = {}
        regenerated_runs[alpha_key] = {}

        for seed in seeds:
            seed_key = str(seed)

            metrics_path = (
                suite_dir
                / "runs"
                / f"alpha_{alpha_key}"
                / f"seed_{seed}"
                / "round_metrics.json"
            )

            if not metrics_path.is_file():
                raise RuntimeError(
                    "Missing raw metrics artifact: "
                    f"{metrics_path}"
                )

            raw_artifact = load_json(
                metrics_path
            )

            validate_raw_metrics_artifact(
                raw_artifact,
                expected_seed=seed,
                expected_clients=clients,
                expected_split=evaluation_split,
            )

            final_evaluation = (
                extract_final_evaluation(
                    raw_artifact
                )
            )

            stored_evaluation = (
                stored_seed_runs[seed_key]
            )

            if (
                canonical_json_bytes(
                    stored_evaluation
                )
                != canonical_json_bytes(
                    final_evaluation
                )
            ):
                raise RuntimeError(
                    "Stored raw run differs "
                    "from source artifact: "
                    f"alpha={alpha_key}, "
                    f"seed={seed}"
                )

            raw_evaluations[
                alpha_key
            ][seed_key] = final_evaluation

            regenerated_runs[
                alpha_key
            ][seed_key] = final_evaluation

            source_artifacts_checked += 1

    validate_cross_seed_example_counts(
        raw_evaluations,
        alphas=alphas,
        seeds=seeds,
        clients=clients,
    )

    regenerated_aggregates = {}

    for alpha in alphas:
        alpha_key = normalize_alpha(alpha)

        evaluations = [
            raw_evaluations[
                alpha_key
            ][str(seed)]
            for seed in seeds
        ]

        validate_aggregate(
            artifact["aggregates"][alpha_key],
            evaluations=evaluations,
            context=f"alpha_{alpha_key}",
        )

        aggregate = {
            "alpha": float(alpha),
            "enterprises": {},
            "macro_metrics": {},
            "weighted_metrics": {},
        }

        for client_id in clients:
            reference_num_examples = (
                evaluations[0][
                    "enterprises"
                ][client_id][
                    "num_examples"
                ]
            )

            enterprise_metrics = {}

            for metric_name in METRIC_NAMES:
                values = [
                    evaluation[
                        "enterprises"
                    ][client_id][
                        "metrics"
                    ][metric_name]
                    for evaluation in evaluations
                ]

                enterprise_metrics[
                    metric_name
                ] = summarize_values(values)

            aggregate["enterprises"][
                client_id
            ] = {
                "metrics": enterprise_metrics,
                "num_examples":
                    reference_num_examples,
            }

        for metric_group in (
            "macro_metrics",
            "weighted_metrics",
        ):
            for metric_name in METRIC_NAMES:
                values = [
                    evaluation[
                        metric_group
                    ][metric_name]
                    for evaluation in evaluations
                ]

                aggregate[
                    metric_group
                ][metric_name] = summarize_values(values)

        regenerated_aggregates[
            alpha_key
        ] = aggregate

    if 0.0 not in alphas:
        raise RuntimeError(
            "Alpha=0 endpoint is required "
            "for FedAvg equivalence validation"
        )

    if 1.0 not in alphas:
        raise RuntimeError(
            "Alpha=1 endpoint is required "
            "for relation-aware reference "
            "validation"
        )

    for seed in seeds:
        seed_key = str(seed)

        alpha_zero_evaluation = (
            raw_evaluations[
                normalize_alpha(0.0)
            ][seed_key]
        )

        fedavg_evaluation = (
            load_baseline_evaluation(
                baseline_artifact,
                method="fedavg",
                seed=seed,
            )
        )

        compare_evaluations(
            alpha_zero_evaluation,
            fedavg_evaluation,
            clients=clients,
            context=(
                f"alpha_zero_vs_fedavg."
                f"seed_{seed}"
            ),
        )

        alpha_one_evaluation = (
            raw_evaluations[
                normalize_alpha(1.0)
            ][seed_key]
        )

        relation_aware_evaluation = (
            load_baseline_evaluation(
                baseline_artifact,
                method=METHOD,
                seed=seed,
            )
        )

        compare_evaluations(
            alpha_one_evaluation,
            relation_aware_evaluation,
            clients=clients,
            context=(
                f"alpha_one_vs_relation_aware."
                f"seed_{seed}"
            ),
        )

    regenerated_artifact = {
        "aggregates": regenerated_aggregates,
        "alphas": alphas,
        "clients": clients,
        "evaluation_split": evaluation_split,
        "format_version": artifact[
            "format_version"
        ],
        "runs": regenerated_runs,
        "seeds": seeds,
        "source_config": artifact[
            "source_config"
        ],
    }

    stored_bytes = canonical_json_bytes(
        artifact
    )

    regenerated_bytes = canonical_json_bytes(
        regenerated_artifact
    )

    if stored_bytes != regenerated_bytes:
        def find_first_difference(
            stored,
            regenerated,
            path="root",
        ):
            if type(stored) is not type(regenerated):
                return (
                    path,
                    stored,
                    regenerated,
                )

            if isinstance(stored, dict):
                stored_keys = list(stored)
                regenerated_keys = list(regenerated)

                if stored_keys != regenerated_keys:
                    return (
                        f"{path}.__keys__",
                        stored_keys,
                        regenerated_keys,
                    )

                for key in stored:
                    difference = find_first_difference(
                        stored[key],
                        regenerated[key],
                        f"{path}.{key}",
                    )

                    if difference is not None:
                        return difference

                return None

            if isinstance(stored, list):
                if len(stored) != len(regenerated):
                    return (
                        f"{path}.__length__",
                        len(stored),
                        len(regenerated),
                    )

                for index, (
                    stored_value,
                    regenerated_value,
                ) in enumerate(
                    zip(stored, regenerated)
                ):
                    difference = find_first_difference(
                        stored_value,
                        regenerated_value,
                        f"{path}[{index}]",
                    )

                    if difference is not None:
                        return difference

                return None

            if stored != regenerated:
                return (
                    path,
                    stored,
                    regenerated,
                )

            return None

        difference = find_first_difference(
            artifact,
            regenerated_artifact,
        )

        print()
        print("FIRST REGENERATION DIFFERENCE")
        print("-----------------------------")
        print("Path:", difference[0])
        print("Stored:", repr(difference[1]))
        print("Regenerated:", repr(difference[2]))

        raise RuntimeError(
            "Independent regeneration of "
            "relation-aware ablation artifact "
            "is not byte-identical"
        )

    suite_sha256 = sha256_bytes(
        stored_bytes
    )

    print("Suite validation diagnostics")
    print("----------------------------")
    print(
        "Source artifacts checked: "
        f"{source_artifacts_checked}"
    )
    print(
        "Suite SHA-256: "
        f"{suite_sha256}"
    )
    print(
        "Suite artifact contract: PASS"
    )
    print(
        "Source artifact contracts: PASS"
    )
    print(
        "Stored raw runs: "
        "IDENTICAL TO SOURCES"
    )
    print(
        "Cross-alpha comparability: PASS"
    )
    print(
        "Cross-seed example counts: PASS"
    )
    print(
        "Sample standard deviations: VERIFIED"
    )
    print(
        "Aggregate statistics: VERIFIED"
    )
    print(
        "Alpha=0 endpoint equivalence "
        "to FedAvg: PASS"
    )
    print(
        "Alpha=1 endpoint equivalence "
        "to relation-aware reference: PASS"
    )
    print(
        "Independent regeneration: "
        "BYTE-IDENTICAL"
    )

    print()
    print(
        "Relation-aware ablation "
        "validation: PASS"
    )


if __name__ == "__main__":
    main()



