import argparse
import json
import os
from pathlib import Path


METHOD_ORDER = (
    "zero_shot",
    "local_only",
    "centralized",
    "fedavg",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare validated FedRelShield "
            "baseline and federated experiment artifacts."
        )
    )

    parser.add_argument(
        "--zero-shot",
        default=(
            "output/fedrelshield/baselines/"
            "zero_shot/baseline_metrics.json"
        ),
    )

    parser.add_argument(
        "--local-only",
        default=(
            "output/fedrelshield/baselines/"
            "local_only/baseline_metrics.json"
        ),
    )

    parser.add_argument(
        "--centralized",
        default=(
            "output/fedrelshield/baselines/"
            "centralized/baseline_metrics.json"
        ),
    )

    parser.add_argument(
        "--fedavg",
        default=(
            "output/fedrelshield/"
            "federated_training/round_metrics.json"
        ),
    )

    parser.add_argument(
        "--output",
        default=(
            "output/fedrelshield/comparisons/"
            "baseline_comparison.json"
        ),
    )

    return parser.parse_args()


def load_json(path):
    path = Path(
        os.path.expanduser(path)
    )

    if not path.is_file():
        raise RuntimeError(
            f"Artifact does not exist: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def write_json(artifact, path):
    path = Path(
        os.path.expanduser(path)
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            artifact,
            file,
            indent=2,
            sort_keys=True,
        )

        file.write("\n")

    os.replace(
        temporary_path,
        path,
    )


def validate_evaluation(
    evaluation,
    expected_clients,
    description,
):
    required_keys = {
        "enterprises",
        "macro_metrics",
        "weighted_metrics",
    }

    if set(evaluation) != required_keys:
        raise RuntimeError(
            f"{description}: evaluation keys "
            "do not match the required contract"
        )

    enterprises = evaluation[
        "enterprises"
    ]

    if set(enterprises) != set(
        expected_clients
    ):
        raise RuntimeError(
            f"{description}: enterprise "
            "coverage mismatch"
        )

    macro_metrics = evaluation[
        "macro_metrics"
    ]

    weighted_metrics = evaluation[
        "weighted_metrics"
    ]

    if not macro_metrics:
        raise RuntimeError(
            f"{description}: macro metrics "
            "are empty"
        )

    if not weighted_metrics:
        raise RuntimeError(
            f"{description}: weighted metrics "
            "are empty"
        )

    if set(macro_metrics) != set(
        weighted_metrics
    ):
        raise RuntimeError(
            f"{description}: macro and weighted "
            "metric names differ"
        )

    metric_names = set(
        macro_metrics
    )

    for client_id in expected_clients:
        enterprise_result = enterprises[
            client_id
        ]

        if set(enterprise_result) != {
            "num_examples",
            "metrics",
        }:
            raise RuntimeError(
                f"{description}: invalid enterprise "
                f"result contract: {client_id}"
            )

        if (
            enterprise_result["num_examples"]
            <= 0
        ):
            raise RuntimeError(
                f"{description}: enterprise has "
                f"no evaluation examples: {client_id}"
            )

        if set(
            enterprise_result["metrics"]
        ) != metric_names:
            raise RuntimeError(
                f"{description}: metric names "
                f"differ for {client_id}"
            )

    return metric_names


def validate_baseline_artifact(
    artifact,
    expected_name,
):
    required_keys = {
        "format_version",
        "baseline",
        "seed",
        "evaluation_split",
        "clients",
        "evaluation",
    }

    allowed_keys = set(
        required_keys
    )

    if expected_name == "local_only":
        allowed_keys.add(
            "local_training"
        )

    if expected_name == "centralized":
        allowed_keys.add(
            "centralized_training"
        )

    if set(artifact) != allowed_keys:
        raise RuntimeError(
            f"{expected_name}: baseline artifact "
            "keys do not match the contract"
        )

    if artifact["format_version"] != "1.0":
        raise RuntimeError(
            f"{expected_name}: unexpected "
            "artifact format version"
        )

    if artifact["baseline"] != expected_name:
        raise RuntimeError(
            f"{expected_name}: artifact baseline "
            "name mismatch"
        )

    if not artifact["clients"]:
        raise RuntimeError(
            f"{expected_name}: client list is empty"
        )

    validate_evaluation(
        evaluation=artifact["evaluation"],
        expected_clients=artifact["clients"],
        description=expected_name,
    )


def validate_federated_artifact(
    artifact,
):
    required_keys = {
        "format_version",
        "seed",
        "evaluation_split",
        "clients",
        "federation",
        "rounds",
    }

    if set(artifact) != required_keys:
        raise RuntimeError(
            "fedavg: federated artifact keys "
            "do not match the contract"
        )

    if artifact["format_version"] != "1.0":
        raise RuntimeError(
            "fedavg: unexpected artifact "
            "format version"
        )

    if not artifact["clients"]:
        raise RuntimeError(
            "fedavg: client list is empty"
        )

    rounds = artifact["rounds"]

    if not rounds:
        raise RuntimeError(
            "fedavg: federated artifact "
            "contains no rounds"
        )

    if len(rounds) != artifact[
        "federation"
    ]["num_rounds"]:
        raise RuntimeError(
            "fedavg: round count does not "
            "match federation configuration"
        )

    for expected_round_id, round_result in enumerate(
        rounds
    ):
        if set(round_result) != {
            "round_id",
            "local_training",
            "global_evaluation",
        }:
            raise RuntimeError(
                "fedavg: round result keys do "
                "not match the contract"
            )

        if round_result["round_id"] != (
            expected_round_id
        ):
            raise RuntimeError(
                "fedavg: round IDs are not "
                "contiguous and zero-based"
            )

        if set(
            round_result["local_training"]
        ) != set(artifact["clients"]):
            raise RuntimeError(
                "fedavg: local training "
                "client coverage mismatch"
            )

        validate_evaluation(
            evaluation=round_result[
                "global_evaluation"
            ],
            expected_clients=artifact[
                "clients"
            ],
            description=(
                f"fedavg round {expected_round_id}"
            ),
        )


def evaluation_example_counts(
    evaluation,
    clients,
):
    return {
        client_id: evaluation[
            "enterprises"
        ][client_id]["num_examples"]
        for client_id in clients
    }


def validate_cross_method_contract(
    method_evaluations,
    expected_clients,
):
    reference_method = METHOD_ORDER[0]

    reference_evaluation = (
        method_evaluations[
            reference_method
        ]
    )

    reference_metric_names = set(
        reference_evaluation[
            "macro_metrics"
        ]
    )

    reference_example_counts = (
        evaluation_example_counts(
            evaluation=reference_evaluation,
            clients=expected_clients,
        )
    )

    for method_name in METHOD_ORDER[1:]:
        evaluation = method_evaluations[
            method_name
        ]

        metric_names = set(
            evaluation["macro_metrics"]
        )

        if metric_names != (
            reference_metric_names
        ):
            raise RuntimeError(
                "Cross-method metric names differ: "
                f"{method_name}"
            )

        example_counts = (
            evaluation_example_counts(
                evaluation=evaluation,
                clients=expected_clients,
            )
        )

        if example_counts != (
            reference_example_counts
        ):
            raise RuntimeError(
                "Cross-method evaluation example "
                f"counts differ: {method_name}"
            )

        for client_id in expected_clients:
            enterprise_metric_names = set(
                evaluation[
                    "enterprises"
                ][client_id]["metrics"]
            )

            if enterprise_metric_names != (
                reference_metric_names
            ):
                raise RuntimeError(
                    "Cross-method enterprise metric "
                    "names differ: "
                    f"{method_name}/{client_id}"
                )


def build_summary(methods):
    best_macro_method = max(
        METHOD_ORDER,
        key=lambda method_name: (
            methods[
                method_name
            ]["evaluation"][
                "macro_metrics"
            ]["mrr"]
        ),
    )

    best_weighted_method = max(
        METHOD_ORDER,
        key=lambda method_name: (
            methods[
                method_name
            ]["evaluation"][
                "weighted_metrics"
            ]["mrr"]
        ),
    )

    return {
        "best_macro_mrr": {
            "method": best_macro_method,
            "value": methods[
                best_macro_method
            ]["evaluation"][
                "macro_metrics"
            ]["mrr"],
        },
        "best_weighted_mrr": {
            "method": best_weighted_method,
            "value": methods[
                best_weighted_method
            ]["evaluation"][
                "weighted_metrics"
            ]["mrr"],
        },
    }


def validate_comparison_artifact(
    artifact,
):
    required_keys = {
        "format_version",
        "seed",
        "evaluation_split",
        "clients",
        "methods",
        "summary",
    }

    if set(artifact) != required_keys:
        raise RuntimeError(
            "Comparison artifact keys do not "
            "match the required contract"
        )

    if artifact["format_version"] != "1.0":
        raise RuntimeError(
            "Unexpected comparison artifact "
            "format version"
        )

    if set(
        artifact["methods"]
    ) != set(METHOD_ORDER):
        raise RuntimeError(
            "Comparison methods do not match "
            "the required contract"
        )

    for method_name in METHOD_ORDER:
        method_result = artifact[
            "methods"
        ][method_name]

        required_method_keys = {
            "source_type",
            "evaluation",
        }

        if method_name == "fedavg":
            required_method_keys.add(
                "round_id"
            )

        if set(method_result) != (
            required_method_keys
        ):
            raise RuntimeError(
                "Comparison method keys do not "
                f"match contract: {method_name}"
            )

        validate_evaluation(
            evaluation=method_result[
                "evaluation"
            ],
            expected_clients=artifact[
                "clients"
            ],
            description=(
                f"comparison/{method_name}"
            ),
        )

    expected_summary = build_summary(
        artifact["methods"]
    )

    if artifact["summary"] != (
        expected_summary
    ):
        raise RuntimeError(
            "Comparison summary does not match "
            "the method evaluations"
        )


def print_comparison_table(
    methods,
):
    print()
    print("Baseline comparison")
    print("-------------------")
    print(
        f"{'Method':<16}"
        f"{'Macro MRR':>14}"
        f"{'Weighted MRR':>16}"
    )
    print(
        f"{'-' * 16}"
        f"{'-' * 14}"
        f"{'-' * 16}"
    )

    for method_name in METHOD_ORDER:
        evaluation = methods[
            method_name
        ]["evaluation"]

        macro_mrr = evaluation[
            "macro_metrics"
        ]["mrr"]

        weighted_mrr = evaluation[
            "weighted_metrics"
        ]["mrr"]

        print(
            f"{method_name:<16}"
            f"{macro_mrr:>14.6f}"
            f"{weighted_mrr:>16.6f}"
        )


def main():
    args = parse_args()

    zero_shot = load_json(
        args.zero_shot
    )

    local_only = load_json(
        args.local_only
    )

    centralized = load_json(
        args.centralized
    )

    fedavg = load_json(
        args.fedavg
    )

    validate_baseline_artifact(
        artifact=zero_shot,
        expected_name="zero_shot",
    )

    validate_baseline_artifact(
        artifact=local_only,
        expected_name="local_only",
    )

    validate_baseline_artifact(
        artifact=centralized,
        expected_name="centralized",
    )

    validate_federated_artifact(
        fedavg
    )

    expected_seed = zero_shot[
        "seed"
    ]

    expected_clients = zero_shot[
        "clients"
    ]

    expected_evaluation_split = (
        zero_shot["evaluation_split"]
    )

    for method_name, artifact in (
        ("local_only", local_only),
        ("centralized", centralized),
        ("fedavg", fedavg),
    ):
        if artifact["seed"] != expected_seed:
            raise RuntimeError(
                "Cross-method seed mismatch: "
                f"{method_name}"
            )

        if artifact["clients"] != (
            expected_clients
        ):
            raise RuntimeError(
                "Cross-method client order "
                f"mismatch: {method_name}"
            )

        if artifact["evaluation_split"] != (
            expected_evaluation_split
        ):
            raise RuntimeError(
                "Cross-method evaluation split "
                f"mismatch: {method_name}"
            )

    final_fedavg_round = (
        fedavg["rounds"][-1]
    )

    methods = {
        "zero_shot": {
            "source_type": "baseline",
            "evaluation": zero_shot[
                "evaluation"
            ],
        },
        "local_only": {
            "source_type": "baseline",
            "evaluation": local_only[
                "evaluation"
            ],
        },
        "centralized": {
            "source_type": "baseline",
            "evaluation": centralized[
                "evaluation"
            ],
        },
        "fedavg": {
            "source_type": (
                "federated_final_round"
            ),
            "round_id": final_fedavg_round[
                "round_id"
            ],
            "evaluation": final_fedavg_round[
                "global_evaluation"
            ],
        },
    }

    method_evaluations = {
        method_name: methods[
            method_name
        ]["evaluation"]
        for method_name in METHOD_ORDER
    }

    validate_cross_method_contract(
        method_evaluations=method_evaluations,
        expected_clients=expected_clients,
    )

    artifact = {
        "format_version": "1.0",
        "seed": expected_seed,
        "evaluation_split": (
            expected_evaluation_split
        ),
        "clients": expected_clients,
        "methods": methods,
        "summary": build_summary(
            methods
        ),
    }

    validate_comparison_artifact(
        artifact
    )

    write_json(
        artifact=artifact,
        path=args.output,
    )

    serialized_artifact = load_json(
        args.output
    )

    validate_comparison_artifact(
        serialized_artifact
    )

    if serialized_artifact != artifact:
        raise RuntimeError(
            "Serialized comparison artifact "
            "does not match the in-memory artifact"
        )

    print_comparison_table(
        methods
    )

    print()
    print(
        "Best macro MRR: "
        f"{artifact['summary']['best_macro_mrr']['method']} "
        f"({artifact['summary']['best_macro_mrr']['value']:.6f})"
    )

    print(
        "Best weighted MRR: "
        f"{artifact['summary']['best_weighted_mrr']['method']} "
        f"({artifact['summary']['best_weighted_mrr']['value']:.6f})"
    )

    print()
    print("Source artifact contracts: PASS")
    print("Cross-method seed consistency: PASS")
    print("Cross-method client order: PASS")
    print("Cross-method evaluation split: PASS")
    print("Cross-method enterprise coverage: PASS")
    print("Cross-method metric names: PASS")
    print("Cross-method example counts: PASS")
    print("Final FedAvg round selection: PASS")
    print("Comparison artifact validation: PASS")
    print("Serialized artifact validation: PASS")
    print()
    print("Baseline comparison: PASS")
    print(f"Artifact: {args.output}")


if __name__ == "__main__":
    main()
