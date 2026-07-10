import hashlib
import json
from pathlib import Path


FEDAVG_PATH = Path(
    "output/fedrelshield/"
    "federated_training/"
    "round_metrics.json"
)

RELATION_AWARE_PATH = Path(
    "output/fedrelshield/"
    "relation_aware/"
    "round_metrics.json"
)

ALPHA_ZERO_PATH = Path(
    "output/fedrelshield/"
    "relation_aware_alpha0_check/"
    "round_metrics.json"
)

ALPHA_ONE_PATH = Path(
    "output/fedrelshield/"
    "relation_aware_alpha1_check/"
    "round_metrics.json"
)


METRIC_NAMES = (
    "hits@1",
    "hits@3",
    "hits@10",
    "mrr",
)


def load_json(path):
    if not path.is_file():
        raise RuntimeError(
            f"Required artifact does not exist: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def canonical_json_bytes(artifact):
    return json.dumps(
        artifact,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_artifact(artifact):
    return hashlib.sha256(
        canonical_json_bytes(artifact)
    ).hexdigest()


def get_final_round(artifact):
    rounds = artifact.get("rounds")

    if not isinstance(rounds, list):
        raise RuntimeError(
            "Artifact rounds must be a list"
        )

    if not rounds:
        raise RuntimeError(
            "Artifact must contain at least one round"
        )

    return rounds[-1]


def get_evaluation(artifact):
    final_round = get_final_round(artifact)

    try:
        return final_round["global_evaluation"]

    except KeyError as error:
        raise RuntimeError(
            "Final round is missing global_evaluation"
        ) from error


def validate_basic_contract(
    name,
    artifact,
):
    required_top_level_keys = {
        "clients",
        "evaluation_split",
        "federation",
        "format_version",
        "rounds",
        "seed",
    }

    if set(artifact) != required_top_level_keys:
        raise RuntimeError(
            f"{name}: unexpected top-level artifact keys"
        )

    clients = artifact["clients"]

    if not isinstance(clients, list):
        raise RuntimeError(
            f"{name}: clients must be a list"
        )

    if not clients:
        raise RuntimeError(
            f"{name}: clients must be non-empty"
        )

    if len(clients) != len(set(clients)):
        raise RuntimeError(
            f"{name}: duplicate client IDs"
        )

    if not isinstance(
        artifact["evaluation_split"],
        str,
    ):
        raise RuntimeError(
            f"{name}: evaluation_split must be a string"
        )

    if not isinstance(
        artifact["federation"],
        dict,
    ):
        raise RuntimeError(
            f"{name}: federation must be a mapping"
        )

    if not isinstance(
        artifact["format_version"],
        str,
    ):
        raise RuntimeError(
            f"{name}: format_version must be a string"
        )

    if not isinstance(
        artifact["seed"],
        int,
    ):
        raise RuntimeError(
            f"{name}: seed must be an integer"
        )

    evaluation = get_evaluation(artifact)

    required_evaluation_keys = {
        "enterprises",
        "macro_metrics",
        "weighted_metrics",
    }

    if set(evaluation) != required_evaluation_keys:
        raise RuntimeError(
            f"{name}: unexpected evaluation keys"
        )

    enterprises = evaluation["enterprises"]

    if list(enterprises) != clients:
        raise RuntimeError(
            f"{name}: enterprise order does not match "
            "client order"
        )

    for enterprise_id in clients:
        enterprise_result = enterprises[
            enterprise_id
        ]

        if set(enterprise_result) != {
            "metrics",
            "num_examples",
        }:
            raise RuntimeError(
                f"{name}: invalid enterprise result "
                f"for {enterprise_id}"
            )

        metrics = enterprise_result["metrics"]

        if set(metrics) != set(METRIC_NAMES):
            raise RuntimeError(
                f"{name}: unexpected metric names "
                f"for {enterprise_id}"
            )

        if (
            not isinstance(
                enterprise_result["num_examples"],
                int,
            )
            or enterprise_result["num_examples"] <= 0
        ):
            raise RuntimeError(
                f"{name}: invalid example count "
                f"for {enterprise_id}"
            )

    for aggregate_name in (
        "macro_metrics",
        "weighted_metrics",
    ):
        metrics = evaluation[aggregate_name]

        if set(metrics) != set(METRIC_NAMES):
            raise RuntimeError(
                f"{name}: unexpected aggregate metrics "
                f"for {aggregate_name}"
            )


def validate_cross_artifact_contract(
    artifacts,
):
    reference_name = next(iter(artifacts))
    reference = artifacts[reference_name]

    reference_clients = reference["clients"]
    reference_split = reference[
        "evaluation_split"
    ]
    reference_seed = reference["seed"]

    reference_evaluation = get_evaluation(
        reference
    )

    for name, artifact in artifacts.items():
        if artifact["clients"] != reference_clients:
            raise RuntimeError(
                f"{name}: client order differs from "
                f"{reference_name}"
            )

        if (
            artifact["evaluation_split"]
            != reference_split
        ):
            raise RuntimeError(
                f"{name}: evaluation split differs from "
                f"{reference_name}"
            )

        if artifact["seed"] != reference_seed:
            raise RuntimeError(
                f"{name}: seed differs from "
                f"{reference_name}"
            )

        evaluation = get_evaluation(artifact)

        for enterprise_id in reference_clients:
            reference_result = (
                reference_evaluation[
                    "enterprises"
                ][enterprise_id]
            )

            result = evaluation[
                "enterprises"
            ][enterprise_id]

            if (
                result["num_examples"]
                != reference_result["num_examples"]
            ):
                raise RuntimeError(
                    f"{name}: example count differs for "
                    f"{enterprise_id}"
                )

            if (
                set(result["metrics"])
                != set(reference_result["metrics"])
            ):
                raise RuntimeError(
                    f"{name}: metric names differ for "
                    f"{enterprise_id}"
                )


def validate_evaluations_identical(
    left_name,
    left,
    right_name,
    right,
):
    left_evaluation = get_evaluation(left)
    right_evaluation = get_evaluation(right)

    if left_evaluation != right_evaluation:
        raise RuntimeError(
            f"{left_name} and {right_name} "
            "evaluation metrics are not identical"
        )


def validate_artifacts_identical(
    left_name,
    left,
    right_name,
    right,
):
    if canonical_json_bytes(
        left
    ) != canonical_json_bytes(
        right
    ):
        raise RuntimeError(
            f"{left_name} and {right_name} "
            "artifacts are not identical"
        )


def validate_evaluations_different(
    left_name,
    left,
    right_name,
    right,
):
    left_evaluation = get_evaluation(left)
    right_evaluation = get_evaluation(right)

    if left_evaluation == right_evaluation:
        raise RuntimeError(
            f"{left_name} and {right_name} "
            "evaluation metrics unexpectedly match"
        )


def print_mrr_diagnostics(
    name,
    artifact,
):
    evaluation = get_evaluation(artifact)

    macro_mrr = evaluation[
        "macro_metrics"
    ]["mrr"]

    weighted_mrr = evaluation[
        "weighted_metrics"
    ]["mrr"]

    print(
        f"{name:<24} "
        f"macro={macro_mrr:.6f} "
        f"weighted={weighted_mrr:.6f}"
    )


def main():
    artifacts = {
        "fedavg": load_json(
            FEDAVG_PATH
        ),
        "relation_aware": load_json(
            RELATION_AWARE_PATH
        ),
        "alpha_zero": load_json(
            ALPHA_ZERO_PATH
        ),
        "alpha_one": load_json(
            ALPHA_ONE_PATH
        ),
    }

    print()
    print("Validating endpoint artifacts")
    print("-----------------------------")

    for name, artifact in artifacts.items():
        validate_basic_contract(
            name=name,
            artifact=artifact,
        )

    print("Artifact contracts: PASS")

    validate_cross_artifact_contract(
        artifacts
    )

    print("Cross-artifact comparability: PASS")

    validate_artifacts_identical(
        "fedavg",
        artifacts["fedavg"],
        "alpha_zero",
        artifacts["alpha_zero"],
    )

    print(
        "Alpha=0 artifact equivalence "
        "to FedAvg: PASS"
    )

    validate_evaluations_identical(
        "fedavg",
        artifacts["fedavg"],
        "alpha_zero",
        artifacts["alpha_zero"],
    )

    print(
        "Alpha=0 evaluation equivalence "
        "to FedAvg: PASS"
    )

    validate_artifacts_identical(
        "relation_aware",
        artifacts["relation_aware"],
        "alpha_one",
        artifacts["alpha_one"],
    )

    print(
        "Alpha=1 artifact equivalence "
        "to relation-aware reference: PASS"
    )

    validate_evaluations_identical(
        "relation_aware",
        artifacts["relation_aware"],
        "alpha_one",
        artifacts["alpha_one"],
    )

    print(
        "Alpha=1 evaluation equivalence "
        "to relation-aware reference: PASS"
    )

    validate_evaluations_different(
        "fedavg",
        artifacts["fedavg"],
        "alpha_one",
        artifacts["alpha_one"],
    )

    print(
        "Alpha=1 non-degeneracy "
        "from FedAvg: PASS"
    )

    print()
    print("Endpoint diagnostics")
    print("--------------------")

    for name, artifact in artifacts.items():
        print_mrr_diagnostics(
            name=name,
            artifact=artifact,
        )

    print()
    print("Artifact SHA-256")
    print("----------------")

    for name, artifact in artifacts.items():
        print(
            f"{name:<24} "
            f"{sha256_artifact(artifact)}"
        )

    print()
    print(
        "Relation-aware endpoint validation: PASS"
    )


if __name__ == "__main__":
    main()
