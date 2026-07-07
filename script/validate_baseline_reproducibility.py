import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate exact reproducibility of a "
            "FedRelShield baseline experiment."
        )
    )

    parser.add_argument(
        "-c",
        "--config",
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1024,
    )

    return parser.parse_args()


def run_command(command):
    print()
    print("$ " + " ".join(command))

    subprocess.run(
        command,
        check=True,
    )


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def write_yaml(config, path):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
        )


def file_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def validate_finite_metrics(
    metrics,
    description,
):
    if not metrics:
        raise RuntimeError(
            f"{description} must be non-empty"
        )

    for metric_name, value in metrics.items():
        if not isinstance(value, (int, float)):
            raise RuntimeError(
                f"{description} metric is not numeric: "
                f"{metric_name}"
            )

        if not math.isfinite(value):
            raise RuntimeError(
                f"{description} metric is not finite: "
                f"{metric_name}"
            )


def validate_local_training_contract(
    artifact,
    expected_clients,
):
    if "local_training" not in artifact:
        raise RuntimeError(
            "Local-only artifact is missing "
            "local_training"
        )

    local_training = artifact[
        "local_training"
    ]

    if set(local_training) != set(expected_clients):
        raise RuntimeError(
            "Local training client coverage mismatch"
        )

    for client_id in expected_clients:
        result = local_training[client_id]

        if set(result) != {
            "average_loss",
            "num_examples",
        }:
            raise RuntimeError(
                "Local training result keys do not "
                f"match contract: {client_id}"
            )

        num_examples = result[
            "num_examples"
        ]

        average_loss = result[
            "average_loss"
        ]

        if (
            not isinstance(num_examples, int)
            or isinstance(num_examples, bool)
            or num_examples <= 0
        ):
            raise RuntimeError(
                "Local training example count must "
                f"be a positive integer: {client_id}"
            )

        if not isinstance(
            average_loss,
            (int, float),
        ):
            raise RuntimeError(
                "Local training loss is not numeric: "
                f"{client_id}"
            )

        if not math.isfinite(average_loss):
            raise RuntimeError(
                "Local training loss is not finite: "
                f"{client_id}"
            )


def validate_artifact_contract(
    artifact,
    expected_baseline,
    expected_seed,
    expected_clients,
):
    common_required_keys = {
        "format_version",
        "baseline",
        "seed",
        "evaluation_split",
        "clients",
        "evaluation",
    }

    if expected_baseline == "zero_shot":
        required_keys = common_required_keys

    elif expected_baseline == "local_only":
        required_keys = (
            common_required_keys
            | {"local_training"}
        )

    else:
        raise RuntimeError(
            "Unsupported baseline for reproducibility "
            f"validation: {expected_baseline}"
        )

    if set(artifact) != required_keys:
        raise RuntimeError(
            "Baseline artifact keys do not "
            "match the required contract"
        )

    if artifact["format_version"] != "1.0":
        raise RuntimeError(
            "Unexpected artifact format version"
        )

    if artifact["baseline"] != expected_baseline:
        raise RuntimeError(
            "Artifact baseline does not match config"
        )

    if artifact["seed"] != expected_seed:
        raise RuntimeError(
            "Artifact seed does not match "
            "the requested seed"
        )

    if artifact["clients"] != expected_clients:
        raise RuntimeError(
            "Artifact client order does not "
            "match the configured clients"
        )

    evaluation_split = artifact[
        "evaluation_split"
    ]

    if (
        not isinstance(evaluation_split, str)
        or not evaluation_split
    ):
        raise RuntimeError(
            "Evaluation split must be a non-empty string"
        )

    evaluation = artifact["evaluation"]

    required_evaluation_keys = {
        "enterprises",
        "macro_metrics",
        "weighted_metrics",
    }

    if set(evaluation) != required_evaluation_keys:
        raise RuntimeError(
            "Evaluation artifact keys do not "
            "match the required contract"
        )

    if set(
        evaluation["enterprises"]
    ) != set(expected_clients):
        raise RuntimeError(
            "Enterprise evaluation coverage mismatch"
        )

    validate_finite_metrics(
        evaluation["macro_metrics"],
        "Macro metrics",
    )

    validate_finite_metrics(
        evaluation["weighted_metrics"],
        "Weighted metrics",
    )

    for enterprise_id in expected_clients:
        enterprise_result = evaluation[
            "enterprises"
        ][enterprise_id]

        if set(enterprise_result) != {
            "num_examples",
            "metrics",
        }:
            raise RuntimeError(
                "Enterprise result keys do not "
                f"match contract: {enterprise_id}"
            )

        num_examples = enterprise_result[
            "num_examples"
        ]

        if (
            not isinstance(num_examples, int)
            or isinstance(num_examples, bool)
            or num_examples <= 0
        ):
            raise RuntimeError(
                "Enterprise result example count "
                "must be a positive integer: "
                f"{enterprise_id}"
            )

        validate_finite_metrics(
            enterprise_result["metrics"],
            (
                "Enterprise metrics for "
                f"{enterprise_id}"
            ),
        )

    if expected_baseline == "local_only":
        validate_local_training_contract(
            artifact=artifact,
            expected_clients=expected_clients,
        )


def main():
    args = parse_args()

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as file:
        base_cfg = yaml.safe_load(file)

    expected_baseline = base_cfg[
        "baseline"
    ]["name"]

    expected_clients = list(
        base_cfg["clients"]
    )

    base_output = Path(
        os.path.expanduser(
            base_cfg["output_dir"]
        )
    )

    parent_output = base_output.parent

    first_output = (
        parent_output
        / f"{expected_baseline}_repro_run_1"
    )

    second_output = (
        parent_output
        / f"{expected_baseline}_repro_run_2"
    )

    generated_config_dir = (
        parent_output
        / "reproducibility_configs"
    )

    generated_config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    first_cfg = copy.deepcopy(base_cfg)
    second_cfg = copy.deepcopy(base_cfg)

    first_cfg["output_dir"] = str(
        first_output
    )

    second_cfg["output_dir"] = str(
        second_output
    )

    first_cfg_path = (
        generated_config_dir
        / f"{expected_baseline}_run_1.yaml"
    )

    second_cfg_path = (
        generated_config_dir
        / f"{expected_baseline}_run_2.yaml"
    )

    write_yaml(
        first_cfg,
        first_cfg_path,
    )

    write_yaml(
        second_cfg,
        second_cfg_path,
    )

    shutil.rmtree(
        first_output,
        ignore_errors=True,
    )

    shutil.rmtree(
        second_output,
        ignore_errors=True,
    )

    print()
    print("Running baseline reproducibility trial 1")

    run_command(
        [
            sys.executable,
            "script/run_baseline.py",
            "-c",
            str(first_cfg_path),
            "--seed",
            str(args.seed),
        ]
    )

    print()
    print("Running baseline reproducibility trial 2")

    run_command(
        [
            sys.executable,
            "script/run_baseline.py",
            "-c",
            str(second_cfg_path),
            "--seed",
            str(args.seed),
        ]
    )

    first_artifact_path = (
        first_output
        / "baseline_metrics.json"
    )

    second_artifact_path = (
        second_output
        / "baseline_metrics.json"
    )

    if not first_artifact_path.is_file():
        raise RuntimeError(
            "First reproducibility artifact "
            "does not exist"
        )

    if not second_artifact_path.is_file():
        raise RuntimeError(
            "Second reproducibility artifact "
            "does not exist"
        )

    first_artifact = load_json(
        first_artifact_path
    )

    second_artifact = load_json(
        second_artifact_path
    )

    validate_artifact_contract(
        artifact=first_artifact,
        expected_baseline=expected_baseline,
        expected_seed=args.seed,
        expected_clients=expected_clients,
    )

    validate_artifact_contract(
        artifact=second_artifact,
        expected_baseline=expected_baseline,
        expected_seed=args.seed,
        expected_clients=expected_clients,
    )

    first_hash = file_sha256(
        first_artifact_path
    )

    second_hash = file_sha256(
        second_artifact_path
    )

    print()
    print("Reproducibility diagnostics")
    print("---------------------------")
    print(
        "Run 1 SHA-256: "
        f"{first_hash}"
    )
    print(
        "Run 2 SHA-256: "
        f"{second_hash}"
    )

    artifacts_equal = (
        first_artifact == second_artifact
    )

    files_equal = (
        first_hash == second_hash
    )

    print(
        "JSON artifacts: "
        + (
            "IDENTICAL"
            if artifacts_equal
            else "DIFFERENT"
        )
    )

    print(
        "Serialized artifacts: "
        + (
            "IDENTICAL"
            if files_equal
            else "DIFFERENT"
        )
    )

    if not artifacts_equal:
        raise RuntimeError(
            "Baseline JSON artifacts differ "
            "between identical executions"
        )

    if not files_equal:
        raise RuntimeError(
            "Baseline serialized artifacts differ "
            "between identical executions"
        )

    first_evaluation = first_artifact[
        "evaluation"
    ]

    second_evaluation = second_artifact[
        "evaluation"
    ]

    for enterprise_id in expected_clients:
        if (
            first_evaluation["enterprises"][
                enterprise_id
            ]
            != second_evaluation["enterprises"][
                enterprise_id
            ]
        ):
            raise RuntimeError(
                "Per-enterprise metrics differ: "
                f"{enterprise_id}"
            )

    if (
        first_evaluation["macro_metrics"]
        != second_evaluation["macro_metrics"]
    ):
        raise RuntimeError(
            "Macro metrics differ"
        )

    if (
        first_evaluation["weighted_metrics"]
        != second_evaluation["weighted_metrics"]
    ):
        raise RuntimeError(
            "Weighted metrics differ"
        )

    if expected_baseline == "local_only":
        first_local_training = first_artifact[
            "local_training"
        ]

        second_local_training = second_artifact[
            "local_training"
        ]

        for client_id in expected_clients:
            if (
                first_local_training[client_id]
                != second_local_training[client_id]
            ):
                raise RuntimeError(
                    "Local training result differs: "
                    f"{client_id}"
                )

    print()
    print("Artifact contract: PASS")
    print("Client coverage: PASS")

    if expected_baseline == "local_only":
        print("Local training contract: PASS")
        print(
            "Per-client training losses: IDENTICAL"
        )

    print("Per-enterprise metrics: IDENTICAL")
    print("Macro metrics: IDENTICAL")
    print("Weighted metrics: IDENTICAL")
    print("JSON artifacts: IDENTICAL")
    print("Serialized artifacts: IDENTICAL")
    print()
    print(
        "Baseline reproducibility "
        "validation: PASS"
    )


if __name__ == "__main__":
    main()
