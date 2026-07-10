import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate exact reproducibility of a "
            "FedRelShield federated training experiment."
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


def states_equal(state_a, state_b):
    if state_a.keys() != state_b.keys():
        return False

    return all(
        torch.equal(
            state_a[name],
            state_b[name],
        )
        for name in state_a
    )


def validate_artifact_contract(
    artifact,
    expected_seed,
    expected_clients,
    expected_federation,
    expected_evaluation_split,
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
            "Federated artifact keys do not "
            "match the required contract"
        )

    if artifact["format_version"] != "1.0":
        raise RuntimeError(
            "Unexpected artifact format version"
        )

    if artifact["seed"] != expected_seed:
        raise RuntimeError(
            "Artifact seed does not match "
            "the requested seed"
        )

    if artifact["evaluation_split"] != (
        expected_evaluation_split
    ):
        raise RuntimeError(
            "Artifact evaluation split does not "
            "match the configured split"
        )

    if artifact["clients"] != expected_clients:
        raise RuntimeError(
            "Artifact client order does not "
            "match the configured clients"
        )

    if artifact["federation"] != (
        expected_federation
    ):
        raise RuntimeError(
            "Artifact federation configuration "
            "does not match the config"
        )

    rounds = artifact["rounds"]

    if len(rounds) != expected_federation[
        "num_rounds"
    ]:
        raise RuntimeError(
            "Federated artifact round count "
            "does not match the config"
        )

    expected_round_keys = {
        "round_id",
        "local_training",
        "global_evaluation",
    }

    for expected_round_id, round_result in enumerate(
        rounds
    ):
        if set(round_result) != expected_round_keys:
            raise RuntimeError(
                "Federated round keys do not "
                "match the required contract"
            )

        if round_result["round_id"] != (
            expected_round_id
        ):
            raise RuntimeError(
                "Federated round IDs are not "
                "contiguous and zero-based"
            )

        local_training = round_result[
            "local_training"
        ]

        if set(local_training) != set(
            expected_clients
        ):
            raise RuntimeError(
                "Local training client coverage "
                "does not match the config"
            )

        for client_id in expected_clients:
            client_result = local_training[
                client_id
            ]

            if set(client_result) != {
                "num_examples",
                "average_loss",
            }:
                raise RuntimeError(
                    "Local training result keys do "
                    "not match the required contract: "
                    f"{client_id}"
                )

            if client_result["num_examples"] <= 0:
                raise RuntimeError(
                    "Local training result has no "
                    f"examples: {client_id}"
                )

        evaluation = round_result[
            "global_evaluation"
        ]

        required_evaluation_keys = {
            "enterprises",
            "macro_metrics",
            "weighted_metrics",
        }

        if set(evaluation) != (
            required_evaluation_keys
        ):
            raise RuntimeError(
                "Global evaluation keys do not "
                "match the required contract"
            )

        enterprises = evaluation[
            "enterprises"
        ]

        if set(enterprises) != set(
            expected_clients
        ):
            raise RuntimeError(
                "Global evaluation client coverage "
                "does not match the config"
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
                    "Enterprise evaluation keys do "
                    "not match the required contract: "
                    f"{client_id}"
                )

            if (
                enterprise_result["num_examples"]
                <= 0
            ):
                raise RuntimeError(
                    "Enterprise evaluation has no "
                    f"examples: {client_id}"
                )

            if not enterprise_result["metrics"]:
                raise RuntimeError(
                    "Enterprise metrics are empty: "
                    f"{client_id}"
                )

        if not evaluation["macro_metrics"]:
            raise RuntimeError(
                "Macro metrics must be non-empty"
            )

        if not evaluation["weighted_metrics"]:
            raise RuntimeError(
                "Weighted metrics must be non-empty"
            )


def validate_checkpoint_contract(
    checkpoint,
    expected_seed,
    expected_clients,
    expected_federation,
):
    required_keys = {
        "format_version",
        "completed_round",
        "seed",
        "client_ids",
        "federation_config",
        "global_state",
        "metrics_artifact",
    }

    if set(checkpoint) != required_keys:
        raise RuntimeError(
            "Federated checkpoint keys do not "
            "match the required contract"
        )

    if checkpoint["format_version"] != "3.0":
        raise RuntimeError(
            "Unexpected checkpoint format version"
        )

    if checkpoint["seed"] != expected_seed:
        raise RuntimeError(
            "Checkpoint seed does not match "
            "the requested seed"
        )

    if checkpoint["client_ids"] != (
        expected_clients
    ):
        raise RuntimeError(
            "Checkpoint client order does not "
            "match the config"
        )

    if checkpoint["federation_config"] != (
        expected_federation
    ):
        raise RuntimeError(
            "Checkpoint federation configuration "
            "does not match the config"
        )

    expected_completed_round = (
        expected_federation["num_rounds"] - 1
    )

    if checkpoint["completed_round"] != (
        expected_completed_round
    ):
        raise RuntimeError(
            "Checkpoint completed round does not "
            "match the completed experiment"
        )

    if not checkpoint["global_state"]:
        raise RuntimeError(
            "Checkpoint global state is empty"
        )

    if not checkpoint["metrics_artifact"]:
        raise RuntimeError(
            "Checkpoint metrics artifact is empty"
        )


def main():
    args = parse_args()

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as file:
        base_cfg = yaml.safe_load(file)

    expected_clients = list(
        base_cfg["clients"]
    )

    expected_federation = {
        "num_rounds": base_cfg[
            "federation"
        ]["num_rounds"],
        "local_epochs": base_cfg[
            "federation"
        ]["local_epochs"],
        "batch_per_epoch": base_cfg[
            "federation"
        ]["batch_per_epoch"],
    }

    expected_checkpoint_federation = {
        "num_rounds": base_cfg[
            "federation"
        ]["num_rounds"],
        "local_epochs": base_cfg[
            "federation"
        ]["local_epochs"],
        "batch_per_epoch": base_cfg[
            "federation"
        ]["batch_per_epoch"],
        "batch_size": base_cfg[
            "train"
        ]["batch_size"],
        "evaluation_split": base_cfg[
            "evaluation"
        ]["split"],
}

    expected_evaluation_split = base_cfg[
        "evaluation"
    ]["split"]

    base_output = Path(
        os.path.expanduser(
            base_cfg["output_dir"]
        )
    )

    parent_output = base_output.parent

    first_output = (
        parent_output
        / "federated_repro_run_1"
    )

    second_output = (
        parent_output
        / "federated_repro_run_2"
    )

    generated_config_dir = (
        parent_output
        / "reproducibility_configs"
    )

    generated_config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    first_checkpoint = (
        first_output
        / "latest.pth"
    )

    second_checkpoint = (
        second_output
        / "latest.pth"
    )

    first_cfg = copy.deepcopy(base_cfg)
    second_cfg = copy.deepcopy(base_cfg)

    first_cfg["output_dir"] = str(
        first_output
    )

    second_cfg["output_dir"] = str(
        second_output
    )

    first_cfg[
        "checkpointing"
    ]["checkpoint_path"] = str(
        first_checkpoint
    )

    second_cfg[
        "checkpointing"
    ]["checkpoint_path"] = str(
        second_checkpoint
    )

    first_cfg_path = (
        generated_config_dir
        / "federated_run_1.yaml"
    )

    second_cfg_path = (
        generated_config_dir
        / "federated_run_2.yaml"
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
    print(
        "Running federated reproducibility trial 1"
    )

    run_command(
        [
            sys.executable,
            "script/run_federated_training.py",
            "-c",
            str(first_cfg_path),
            "--seed",
            str(args.seed),
        ]
    )

    print()
    print(
        "Running federated reproducibility trial 2"
    )

    run_command(
        [
            sys.executable,
            "script/run_federated_training.py",
            "-c",
            str(second_cfg_path),
            "--seed",
            str(args.seed),
        ]
    )

    first_artifact_path = (
        first_output
        / "round_metrics.json"
    )

    second_artifact_path = (
        second_output
        / "round_metrics.json"
    )

    if not first_artifact_path.is_file():
        raise RuntimeError(
            "First metrics artifact does not exist"
        )

    if not second_artifact_path.is_file():
        raise RuntimeError(
            "Second metrics artifact does not exist"
        )

    if not first_checkpoint.is_file():
        raise RuntimeError(
            "First checkpoint does not exist"
        )

    if not second_checkpoint.is_file():
        raise RuntimeError(
            "Second checkpoint does not exist"
        )

    first_artifact = load_json(
        first_artifact_path
    )

    second_artifact = load_json(
        second_artifact_path
    )

    validate_artifact_contract(
        artifact=first_artifact,
        expected_seed=args.seed,
        expected_clients=expected_clients,
        expected_federation=expected_federation,
        expected_evaluation_split=(
            expected_evaluation_split
        ),
    )

    validate_artifact_contract(
        artifact=second_artifact,
        expected_seed=args.seed,
        expected_clients=expected_clients,
        expected_federation=expected_federation,
        expected_evaluation_split=(
            expected_evaluation_split
        ),
    )

    first_checkpoint_data = torch.load(
        first_checkpoint,
        map_location="cpu",
    )

    second_checkpoint_data = torch.load(
        second_checkpoint,
        map_location="cpu",
    )

    validate_checkpoint_contract(
        checkpoint=first_checkpoint_data,
        expected_seed=args.seed,
        expected_clients=expected_clients,
        expected_federation=(
            expected_checkpoint_federation
        ),
    )

    validate_checkpoint_contract(
        checkpoint=second_checkpoint_data,
        expected_seed=args.seed,
        expected_clients=expected_clients,
        expected_federation=(
            expected_checkpoint_federation
        ),
    )

    first_hash = file_sha256(
        first_artifact_path
    )

    second_hash = file_sha256(
        second_artifact_path
    )

    artifacts_equal = (
        first_artifact == second_artifact
    )

    files_equal = (
        first_hash == second_hash
    )

    final_states_equal = states_equal(
        first_checkpoint_data["global_state"],
        second_checkpoint_data["global_state"],
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
    print(
        "Final global model tensors: "
        + (
            "IDENTICAL"
            if final_states_equal
            else "DIFFERENT"
        )
    )

    if not artifacts_equal:
        raise RuntimeError(
            "Federated JSON artifacts differ "
            "between identical executions"
        )

    if not files_equal:
        raise RuntimeError(
            "Federated serialized artifacts differ "
            "between identical executions"
        )

    if not final_states_equal:
        raise RuntimeError(
            "Final global model states differ "
            "between identical executions"
        )

    first_rounds = first_artifact[
        "rounds"
    ]

    second_rounds = second_artifact[
        "rounds"
    ]

    for round_index, (
        first_round,
        second_round,
    ) in enumerate(
        zip(
            first_rounds,
            second_rounds,
        )
    ):
        if (
            first_round["local_training"]
            != second_round["local_training"]
        ):
            raise RuntimeError(
                "Per-client local training differs "
                f"at round {round_index}"
            )

        if (
            first_round["global_evaluation"]
            != second_round["global_evaluation"]
        ):
            raise RuntimeError(
                "Global evaluation differs "
                f"at round {round_index}"
            )

    if (
        first_checkpoint_data["metrics_artifact"]
        != first_artifact
    ):
        raise RuntimeError(
            "First checkpoint metrics artifact "
            "does not match the serialized artifact"
        )

    if (
        second_checkpoint_data["metrics_artifact"]
        != second_artifact
    ):
        raise RuntimeError(
            "Second checkpoint metrics artifact "
            "does not match the serialized artifact"
        )

    print()
    print("Artifact contract: PASS")
    print("Checkpoint contract: PASS")
    print("Client coverage: PASS")
    print("Federation configuration: PASS")
    print("Per-client local losses: IDENTICAL")
    print("Per-enterprise metrics: IDENTICAL")
    print("Macro metrics: IDENTICAL")
    print("Weighted metrics: IDENTICAL")
    print("JSON artifacts: IDENTICAL")
    print("Serialized artifacts: IDENTICAL")
    print("Final global model tensors: IDENTICAL")
    print("Checkpoint metrics artifacts: CONSISTENT")
    print()
    print(
        "Federated reproducibility "
        "validation: PASS"
    )


if __name__ == "__main__":
    main()
