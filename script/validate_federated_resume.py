import argparse
import json
import os
import shutil
import subprocess
import sys

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate FedRelShield federated "
            "checkpoint and resume support."
        )
    )

    parser.add_argument(
        "-c",
        "--config",
        required=True,
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


def main():
    args = parse_args()

    import yaml

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as file:
        cfg = yaml.safe_load(file)

    output_dir = os.path.expanduser(
        cfg["output_dir"]
    )

    checkpoint_path = os.path.expanduser(
        cfg["checkpointing"]["checkpoint_path"]
    )

    metrics_path = os.path.join(
        output_dir,
        "round_metrics.json",
    )

    shutil.rmtree(
        output_dir,
        ignore_errors=True,
    )

    run_command(
        [
            sys.executable,
            "script/run_federated_training.py",
            "-c",
            args.config,
            "--stop-after-round",
            "1",
        ]
    )

    if not os.path.isfile(checkpoint_path):
        raise RuntimeError(
            "Checkpoint was not created"
        )

    checkpoint_before = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if checkpoint_before["completed_round"] != 1:
        raise RuntimeError(
            "Interrupted checkpoint round is not 1"
        )

    metrics_before = load_json(
        metrics_path
    )

    if [
        item["round_id"]
        for item in metrics_before["rounds"]
    ] != [0, 1]:
        raise RuntimeError(
            "Interrupted metrics history is invalid"
        )

    checkpoint_metrics_before = (
        checkpoint_before["metrics_artifact"]
    )

    if (
        checkpoint_metrics_before
        != metrics_before
    ):
        raise RuntimeError(
            "Checkpoint metrics and JSON metrics differ"
        )

    saved_global_state = {
        name: tensor.clone()
        for name, tensor
        in checkpoint_before[
            "global_state"
        ].items()
    }

    run_command(
        [
            sys.executable,
            "script/run_federated_training.py",
            "-c",
            args.config,
            "--resume",
            checkpoint_path,
        ]
    )

    checkpoint_after = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    if checkpoint_after["completed_round"] != 3:
        raise RuntimeError(
            "Final checkpoint round is not 3"
        )

    metrics_after = load_json(
        metrics_path
    )

    round_ids = [
        item["round_id"]
        for item in metrics_after["rounds"]
    ]

    if round_ids != [0, 1, 2, 3]:
        raise RuntimeError(
            "Final metrics history must contain "
            "rounds 0, 1, 2, 3 exactly once"
        )

    if (
        metrics_after["rounds"][:2]
        != metrics_before["rounds"]
    ):
        raise RuntimeError(
            "Resume changed pre-resume metrics history"
        )

    if (
        checkpoint_after["metrics_artifact"]
        != metrics_after
    ):
        raise RuntimeError(
            "Final checkpoint metrics and JSON differ"
        )

    if all(
        torch.equal(
            saved_global_state[name],
            checkpoint_after[
                "global_state"
            ][name],
        )
        for name in saved_global_state
    ):
        raise RuntimeError(
            "Global state did not change after resume"
        )

    print()
    print("Interrupted checkpoint round: PASS")
    print("Interrupted metrics history: PASS")
    print("Checkpoint/metrics consistency: PASS")
    print("Resume continuation: PASS")
    print("Pre-resume history preservation: PASS")
    print("Final round coverage: PASS")
    print("Final global state progression: PASS")
    print()
    print("Federated resume validation: PASS")


if __name__ == "__main__":
    main()
