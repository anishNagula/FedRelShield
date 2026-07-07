import argparse
import copy
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
            "Validate and diagnose trajectory equivalence "
            "between uninterrupted and resumed "
            "federated ULTRA training."
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

    env = os.environ.copy()

    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )

    subprocess.run(
        command,
        check=True,
        env=env,
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


def state_difference_summary(
    state_a,
    state_b,
):
    if state_a.keys() != state_b.keys():
        return {
            "keys_equal": False,
        }

    differing_tensors = 0
    max_absolute_difference = 0.0
    first_differing_tensor = None

    for name in state_a:
        tensor_a = state_a[name]
        tensor_b = state_b[name]

        if not torch.equal(
            tensor_a,
            tensor_b,
        ):
            differing_tensors += 1

            if first_differing_tensor is None:
                first_differing_tensor = name

            if (
                tensor_a.is_floating_point()
                or tensor_a.is_complex()
            ):
                difference = (
                    tensor_a - tensor_b
                ).abs().max().item()

                max_absolute_difference = max(
                    max_absolute_difference,
                    difference,
                )

    return {
        "keys_equal": True,
        "differing_tensors":
            differing_tensors,
        "max_absolute_difference":
            max_absolute_difference,
        "first_differing_tensor":
            first_differing_tensor,
    }


def compare_round_prefix(
    uninterrupted_metrics,
    interrupted_metrics,
):
    interrupted_rounds = interrupted_metrics[
        "rounds"
    ]

    uninterrupted_prefix = (
        uninterrupted_metrics["rounds"][
            :len(interrupted_rounds)
        ]
    )

    return (
        uninterrupted_prefix
        == interrupted_rounds
    )


def main():
    args = parse_args()

    with open(
        args.config,
        "r",
        encoding="utf-8",
    ) as file:
        base_cfg = yaml.safe_load(file)

    base_output = Path(
        os.path.expanduser(
            base_cfg["output_dir"]
        )
    )

    parent_output = base_output.parent

    reference_output = (
        parent_output
        / "equivalence_reference_prefix"
    )

    uninterrupted_output = (
        parent_output
        / "equivalence_uninterrupted"
    )

    resumed_output = (
        parent_output
        / "equivalence_resumed"
    )

    generated_config_dir = (
        parent_output
        / "equivalence_configs"
    )

    generated_config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_checkpoint = (
        reference_output
        / "latest.pth"
    )

    uninterrupted_checkpoint = (
        uninterrupted_output
        / "latest.pth"
    )

    resumed_checkpoint = (
        resumed_output
        / "latest.pth"
    )

    reference_cfg = copy.deepcopy(
        base_cfg
    )

    reference_cfg["output_dir"] = str(
        reference_output
    )

    reference_cfg[
        "checkpointing"
    ]["checkpoint_path"] = str(
        reference_checkpoint
    )

    uninterrupted_cfg = copy.deepcopy(
        base_cfg
    )

    uninterrupted_cfg["output_dir"] = str(
        uninterrupted_output
    )

    uninterrupted_cfg[
        "checkpointing"
    ]["checkpoint_path"] = str(
        uninterrupted_checkpoint
    )

    resumed_cfg = copy.deepcopy(
        base_cfg
    )

    resumed_cfg["output_dir"] = str(
        resumed_output
    )

    resumed_cfg[
        "checkpointing"
    ]["checkpoint_path"] = str(
        resumed_checkpoint
    )

    reference_cfg_path = (
        generated_config_dir
        / "reference_prefix.yaml"
    )

    uninterrupted_cfg_path = (
        generated_config_dir
        / "uninterrupted.yaml"
    )

    resumed_cfg_path = (
        generated_config_dir
        / "resumed.yaml"
    )

    write_yaml(
        reference_cfg,
        reference_cfg_path,
    )

    write_yaml(
        uninterrupted_cfg,
        uninterrupted_cfg_path,
    )

    write_yaml(
        resumed_cfg,
        resumed_cfg_path,
    )

    for output_dir in (
        reference_output,
        uninterrupted_output,
        resumed_output,
    ):
        shutil.rmtree(
            output_dir,
            ignore_errors=True,
        )

    print()
    print("Running reference prefix through round 1")

    run_command(
        [
            sys.executable,
            "script/run_federated_training.py",
            "-c",
            str(reference_cfg_path),
            "--seed",
            str(args.seed),
            "--stop-after-round",
            "1",
        ]
    )

    reference_prefix_checkpoint = torch.load(
        reference_checkpoint,
        map_location="cpu",
    )

    reference_prefix_metrics = load_json(
        reference_output
        / "round_metrics.json"
    )

    print()
    print("Running uninterrupted trajectory")

    run_command(
        [
            sys.executable,
            "script/run_federated_training.py",
            "-c",
            str(uninterrupted_cfg_path),
            "--seed",
            str(args.seed),
        ]
    )

    uninterrupted_metrics = load_json(
        uninterrupted_output
        / "round_metrics.json"
    )

    print()
    print("Running interrupted trajectory")

    run_command(
        [
            sys.executable,
            "script/run_federated_training.py",
            "-c",
            str(resumed_cfg_path),
            "--seed",
            str(args.seed),
            "--stop-after-round",
            "1",
        ]
    )

    interrupted_checkpoint = torch.load(
        resumed_checkpoint,
        map_location="cpu",
    )

    interrupted_metrics = load_json(
        resumed_output
        / "round_metrics.json"
    )

    if (
        reference_prefix_checkpoint[
            "completed_round"
        ]
        != 1
    ):
        raise RuntimeError(
            "Reference prefix did not stop "
            "after round 1"
        )

    if (
        interrupted_checkpoint[
            "completed_round"
        ]
        != 1
    ):
        raise RuntimeError(
            "Interrupted trajectory did not stop "
            "after round 1"
        )

    print()
    print("Pre-resume boundary diagnostics")
    print("-------------------------------")

    reference_interrupted_state_equal = (
        states_equal(
            reference_prefix_checkpoint[
                "global_state"
            ],
            interrupted_checkpoint[
                "global_state"
            ],
        )
    )

    print(
        "Reference-prefix vs interrupted "
        "checkpoint state: "
        + (
            "IDENTICAL"
            if reference_interrupted_state_equal
            else "DIFFERENT"
        )
    )

    if not reference_interrupted_state_equal:
        print(
            state_difference_summary(
                reference_prefix_checkpoint[
                    "global_state"
                ],
                interrupted_checkpoint[
                    "global_state"
                ],
            )
        )

    reference_interrupted_metrics_equal = (
        reference_prefix_metrics
        == interrupted_metrics
    )

    print(
        "Reference-prefix vs interrupted "
        "metrics: "
        + (
            "IDENTICAL"
            if reference_interrupted_metrics_equal
            else "DIFFERENT"
        )
    )

    uninterrupted_prefix_equal = (
        compare_round_prefix(
            uninterrupted_metrics,
            interrupted_metrics,
        )
    )

    print(
        "Uninterrupted prefix vs interrupted "
        "metrics: "
        + (
            "IDENTICAL"
            if uninterrupted_prefix_equal
            else "DIFFERENT"
        )
    )

    if not (
        reference_interrupted_state_equal
        and reference_interrupted_metrics_equal
        and uninterrupted_prefix_equal
    ):
        raise RuntimeError(
            "Trajectories already differ before resume"
        )

    print()
    print("Pre-resume trajectory boundary: PASS")

    print()
    print("Resuming interrupted trajectory")

    run_command(
        [
            sys.executable,
            "script/run_federated_training.py",
            "-c",
            str(resumed_cfg_path),
            "--seed",
            str(args.seed),
            "--resume",
            str(resumed_checkpoint),
        ]
    )

    resumed_metrics = load_json(
        resumed_output
        / "round_metrics.json"
    )

    uninterrupted_final = torch.load(
        uninterrupted_checkpoint,
        map_location="cpu",
    )

    resumed_final = torch.load(
        resumed_checkpoint,
        map_location="cpu",
    )

    print()
    print("Post-resume diagnostics")
    print("-----------------------")

    uninterrupted_rounds = (
        uninterrupted_metrics["rounds"]
    )

    resumed_rounds = (
        resumed_metrics["rounds"]
    )

    first_differing_round = None

    for index, (
        uninterrupted_round,
        resumed_round,
    ) in enumerate(
        zip(
            uninterrupted_rounds,
            resumed_rounds,
        )
    ):
        local_equal = (
            uninterrupted_round[
                "local_training"
            ]
            == resumed_round[
                "local_training"
            ]
        )

        evaluation_equal = (
            uninterrupted_round[
                "global_evaluation"
            ]
            == resumed_round[
                "global_evaluation"
            ]
        )

        print(
            f"Round {index}: "
            f"local_training="
            f"{'IDENTICAL' if local_equal else 'DIFFERENT'}, "
            f"global_evaluation="
            f"{'IDENTICAL' if evaluation_equal else 'DIFFERENT'}"
        )

        if (
            first_differing_round is None
            and not (
                local_equal
                and evaluation_equal
            )
        ):
            first_differing_round = index

    final_state_equal = states_equal(
        uninterrupted_final["global_state"],
        resumed_final["global_state"],
    )

    print(
        "Final global model state: "
        + (
            "IDENTICAL"
            if final_state_equal
            else "DIFFERENT"
        )
    )

    if not final_state_equal:
        print(
            state_difference_summary(
                uninterrupted_final[
                    "global_state"
                ],
                resumed_final[
                    "global_state"
                ],
            )
        )

    if first_differing_round is not None:
        print()
        print(
            "FIRST DIVERGENCE: round "
            f"{first_differing_round}"
        )

        print()
        print("Uninterrupted local training:")
        print(
            json.dumps(
                uninterrupted_rounds[
                    first_differing_round
                ]["local_training"],
                indent=2,
                sort_keys=True,
            )
        )

        print()
        print("Resumed local training:")
        print(
            json.dumps(
                resumed_rounds[
                    first_differing_round
                ]["local_training"],
                indent=2,
                sort_keys=True,
            )
        )

        raise RuntimeError(
            "Post-resume trajectory divergence "
            f"begins at round "
            f"{first_differing_round}"
        )

    if uninterrupted_metrics != resumed_metrics:
        raise RuntimeError(
            "Metrics artifacts differ despite "
            "matching round diagnostics"
        )

    if not final_state_equal:
        raise RuntimeError(
            "Final model states differ despite "
            "matching metrics"
        )

    print()
    print("Interrupted execution: PASS")
    print("Pre-resume trajectory: IDENTICAL")
    print("Deterministic local training: PASS")
    print("Per-round local losses: IDENTICAL")
    print("Per-round evaluation metrics: IDENTICAL")
    print("Metrics artifacts: IDENTICAL")
    print("Final global model tensors: IDENTICAL")
    print()
    print(
        "Federated trajectory equivalence "
        "validation: PASS"
    )


if __name__ == "__main__":
    main()

