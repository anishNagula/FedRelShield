import argparse
import json
import os
import sys

import torch
import yaml
from easydict import EasyDict


sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)


from fedrelshield.experiments import (
    BaselineRunConfig,
    BaselineRunner,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a standardized FedRelShield "
            "baseline experiment."
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


def configure_deterministic_execution():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    torch.use_deterministic_algorithms(True)


def load_config(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return EasyDict(
            yaml.safe_load(file)
        )


def validate_result_artifact(
    artifact,
):
    required_keys = {
        "format_version",
        "baseline",
        "seed",
        "evaluation_split",
        "clients",
        "evaluation",
    }

    if not required_keys.issubset(
        artifact
    ):
        raise RuntimeError(
            "Baseline artifact is missing "
            "required contract keys"
        )

    allowed_keys = set(required_keys)

    if artifact["baseline"] == "local_only":
        allowed_keys.add("local_training")

    if artifact["baseline"] == "centralized":
        allowed_keys.add(
            "centralized_training"
        )

    if set(artifact) != allowed_keys:
        raise RuntimeError(
            "Baseline artifact contains "
            "unexpected contract keys"
        )

    if artifact["format_version"] != "1.0":
        raise RuntimeError(
            "Unexpected baseline artifact "
            "format version"
        )

    if not artifact["clients"]:
        raise RuntimeError(
            "Baseline artifact has no clients"
        )

    evaluation_result = artifact[
        "evaluation"
    ]

    if set(
        evaluation_result["enterprises"]
    ) != set(
        artifact["clients"]
    ):
        raise RuntimeError(
            "Baseline evaluation client "
            "coverage mismatch"
        )

    for aggregate_name in (
        "macro_metrics",
        "weighted_metrics",
    ):
        aggregate_metrics = (
            evaluation_result[aggregate_name]
        )

        if "mrr" not in aggregate_metrics:
            raise RuntimeError(
                f"{aggregate_name} is missing mrr"
            )


def write_result_artifact(
    artifact,
    path,
):
    temporary_path = path + ".tmp"

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


def main():
    args = parse_args()

    configure_deterministic_execution()

    torch.manual_seed(args.seed)

    cfg = load_config(args.config)

    device = torch.device("cpu")

    output_dir = os.path.expanduser(
        cfg.output_dir
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    run_config = BaselineRunConfig(
        baseline_name=cfg.baseline.name,
        seed=args.seed,
        evaluation_split=(
            cfg.evaluation.split
        ),
        client_ids=tuple(cfg.clients),
    )

    runner = BaselineRunner(
        cfg=cfg,
        run_config=run_config,
        device=device,
    )

    artifact = runner.run()

    validate_result_artifact(
        artifact
    )

    result_path = os.path.join(
        output_dir,
        "baseline_metrics.json",
    )

    write_result_artifact(
        artifact=artifact,
        path=result_path,
    )

    evaluation_result = artifact[
        "evaluation"
    ]

    print()
    print(
        "Baseline: "
        f"{artifact['baseline']}"
    )

    for enterprise_id in artifact["clients"]:
        metrics = evaluation_result[
            "enterprises"
        ][enterprise_id]["metrics"]

        print(
            f"  {enterprise_id}: "
            f"MRR={metrics['mrr']:.6f}"
        )

    print(
        "  macro MRR: "
        f"{evaluation_result['macro_metrics']['mrr']:.6f}"
    )

    print(
        "  weighted MRR: "
        f"{evaluation_result['weighted_metrics']['mrr']:.6f}"
    )

    print()
    print("Baseline artifact validation: PASS")
    print("Baseline execution: PASS")
    print(f"Metrics: {result_path}")


if __name__ == "__main__":
    main()

