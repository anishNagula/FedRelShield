import argparse
import copy
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


DEFAULT_CONFIG = (
    "config/fedrelshield/"
    "federated_relation_aware.yaml"
)

DEFAULT_ALPHAS = (
    0.0,
    0.25,
    0.5,
    0.75,
    1.0,
)

DEFAULT_SEEDS = (
    1024,
    2048,
    4096,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the FedRelShield relation-aware "
            "alpha ablation across multiple seeds."
        )
    )

    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
    )

    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=list(DEFAULT_ALPHAS),
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "output/fedrelshield/"
            "relation_aware_ablation"
        ),
    )

    parser.add_argument(
        "--aggregate-only",
        action="store_true",
    )

    return parser.parse_args()


def load_yaml(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return yaml.safe_load(file)


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


def write_json(artifact, path):
    temporary_path = str(path) + ".tmp"

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


def run_command(command):
    print()
    print("$ " + " ".join(command))

    subprocess.run(
        command,
        check=True,
    )


def validate_seeds(seeds):
    if not seeds:
        raise RuntimeError(
            "Relation-aware ablation requires "
            "at least one seed"
        )

    if len(seeds) != len(set(seeds)):
        raise RuntimeError(
            "Relation-aware ablation seeds "
            "must be unique"
        )

    for seed in seeds:
        if seed < 0:
            raise RuntimeError(
                "Relation-aware ablation seeds "
                "must be non-negative"
            )


def validate_alphas(alphas):
    if not alphas:
        raise RuntimeError(
            "Relation-aware ablation requires "
            "at least one alpha"
        )

    if len(alphas) != len(set(alphas)):
        raise RuntimeError(
            "Relation-aware ablation alphas "
            "must be unique"
        )

    for alpha in alphas:
        if not math.isfinite(alpha):
            raise RuntimeError(
                "Relation-aware ablation alphas "
                "must be finite"
            )

        if not 0.0 <= alpha <= 1.0:
            raise RuntimeError(
                "Relation-aware ablation alphas "
                "must be in [0, 1]"
            )


def alpha_key(alpha):
    return f"{alpha:.2f}"


def alpha_directory_name(alpha):
    return f"alpha_{alpha_key(alpha)}"


def validate_source_config(cfg):
    if not isinstance(cfg, dict):
        raise RuntimeError(
            "Source config must be a mapping"
        )

    try:
        clients = list(cfg["clients"])

        evaluation_split = str(
            cfg["evaluation"]["split"]
        )

        federation = cfg["federation"]

        aggregation_method = str(
            federation["aggregation_method"]
        )

    except KeyError as error:
        raise RuntimeError(
            "Source config is missing required field: "
            f"{error.args[0]}"
        ) from error

    if not clients:
        raise RuntimeError(
            "Source config clients must be non-empty"
        )

    if len(clients) != len(set(clients)):
        raise RuntimeError(
            "Source config contains duplicate clients"
        )

    if not evaluation_split:
        raise RuntimeError(
            "Source config evaluation split "
            "must be non-empty"
        )

    if aggregation_method != "relation_aware":
        raise RuntimeError(
            "Ablation source config must use "
            "relation_aware aggregation"
        )

    if "alpha" not in federation:
        raise RuntimeError(
            "Ablation source config must define "
            "federation.alpha"
        )

    return clients, evaluation_split


def build_run_config(
    source_cfg,
    alpha,
    output_dir,
):
    cfg = copy.deepcopy(source_cfg)

    cfg["output_dir"] = str(output_dir)

    cfg[
        "federation"
    ][
        "aggregation_method"
    ] = "relation_aware"

    cfg[
        "federation"
    ][
        "alpha"
    ] = float(alpha)

    checkpoint_path = (
        output_dir
        / "latest.pth"
    )

    cfg[
        "checkpointing"
    ][
        "checkpoint_path"
    ] = str(checkpoint_path)

    return cfg


def extract_final_evaluation(artifact):
    rounds = artifact["rounds"]

    if not rounds:
        raise RuntimeError(
            "Federated metrics artifact "
            "contains no rounds"
        )

    return rounds[-1][
        "global_evaluation"
    ]


def validate_run_artifact(
    artifact,
    seed,
    expected_clients,
    expected_split,
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
            "Unexpected federated metrics "
            "artifact keys"
        )

    if artifact["format_version"] != "1.0":
        raise RuntimeError(
            "Unexpected federated metrics "
            "format version"
        )

    if artifact["seed"] != seed:
        raise RuntimeError(
            "Federated artifact seed mismatch"
        )

    if artifact["clients"] != expected_clients:
        raise RuntimeError(
            "Federated artifact client order "
            "mismatch"
        )

    if (
        artifact["evaluation_split"]
        != expected_split
    ):
        raise RuntimeError(
            "Federated artifact evaluation "
            "split mismatch"
        )

    if not artifact["rounds"]:
        raise RuntimeError(
            "Federated metrics artifact "
            "contains no rounds"
        )

    evaluation = extract_final_evaluation(
        artifact
    )

    required_evaluation_keys = {
        "enterprises",
        "macro_metrics",
        "weighted_metrics",
    }

    if set(evaluation) != required_evaluation_keys:
        raise RuntimeError(
            "Unexpected evaluation keys"
        )

    enterprises = evaluation["enterprises"]

    if set(enterprises) != set(expected_clients):
        raise RuntimeError(
            "Federated enterprise coverage "
            "mismatch"
        )

    if not evaluation["macro_metrics"]:
        raise RuntimeError(
            "Federated macro metrics are empty"
        )

    if not evaluation["weighted_metrics"]:
        raise RuntimeError(
            "Federated weighted metrics are empty"
        )

    return evaluation


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


def validate_alpha_seed_coverage(
    ablation_runs,
    alphas,
    seeds,
):
    expected_alpha_keys = {
        alpha_key(alpha)
        for alpha in alphas
    }

    if set(ablation_runs) != expected_alpha_keys:
        raise RuntimeError(
            "Alpha coverage mismatch during "
            "aggregation"
        )

    expected_seed_keys = {
        str(seed)
        for seed in seeds
    }

    for alpha in alphas:
        key = alpha_key(alpha)

        if (
            set(ablation_runs[key])
            != expected_seed_keys
        ):
            raise RuntimeError(
                f"Seed coverage mismatch for "
                f"alpha={key}"
            )


def aggregate_alpha_runs(
    ablation_runs,
    alphas,
    seeds,
    expected_clients,
):
    validate_alpha_seed_coverage(
        ablation_runs=ablation_runs,
        alphas=alphas,
        seeds=seeds,
    )

    aggregates = {}

    for alpha in alphas:
        key = alpha_key(alpha)

        alpha_runs = ablation_runs[key]

        first_run = alpha_runs[
            str(seeds[0])
        ]

        macro_metric_names = list(
            first_run["macro_metrics"]
        )

        weighted_metric_names = list(
            first_run["weighted_metrics"]
        )

        if (
            set(macro_metric_names)
            != set(weighted_metric_names)
        ):
            raise RuntimeError(
                f"Alpha={key} macro and weighted "
                "metric names differ"
            )

        macro_metrics = {}
        weighted_metrics = {}

        for metric_name in macro_metric_names:
            macro_values = [
                alpha_runs[str(seed)][
                    "macro_metrics"
                ][metric_name]
                for seed in seeds
            ]

            weighted_values = [
                alpha_runs[str(seed)][
                    "weighted_metrics"
                ][metric_name]
                for seed in seeds
            ]

            macro_metrics[
                metric_name
            ] = summarize_values(
                macro_values
            )

            weighted_metrics[
                metric_name
            ] = summarize_values(
                weighted_values
            )

        enterprise_metrics = {}

        for enterprise_id in expected_clients:
            reference_num_examples = None

            first_enterprise_result = (
                first_run[
                    "enterprises"
                ][enterprise_id]
            )

            enterprise_metric_names = list(
                first_enterprise_result[
                    "metrics"
                ]
            )

            summarized_metrics = {}

            for seed in seeds:
                enterprise_result = (
                    alpha_runs[str(seed)][
                        "enterprises"
                    ][enterprise_id]
                )

                num_examples = (
                    enterprise_result[
                        "num_examples"
                    ]
                )

                if reference_num_examples is None:
                    reference_num_examples = (
                        num_examples
                    )

                elif (
                    num_examples
                    != reference_num_examples
                ):
                    raise RuntimeError(
                        f"Alpha={key} example count "
                        "changed across seeds for "
                        f"{enterprise_id}"
                    )

                if (
                    set(
                        enterprise_result[
                            "metrics"
                        ]
                    )
                    != set(
                        enterprise_metric_names
                    )
                ):
                    raise RuntimeError(
                        f"Alpha={key} metric names "
                        "changed across seeds for "
                        f"{enterprise_id}"
                    )

            for metric_name in (
                enterprise_metric_names
            ):
                values = [
                    alpha_runs[str(seed)][
                        "enterprises"
                    ][enterprise_id][
                        "metrics"
                    ][metric_name]
                    for seed in seeds
                ]

                summarized_metrics[
                    metric_name
                ] = summarize_values(
                    values
                )

            enterprise_metrics[
                enterprise_id
            ] = {
                "num_examples":
                    reference_num_examples,
                "metrics":
                    summarized_metrics,
            }

        aggregates[key] = {
            "alpha": float(alpha),
            "macro_metrics": macro_metrics,
            "weighted_metrics": weighted_metrics,
            "enterprises": enterprise_metrics,
        }

    return aggregates


def print_aggregate_results(
    aggregates,
    alphas,
):
    print()
    print("Relation-aware alpha ablation results")
    print("-------------------------------------")

    print(
        f"{'Alpha':<10}"
        f"{'Macro MRR':>22}"
        f"{'Weighted MRR':>22}"
    )

    print("-" * 54)

    for alpha in alphas:
        key = alpha_key(alpha)

        macro = aggregates[
            key
        ]["macro_metrics"]["mrr"]

        weighted = aggregates[
            key
        ]["weighted_metrics"]["mrr"]

        macro_text = (
            f"{macro['mean']:.6f} "
            f"+/- {macro['std']:.6f}"
        )

        weighted_text = (
            f"{weighted['mean']:.6f} "
            f"+/- {weighted['std']:.6f}"
        )

        print(
            f"{key:<10}"
            f"{macro_text:>22}"
            f"{weighted_text:>22}"
        )


def main():
    args = parse_args()

    validate_seeds(args.seeds)
    validate_alphas(args.alphas)

    suite_output = Path(
        os.path.expanduser(
            args.output_dir
        )
    )

    result_path = (
        suite_output
        / "relation_aware_ablation.json"
    )

    if args.aggregate_only:
        if not result_path.is_file():
            raise RuntimeError(
                "Cannot aggregate: relation-aware "
                "ablation artifact does not exist"
            )

        existing_artifact = load_json(
            result_path
        )

        existing_alphas = existing_artifact[
            "alphas"
        ]

        existing_seeds = existing_artifact[
            "seeds"
        ]

        expected_clients = existing_artifact[
            "clients"
        ]

        ablation_runs = existing_artifact[
            "runs"
        ]

        aggregates = aggregate_alpha_runs(
            ablation_runs=ablation_runs,
            alphas=existing_alphas,
            seeds=existing_seeds,
            expected_clients=expected_clients,
        )

        existing_artifact[
            "aggregates"
        ] = aggregates

        write_json(
            existing_artifact,
            result_path,
        )

        print()
        print(
            "Relation-aware multi-seed "
            "aggregation: PASS"
        )

        print(f"Artifact: {result_path}")

        print_aggregate_results(
            aggregates=aggregates,
            alphas=existing_alphas,
        )

        return

    source_cfg = load_yaml(
        args.config
    )

    (
        expected_clients,
        expected_split,
    ) = validate_source_config(
        source_cfg
    )

    shutil.rmtree(
        suite_output,
        ignore_errors=True,
    )

    config_dir = (
        suite_output
        / "configs"
    )

    runs_dir = (
        suite_output
        / "runs"
    )

    config_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    runs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ablation_runs = {
        alpha_key(alpha): {}
        for alpha in args.alphas
    }

    completed_runs = 0

    for alpha in args.alphas:
        key = alpha_key(alpha)

        print()
        print("=" * 60)
        print(f"ALPHA {key}")
        print("=" * 60)

        for seed in args.seeds:
            print()
            print(
                "Running relation-aware: "
                f"alpha={key}, "
                f"seed={seed}"
            )

            run_output = (
                runs_dir
                / alpha_directory_name(alpha)
                / f"seed_{seed}"
            )

            run_output.mkdir(
                parents=True,
                exist_ok=True,
            )

            run_cfg = build_run_config(
                source_cfg=source_cfg,
                alpha=alpha,
                output_dir=run_output,
            )

            config_path = (
                config_dir
                / (
                    f"relation_aware_"
                    f"alpha_{key}_"
                    f"seed_{seed}.yaml"
                )
            )

            write_yaml(
                run_cfg,
                config_path,
            )

            command = [
                sys.executable,
                "script/"
                "run_federated_training.py",
                "-c",
                str(config_path),
                "--seed",
                str(seed),
            ]

            run_command(command)

            artifact_path = (
                run_output
                / "round_metrics.json"
            )

            if not artifact_path.is_file():
                raise RuntimeError(
                    "Relation-aware run did not "
                    "produce expected artifact: "
                    f"{artifact_path}"
                )

            artifact = load_json(
                artifact_path
            )

            evaluation = validate_run_artifact(
                artifact=artifact,
                seed=seed,
                expected_clients=(
                    expected_clients
                ),
                expected_split=(
                    expected_split
                ),
            )

            ablation_runs[key][
                str(seed)
            ] = {
                "macro_metrics": (
                    evaluation[
                        "macro_metrics"
                    ]
                ),
                "weighted_metrics": (
                    evaluation[
                        "weighted_metrics"
                    ]
                ),
                "enterprises": (
                    evaluation[
                        "enterprises"
                    ]
                ),
            }

            completed_runs += 1

    aggregates = aggregate_alpha_runs(
        ablation_runs=ablation_runs,
        alphas=args.alphas,
        seeds=args.seeds,
        expected_clients=expected_clients,
    )

    artifact = {
        "format_version": "1.0",
        "source_config": str(args.config),
        "alphas": list(args.alphas),
        "seeds": list(args.seeds),
        "clients": expected_clients,
        "evaluation_split": expected_split,
        "runs": ablation_runs,
        "aggregates": aggregates,
    }

    write_json(
        artifact,
        result_path,
    )

    print()
    print(
        "Relation-aware ablation execution: PASS"
    )

    print(
        "Runs completed: "
        f"{completed_runs}"
    )

    print(f"Artifact: {result_path}")

    print_aggregate_results(
        aggregates=aggregates,
        alphas=args.alphas,
    )


if __name__ == "__main__":
    main()
