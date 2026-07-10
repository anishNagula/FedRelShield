import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import math

import yaml


METHOD_ORDER = (
    "zero_shot",
    "local_only",
    "centralized",
    "fedavg",
    "relation_aware",
)

DEFAULT_CONFIGS = {
    "zero_shot": (
        "config/fedrelshield/"
        "baseline_zero_shot.yaml"
    ),
    "local_only": (
        "config/fedrelshield/"
        "baseline_local_only.yaml"
    ),
    "centralized": (
        "config/fedrelshield/"
        "baseline_centralized.yaml"
    ),
    "fedavg": (
        "config/fedrelshield/"
        "federated_training.yaml"
    ),
    "relation_aware": (
        "config/fedrelshield/"
        "federated_relation_aware.yaml"
    ),
}

FEDERATED_METHODS = {
    "fedavg",
    "relation_aware",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the FedRelShield baseline suite "
            "across multiple random seeds."
        )
    )

    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096],
    )

    parser.add_argument(
        "--output-dir",
        default=(
            "output/fedrelshield/"
            "baseline_suite"
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
            "Baseline suite requires at least "
            "one seed"
        )

    if len(seeds) != len(set(seeds)):
        raise RuntimeError(
            "Baseline suite seeds must be unique"
        )


def validate_source_configs(configs):
    reference_clients = None
    reference_split = None

    for method in METHOD_ORDER:
        cfg = configs[method]

        clients = list(cfg["clients"])

        evaluation_split = cfg[
            "evaluation"
        ]["split"]

        if reference_clients is None:
            reference_clients = clients
            reference_split = evaluation_split
            continue

        if clients != reference_clients:
            raise RuntimeError(
                "Baseline suite source configs "
                "use different client orders"
            )

        if evaluation_split != reference_split:
            raise RuntimeError(
                "Baseline suite source configs "
                "use different evaluation splits"
            )

    return (
        reference_clients,
        reference_split,
    )


def expected_result_path(
    method,
    output_dir,
):
    if method in FEDERATED_METHODS:
        return (
            output_dir
            / "round_metrics.json"
        )

    return (
        output_dir
        / "baseline_metrics.json"
    )


def extract_evaluation(
    method,
    artifact,
):
    if method in FEDERATED_METHODS:
        rounds = artifact["rounds"]

        if not rounds:
            raise RuntimeError(
                f"{method} artifact has no rounds"
            )

        return rounds[-1][
            "global_evaluation"
        ]

    return artifact["evaluation"]


def validate_run_artifact(
    method,
    artifact,
    seed,
    expected_clients,
    expected_split,
):
    if artifact["seed"] != seed:
        raise RuntimeError(
            f"{method} artifact seed mismatch"
        )

    if artifact["clients"] != expected_clients:
        raise RuntimeError(
            f"{method} artifact client order "
            "mismatch"
        )

    if (
        artifact["evaluation_split"]
        != expected_split
    ):
        raise RuntimeError(
            f"{method} evaluation split mismatch"
        )

    if method in FEDERATED_METHODS:
        if artifact["format_version"] != "1.0":
            raise RuntimeError(
                f"Unexpected {method} metrics "
                "format version"
            )
    
        if not artifact["rounds"]:
            raise RuntimeError(
                f"{method} metrics artifact "
                "contains no rounds"
        )
    else:
        if artifact["format_version"] != "1.0":
            raise RuntimeError(
                f"Unexpected {method} artifact "
                "format version"
            )

        if artifact["baseline"] != method:
            raise RuntimeError(
                f"{method} baseline name mismatch"
            )

    evaluation = extract_evaluation(
        method,
        artifact,
    )

    enterprises = evaluation["enterprises"]

    if set(enterprises) != set(expected_clients):
        raise RuntimeError(
            f"{method} enterprise coverage "
            "mismatch"
        )

    if not evaluation["macro_metrics"]:
        raise RuntimeError(
            f"{method} macro metrics are empty"
        )

    if not evaluation["weighted_metrics"]:
        raise RuntimeError(
            f"{method} weighted metrics are empty"
        )


def build_run_config(
    method,
    source_cfg,
    output_dir,
):
    cfg = copy.deepcopy(source_cfg)

    cfg["output_dir"] = str(output_dir)

    if method in FEDERATED_METHODS:
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


def aggregate_suite_runs(
    suite_runs,
    seeds,
    expected_clients,
):
    aggregates = {}

    for method in METHOD_ORDER:
        method_runs = suite_runs[method]

        expected_seed_keys = {
            str(seed)
            for seed in seeds
        }

        if set(method_runs) != expected_seed_keys:
            raise RuntimeError(
                f"{method} seed coverage mismatch "
                "during aggregation"
            )

        first_run = method_runs[
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
                f"{method} macro and weighted "
                "metric names differ"
            )

        macro_metrics = {}
        weighted_metrics = {}

        for metric_name in macro_metric_names:
            macro_values = [
                method_runs[str(seed)][
                    "macro_metrics"
                ][metric_name]
                for seed in seeds
            ]

            weighted_values = [
                method_runs[str(seed)][
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
            enterprise_metrics[
                enterprise_id
            ] = {}

            reference_num_examples = None

            for seed in seeds:
                enterprise_result = (
                    method_runs[str(seed)][
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
                        f"{method} example count "
                        "changed across seeds for "
                        f"{enterprise_id}"
                    )

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

            for metric_name in (
                enterprise_metric_names
            ):
                values = [
                    method_runs[str(seed)][
                        "enterprises"
                    ][enterprise_id][
                        "metrics"
                    ][metric_name]
                    for seed in seeds
                ]

                summarized_metrics[
                    metric_name
                ] = summarize_values(values)

            enterprise_metrics[
                enterprise_id
            ] = {
                "num_examples": (
                    reference_num_examples
                ),
                "metrics": summarized_metrics,
            }

        aggregates[method] = {
            "macro_metrics": macro_metrics,
            "weighted_metrics": weighted_metrics,
            "enterprises": enterprise_metrics,
        }

    return aggregates


def main():
    args = parse_args()

    validate_seeds(args.seeds)

    suite_output = Path(
        os.path.expanduser(
            args.output_dir
        )
    )
    
    if args.aggregate_only:
        result_path = (
            suite_output
            / "baseline_suite.json"
        )

        if not result_path.is_file():
            raise RuntimeError(
                "Cannot aggregate: baseline suite "
                "artifact does not exist"
            )

        existing_artifact = load_json(
            result_path
        )

        suite_runs = existing_artifact[
            "methods"
        ]

        expected_clients = existing_artifact[
            "clients"
        ]

        existing_seeds = existing_artifact[
            "seeds"
        ]

        aggregates = aggregate_suite_runs(
            suite_runs=suite_runs,
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
        print("Multi-seed aggregation: PASS")
        print(f"Artifact: {result_path}")

        return

    configs = {
        method: load_yaml(path)
        for method, path
        in DEFAULT_CONFIGS.items()
    }

    (
        expected_clients,
        expected_split,
    ) = validate_source_configs(configs)

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

    suite_runs = {
        method: {}
        for method in METHOD_ORDER
    }

    for seed in args.seeds:
        print()
        print("=" * 60)
        print(f"SEED {seed}")
        print("=" * 60)

        for method in METHOD_ORDER:
            print()
            print(
                f"Running {method}: "
                f"seed={seed}"
            )

            run_output = (
                runs_dir
                / method
                / f"seed_{seed}"
            )

            run_output.mkdir(
                parents=True,
                exist_ok=True,
            )

            run_cfg = build_run_config(
                method=method,
                source_cfg=configs[method],
                output_dir=run_output,
            )

            config_path = (
                config_dir
                / (
                    f"{method}_"
                    f"seed_{seed}.yaml"
                )
            )

            write_yaml(
                run_cfg,
                config_path,
            )

            if method in FEDERATED_METHODS:
                command = [
                    sys.executable,
                    "script/"
                    "run_federated_training.py",
                    "-c",
                    str(config_path),
                    "--seed",
                    str(seed),
                ]
            else:
                command = [
                    sys.executable,
                    "script/run_baseline.py",
                    "-c",
                    str(config_path),
                    "--seed",
                    str(seed),
                ]

            run_command(command)

            artifact_path = (
                expected_result_path(
                    method,
                    run_output,
                )
            )

            if not artifact_path.is_file():
                raise RuntimeError(
                    f"{method} did not produce "
                    f"its expected artifact: "
                    f"{artifact_path}"
                )

            artifact = load_json(
                artifact_path
            )

            validate_run_artifact(
                method=method,
                artifact=artifact,
                seed=seed,
                expected_clients=(
                    expected_clients
                ),
                expected_split=(
                    expected_split
                ),
            )

            evaluation = extract_evaluation(
                method,
                artifact,
            )

            suite_runs[method][
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
            
    aggregates = aggregate_suite_runs(
        suite_runs=suite_runs,
        seeds=args.seeds,
        expected_clients=expected_clients,
    )

    artifact = {
        "format_version": "1.0",
        "seeds": list(args.seeds),
        "clients": expected_clients,
        "evaluation_split": expected_split,
        "methods": suite_runs,
        "aggregates": aggregates,
    }

    result_path = (
        suite_output
        / "baseline_suite.json"
    )

    write_json(
        artifact,
        result_path,
    )

    print()
    print("Baseline suite execution: PASS")
    print(
        "Runs completed: "
        f"{len(args.seeds) * len(METHOD_ORDER)}"
    )
    print(f"Artifact: {result_path}")
    
    print()
    print("Multi-seed aggregate results")
    print("----------------------------")
    print(
        f"{'Method':<16}"
        f"{'Macro MRR':>22}"
        f"{'Weighted MRR':>22}"
    )
    print("-" * 60)

    for method in METHOD_ORDER:
        macro = aggregates[
            method
        ]["macro_metrics"]["mrr"]

        weighted = aggregates[
            method
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
            f"{method:<16}"
            f"{macro_text:>22}"
            f"{weighted_text:>22}"
        )


if __name__ == "__main__":
    main()
