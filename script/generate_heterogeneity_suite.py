import argparse
import json
import os
import shutil
import sys
from dataclasses import replace
from pathlib import Path


sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)


from fedrelshield.data.config import (
    enterprise_config_metadata,
    load_enterprise_config,
)
from fedrelshield.data.writer import (
    EnterpriseDatasetWriter,
)
from script.generate_enterprise import (
    generate_enterprise,
)


DEFAULT_ENTERPRISE_CONFIGS = (
    "config/fedrelshield/enterprise_a.yaml",
    "config/fedrelshield/enterprise_b.yaml",
    "config/fedrelshield/enterprise_c.yaml",
    "config/fedrelshield/enterprise_d.yaml",
    "config/fedrelshield/enterprise_e.yaml",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deterministic FedRelShield "
            "heterogeneity benchmark suite."
        )
    )

    parser.add_argument(
        "--configs",
        nargs="+",
        default=list(DEFAULT_ENTERPRISE_CONFIGS),
        help=(
            "Enterprise generation YAML configurations "
            "included in the suite."
        ),
    )

    parser.add_argument(
        "--output-root",
        default=(
            "output/fedrelshield/"
            "heterogeneity_suite"
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Delete an existing suite output directory "
            "before generation."
        ),
    )

    return parser.parse_args()


def load_configs(
    config_paths,
):
    configs = []

    seen_enterprise_ids = set()

    for config_path in config_paths:
        config = load_enterprise_config(
            config_path
        )

        if config.enterprise_id in seen_enterprise_ids:
            raise RuntimeError(
                "Duplicate enterprise configuration: "
                f"{config.enterprise_id}"
            )

        seen_enterprise_ids.add(
            config.enterprise_id
        )

        configs.append(
            (
                str(config_path),
                config,
            )
        )

    if not configs:
        raise RuntimeError(
            "Heterogeneity suite requires at least "
            "one enterprise configuration"
        )

    return configs


def prepare_output_root(
    output_root,
    force,
):
    output_root = Path(
        os.path.expanduser(output_root)
    )

    if output_root.exists():
        if not force:
            raise RuntimeError(
                "Heterogeneity suite output already "
                "exists. Use --force to replace it: "
                f"{output_root}"
            )

        shutil.rmtree(
            output_root
        )

    output_root.mkdir(
        parents=True,
        exist_ok=False,
    )

    return output_root


def generate_suite_enterprise(
    source_config_path,
    source_config,
    output_root,
):
    suite_config = replace(
        source_config,
        output_root=str(output_root),
    )

    topology, dataset = generate_enterprise(
        suite_config
    )

    enterprise_output = (
        output_root
        / suite_config.enterprise_id
    )

    writer = EnterpriseDatasetWriter()

    hashes = writer.write(
        output_dir=enterprise_output,
        enterprise_id=(
            suite_config.enterprise_id
        ),
        dataset=dataset,
        generation_metadata=(
            enterprise_config_metadata(
                suite_config
            )
        ),
    )

    return {
        "enterprise_id": (
            suite_config.enterprise_id
        ),
        "profile": suite_config.profile,
        "schema": suite_config.schema,
        "schema_version": (
            suite_config.schema_version
        ),
        "source_config": source_config_path,
        "output_dir": str(
            enterprise_output
        ),
        "num_entities": len(
            topology.entities
        ),
        "num_structural_edges": len(
            topology.edges
        ),
        "num_train_triples": len(
            dataset.train_triples
        ),
        "num_valid_triples": len(
            dataset.valid_triples
        ),
        "num_test_triples": len(
            dataset.test_triples
        ),
        "num_campaigns": len(
            dataset.campaigns
        ),
        "num_provenance_records": len(
            dataset.provenance
        ),
        "artifact_hashes": hashes,
    }


def write_json(
    artifact,
    path,
):
    temporary_path = Path(
        str(path) + ".tmp"
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


def main():
    args = parse_args()

    configs = load_configs(
        args.configs
    )

    output_root = prepare_output_root(
        output_root=args.output_root,
        force=args.force,
    )

    enterprise_results = {}

    print()
    print("Generating heterogeneity suite")
    print("------------------------------")

    for (
        source_config_path,
        source_config,
    ) in configs:
        enterprise_id = (
            source_config.enterprise_id
        )

        print()
        print(
            f"Generating {enterprise_id}"
        )

        result = generate_suite_enterprise(
            source_config_path=(
                source_config_path
            ),
            source_config=source_config,
            output_root=output_root,
        )

        enterprise_results[
            enterprise_id
        ] = result

        print(
            "  profile: "
            f"{result['profile']}"
        )

        print(
            "  entities: "
            f"{result['num_entities']}"
        )

        print(
            "  structural edges: "
            f"{result['num_structural_edges']}"
        )

        print(
            "  train triples: "
            f"{result['num_train_triples']}"
        )

        print(
            "  valid triples: "
            f"{result['num_valid_triples']}"
        )

        print(
            "  test triples: "
            f"{result['num_test_triples']}"
        )

        print(
            "  campaigns: "
            f"{result['num_campaigns']}"
        )

    suite_artifact = {
        "format_version": "1.0",
        "enterprise_order": [
            config.enterprise_id
            for _, config in configs
        ],
        "enterprises": (
            enterprise_results
        ),
    }

    artifact_path = (
        output_root
        / "heterogeneity_suite.json"
    )

    write_json(
        artifact=suite_artifact,
        path=artifact_path,
    )

    print()
    print(
        "Heterogeneity suite generation: PASS"
    )

    print(
        "Enterprises: "
        f"{len(enterprise_results)}"
    )

    print(
        "Artifact: "
        f"{artifact_path}"
    )


if __name__ == "__main__":
    main()
