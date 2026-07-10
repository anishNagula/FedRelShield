import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)


DATASET_FILES = (
    "campaigns.json",
    "provenance.jsonl",
    "statistics.json",
    "train.txt",
    "valid.txt",
    "test.txt",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate the FedRelShield deterministic "
            "heterogeneity benchmark suite."
        )
    )

    parser.add_argument(
        "--suite-root",
        default=(
            "output/fedrelshield/"
            "heterogeneity_suite"
        ),
    )

    parser.add_argument(
        "--canonical-root",
        default="data/fedrelshield",
    )

    return parser.parse_args()


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def file_sha256(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        while True:
            chunk = file.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def validate_suite_contract(
    artifact,
):
    required_keys = {
        "format_version",
        "enterprise_order",
        "enterprises",
    }

    if set(artifact) != required_keys:
        raise RuntimeError(
            "Heterogeneity suite artifact keys "
            "do not match the required contract"
        )

    if artifact["format_version"] != "1.0":
        raise RuntimeError(
            "Unexpected heterogeneity suite "
            "format version"
        )

    enterprise_order = artifact[
        "enterprise_order"
    ]

    enterprises = artifact[
        "enterprises"
    ]

    if not enterprise_order:
        raise RuntimeError(
            "Heterogeneity suite has no enterprises"
        )

    if len(enterprise_order) != len(
        set(enterprise_order)
    ):
        raise RuntimeError(
            "Heterogeneity suite contains duplicate "
            "enterprise IDs"
        )

    if set(enterprise_order) != set(
        enterprises
    ):
        raise RuntimeError(
            "Enterprise order and enterprise mapping "
            "do not cover the same enterprises"
        )

    required_enterprise_keys = {
        "enterprise_id",
        "profile",
        "schema",
        "schema_version",
        "source_config",
        "output_dir",
        "num_entities",
        "num_structural_edges",
        "num_train_triples",
        "num_valid_triples",
        "num_test_triples",
        "num_campaigns",
        "num_provenance_records",
        "artifact_hashes",
    }

    for enterprise_id in enterprise_order:
        result = enterprises[
            enterprise_id
        ]

        if set(result) != required_enterprise_keys:
            raise RuntimeError(
                "Enterprise suite result keys do "
                "not match contract: "
                f"{enterprise_id}"
            )

        if (
            result["enterprise_id"]
            != enterprise_id
        ):
            raise RuntimeError(
                "Enterprise result ID mismatch: "
                f"{enterprise_id}"
            )

        if set(
            result["artifact_hashes"]
        ) != {
            *DATASET_FILES,
            "manifest.json",
        }:
            raise RuntimeError(
                "Enterprise artifact hash coverage "
                "mismatch: "
                f"{enterprise_id}"
            )

        count_fields = (
            "num_entities",
            "num_structural_edges",
            "num_train_triples",
            "num_valid_triples",
            "num_test_triples",
            "num_campaigns",
            "num_provenance_records",
        )

        for field in count_fields:
            if result[field] < 0:
                raise RuntimeError(
                    "Negative enterprise count: "
                    f"{enterprise_id}.{field}"
                )


def validate_stored_hashes(
    artifact,
    suite_root,
):
    for enterprise_id in artifact[
        "enterprise_order"
    ]:
        result = artifact[
            "enterprises"
        ][enterprise_id]

        enterprise_dir = (
            suite_root
            / enterprise_id
        )

        for (
            filename,
            expected_hash,
        ) in result[
            "artifact_hashes"
        ].items():
            path = (
                enterprise_dir
                / filename
            )

            if not path.is_file():
                raise RuntimeError(
                    "Missing suite enterprise artifact: "
                    f"{path}"
                )

            actual_hash = file_sha256(
                path
            )

            if actual_hash != expected_hash:
                raise RuntimeError(
                    "Stored enterprise artifact hash "
                    "mismatch: "
                    f"{enterprise_id}/{filename}"
                )


def validate_enterprises(
    artifact,
    suite_root,
):
    for enterprise_id in artifact[
        "enterprise_order"
    ]:
        print(
            f"Validating {enterprise_id}"
        )

        subprocess.run(
            [
                sys.executable,
                "script/validate_enterprise.py",
                "--root",
                str(suite_root),
                "--enterprise-id",
                enterprise_id,
                "--reprocess",
            ],
            check=True,
        )


def validate_canonical_equivalence(
    artifact,
    suite_root,
    canonical_root,
):
    for enterprise_id in artifact[
        "enterprise_order"
    ]:
        suite_dir = (
            suite_root
            / enterprise_id
        )

        canonical_dir = (
            canonical_root
            / enterprise_id
        )

        if not canonical_dir.is_dir():
            raise RuntimeError(
                "Missing canonical enterprise: "
                f"{canonical_dir}"
            )

        for filename in DATASET_FILES:
            suite_path = (
                suite_dir
                / filename
            )

            canonical_path = (
                canonical_dir
                / filename
            )

            if not canonical_path.is_file():
                raise RuntimeError(
                    "Missing canonical artifact: "
                    f"{canonical_path}"
                )

            suite_hash = file_sha256(
                suite_path
            )

            canonical_hash = file_sha256(
                canonical_path
            )

            if suite_hash != canonical_hash:
                raise RuntimeError(
                    "Suite artifact differs from "
                    "canonical artifact: "
                    f"{enterprise_id}/{filename}"
                )


def main():
    args = parse_args()

    suite_root = Path(
        os.path.expanduser(
            args.suite_root
        )
    )

    canonical_root = Path(
        os.path.expanduser(
            args.canonical_root
        )
    )

    artifact_path = (
        suite_root
        / "heterogeneity_suite.json"
    )

    if not artifact_path.is_file():
        raise RuntimeError(
            "Heterogeneity suite artifact "
            "does not exist"
        )

    artifact = load_json(
        artifact_path
    )

    validate_suite_contract(
        artifact
    )

    validate_stored_hashes(
        artifact=artifact,
        suite_root=suite_root,
    )

    print()
    print("Validating generated enterprises")
    print("--------------------------------")

    validate_enterprises(
        artifact=artifact,
        suite_root=suite_root,
    )

    validate_canonical_equivalence(
        artifact=artifact,
        suite_root=suite_root,
        canonical_root=canonical_root,
    )

    print()
    print("Suite artifact contract: PASS")
    print("Stored artifact hashes: PASS")
    print("Enterprise validation: PASS")
    print(
        "Canonical dataset-bearing artifacts: "
        "BYTE-IDENTICAL"
    )
    print()
    print(
        "Heterogeneity suite validation: PASS"
    )


if __name__ == "__main__":
    main()
