import argparse
import shutil
from pathlib import Path

import torch

import os
import sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)

from fedrelshield.data.dataset import (
    FedRelShieldDataset,
)
from fedrelshield.data.writer import (
    EnterpriseDatasetWriter,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate a generated FedRelShield "
            "enterprise artifact and ULTRA dataset."
        )
    )

    parser.add_argument(
        "--root",
        required=True,
        help="FedRelShield dataset root",
    )

    parser.add_argument(
        "--enterprise-id",
        required=True,
        help="enterprise identifier",
    )

    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="delete processed dataset before loading",
    )

    return parser.parse_args()


def triple_set(data):
    return {
        (
            int(head),
            int(tail),
            int(relation),
        )
        for (
            head,
            tail,
            relation,
        ) in zip(
            data.target_edge_index[0],
            data.target_edge_index[1],
            data.target_edge_type,
        )
    }


def main():
    args = parse_args()

    root = (
        Path(args.root)
        .expanduser()
        .resolve()
    )

    enterprise_dir = (
        root
        / args.enterprise_id
    )

    if not enterprise_dir.exists():
        raise FileNotFoundError(
            f"Enterprise artifact does not exist: "
            f"{enterprise_dir}"
        )

    writer = EnterpriseDatasetWriter()

    writer.verify_artifact(
        enterprise_dir
    )

    print("Artifact verification: PASS")

    processed_dir = (
        enterprise_dir
        / "processed"
    )

    if (
        args.reprocess
        and processed_dir.exists()
    ):
        shutil.rmtree(
            processed_dir
        )

        print(
            "Removed existing processed dataset"
        )

    dataset = FedRelShieldDataset(
        root=str(root),
        enterprise_id=args.enterprise_id,
    )

    if len(dataset) != 3:
        raise AssertionError(
            "Dataset must contain exactly "
            "train, valid and test Data objects"
        )

    train_data = dataset[0]
    valid_data = dataset[1]
    test_data = dataset[2]

    if train_data.num_nodes <= 0:
        raise AssertionError(
            "Dataset must contain entities"
        )

    if train_data.num_relations <= 0:
        raise AssertionError(
            "Dataset must contain relations"
        )

    if train_data.num_relations % 2 != 0:
        raise AssertionError(
            "ULTRA transductive relation count "
            "must include forward and inverse relations"
        )

    if (
        train_data.num_nodes
        != valid_data.num_nodes
        or train_data.num_nodes
        != test_data.num_nodes
    ):
        raise AssertionError(
            "Entity count differs across splits"
        )

    if (
        train_data.num_relations
        != valid_data.num_relations
        or train_data.num_relations
        != test_data.num_relations
    ):
        raise AssertionError(
            "Relation count differs across splits"
        )

    if not torch.equal(
        train_data.edge_index,
        valid_data.edge_index,
    ):
        raise AssertionError(
            "Validation fact graph differs "
            "from training fact graph"
        )

    if not torch.equal(
        train_data.edge_index,
        test_data.edge_index,
    ):
        raise AssertionError(
            "Test fact graph differs "
            "from training fact graph"
        )

    if not torch.equal(
        train_data.edge_type,
        valid_data.edge_type,
    ):
        raise AssertionError(
            "Validation fact relation types differ "
            "from training fact graph"
        )

    if not torch.equal(
        train_data.edge_type,
        test_data.edge_type,
    ):
        raise AssertionError(
            "Test fact relation types differ "
            "from training fact graph"
        )

    train_targets = triple_set(
        train_data
    )

    valid_targets = triple_set(
        valid_data
    )

    test_targets = triple_set(
        test_data
    )

    if not train_targets:
        raise AssertionError(
            "Training targets are empty"
        )

    if not valid_targets:
        raise AssertionError(
            "Validation targets are empty"
        )

    if not test_targets:
        raise AssertionError(
            "Test targets are empty"
        )

    if train_targets & valid_targets:
        raise AssertionError(
            "Train / validation target overlap detected"
        )

    if train_targets & test_targets:
        raise AssertionError(
            "Train / test target overlap detected"
        )

    if valid_targets & test_targets:
        raise AssertionError(
            "Validation / test target overlap detected"
        )

    num_relations = train_data.num_relations

    if torch.is_tensor(num_relations):
        num_relations = int(num_relations.item())
    else:
        num_relations = int(num_relations)
    
    num_forward_relations = num_relations // 2

    for split_name, data in [
        ("train", train_data),
        ("valid", valid_data),
        ("test", test_data),
    ]:
        if (
            data.target_edge_type.min().item()
            < 0
        ):
            raise AssertionError(
                f"{split_name} contains "
                "negative relation IDs"
            )

        if (
            data.target_edge_type.max().item()
            >= num_forward_relations
        ):
            raise AssertionError(
                f"{split_name} target relations "
                "must contain forward relations only"
            )

        if (
            data.target_edge_index.min().item()
            < 0
        ):
            raise AssertionError(
                f"{split_name} contains "
                "negative entity IDs"
            )

        if (
            data.target_edge_index.max().item()
            >= data.num_nodes
        ):
            raise AssertionError(
                f"{split_name} contains "
                "out-of-range entity IDs"
            )

    print("Dataset processing: PASS")
    print("Transductive contract: PASS")
    print("Split disjointness: PASS")
    print("Relation ID validation: PASS")
    print("Entity ID validation: PASS")

    print()
    print("Dataset summary")
    print("------------------------------")
    print(
        f"Entities: "
        f"{train_data.num_nodes}"
    )
    print(
        f"Forward relations: "
        f"{num_forward_relations}"
    )
    print(
        f"ULTRA relations: "
        f"{num_relations}"
    )
    print(
        f"Fact graph directed edges: "
        f"{train_data.edge_index.shape[1]}"
    )
    print(
        f"Train targets: "
        f"{len(train_targets)}"
    )
    print(
        f"Valid targets: "
        f"{len(valid_targets)}"
    )
    print(
        f"Test targets: "
        f"{len(test_targets)}"
    )

    print()
    print("Enterprise validation: PASS")


if __name__ == "__main__":
    main()

