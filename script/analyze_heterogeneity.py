import argparse
import json
import os
import sys
from pathlib import Path


sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)


from fedrelshield.analysis.heterogeneity import (
    HeterogeneityAnalyzer,
)


DEFAULT_ENTERPRISE_IDS = (
    "enterprise_a",
    "enterprise_b",
    "enterprise_c",
    "enterprise_d",
    "enterprise_e",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze heterogeneity across "
            "FedRelShield enterprise artifacts."
        )
    )

    parser.add_argument(
        "--root",
        default="data/fedrelshield",
        help="enterprise artifact root",
    )

    parser.add_argument(
        "--enterprise-ids",
        nargs="+",
        default=list(DEFAULT_ENTERPRISE_IDS),
    )

    parser.add_argument(
        "--output",
        default=(
            "data/fedrelshield/"
            "heterogeneity_report.json"
        ),
    )

    return parser.parse_args()


def serialize_json(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )


def main():
    args = parse_args()

    analyzer = HeterogeneityAnalyzer(
        root=args.root,
        enterprise_ids=args.enterprise_ids,
    )

    report = analyzer.analyze()

    output_path = Path(args.output)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        serialize_json(report),
        encoding="utf-8",
    )

    print("Heterogeneity analysis: PASS")
    print(
        f"Enterprises: "
        f"{report['summary']['num_enterprises']}"
    )
    print(
        f"Pairs: "
        f"{report['summary']['num_pairs']}"
    )
    print(
        "Mean relation JSD: "
        f"{report['summary']['pairwise_means']['relation_distribution_jsd']:.6f}"
    )
    print(
        "Mean entity-type JSD: "
        f"{report['summary']['pairwise_means']['entity_type_distribution_jsd']:.6f}"
    )
    print(
        "Mean degree JSD: "
        f"{report['summary']['pairwise_means']['degree_distribution_jsd']:.6f}"
    )
    print(
        "Most heterogeneous pair: "
        f"{report['summary']['most_heterogeneous_pair']}"
    )
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
