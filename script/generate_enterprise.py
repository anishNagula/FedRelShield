import argparse
from pathlib import Path

from fedrelshield.data.attacks import AttackInjector
from fedrelshield.data.benign import BenignEventGenerator
from fedrelshield.data.config import (
    enterprise_config_metadata,
    load_enterprise_config,
)
from fedrelshield.data.exporter import KGExporter
from fedrelshield.data.topology import EnterpriseATopologyGenerator
from fedrelshield.data.writer import EnterpriseDatasetWriter


TOPOLOGY_GENERATORS = {
    "enterprise_a": EnterpriseATopologyGenerator,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a deterministic FedRelShield "
            "enterprise benchmark artifact."
        )
    )

    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="enterprise generation YAML configuration",
    )

    return parser.parse_args()


def build_topology(config):
    try:
        generator_class = TOPOLOGY_GENERATORS[
            config.schema
        ]

    except KeyError as error:
        raise ValueError(
            "Unsupported enterprise schema: "
            f"{config.schema}"
        ) from error

    topology = generator_class(
        seed=config.topology_seed,
    ).generate()

    if topology.enterprise_id != config.enterprise_id:
        raise ValueError(
            "Generated topology enterprise ID does not "
            "match configuration: "
            f"{topology.enterprise_id} != "
            f"{config.enterprise_id}"
        )

    return topology


def generate_enterprise(config):
    topology = build_topology(config)

    benign_events = BenignEventGenerator(
        seed=config.benign_seed,
        num_events=config.benign_num_events,
    ).generate(topology)

    attack_campaigns = AttackInjector(
        seed=config.attack_seed,
        num_campaigns=config.attack_num_campaigns,
        start_timestamp=config.attack_start_timestamp,
        timestamp_step=config.attack_timestamp_step,
        max_attempts_per_campaign=(
            config.attack_max_attempts_per_campaign
        ),
    ).inject(topology)

    dataset = KGExporter(
        seed=config.exporter_seed,
        train_campaigns=config.exporter_train_campaigns,
        valid_campaigns=config.exporter_valid_campaigns,
        test_campaigns=config.exporter_test_campaigns,
    ).build(
        topology=topology,
        benign_events=benign_events,
        attack_campaigns=attack_campaigns,
    )

    return topology, dataset


def main():
    args = parse_args()

    config = load_enterprise_config(args.config)

    topology, dataset = generate_enterprise(config)

    output_dir = (
        Path(config.output_root)
        / config.enterprise_id
    )

    writer = EnterpriseDatasetWriter()

    hashes = writer.write(
        output_dir=output_dir,
        enterprise_id=config.enterprise_id,
        dataset=dataset,
        generation_metadata=enterprise_config_metadata(
            config
        ),
    )

    print(f"Enterprise: {config.enterprise_id}")
    print(f"Profile: {config.profile}")
    print(f"Schema: {config.schema}")
    print(f"Output: {output_dir}")
    print(f"Train triples: {len(dataset.train_triples)}")
    print(f"Valid triples: {len(dataset.valid_triples)}")
    print(f"Test triples: {len(dataset.test_triples)}")
    print(f"Campaigns: {len(dataset.campaigns)}")
    print(
        f"Provenance records: "
        f"{len(dataset.provenance)}"
    )
    print(
        f"Manifest SHA-256: "
        f"{hashes['manifest.json']}"
    )


if __name__ == "__main__":
    main()
