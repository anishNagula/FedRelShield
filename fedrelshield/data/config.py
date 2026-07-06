from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class EnterpriseConfig:
    enterprise_id: str
    profile: str
    schema: str
    schema_version: str

    topology_seed: int

    benign_seed: int
    benign_num_events: int

    attack_seed: int
    attack_num_campaigns: int
    attack_start_timestamp: int
    attack_timestamp_step: int
    attack_max_attempts_per_campaign: int

    exporter_seed: int
    exporter_train_campaigns: int
    exporter_valid_campaigns: int
    exporter_test_campaigns: int

    output_root: str

    format_version: str


def enterprise_config_metadata(
    config: EnterpriseConfig,
):
    return asdict(config)


def load_enterprise_config(
    config_path,
) -> EnterpriseConfig:
    path = Path(config_path)

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        raw_config = yaml.safe_load(file)

    if not isinstance(raw_config, dict):
        raise ValueError(
            "Enterprise configuration must be a YAML mapping"
        )

    try:
        enterprise = raw_config["enterprise"]
        generation = raw_config["generation"]

        topology = generation["topology"]
        benign = generation["benign"]
        attacks = generation["attacks"]
        exporter = generation["exporter"]

        output = raw_config["output"]

        config = EnterpriseConfig(
            enterprise_id=enterprise["id"],
            profile=enterprise["profile"],
            schema=enterprise["schema"],
            schema_version=enterprise["schema_version"],

            topology_seed=topology["seed"],

            benign_seed=benign["seed"],
            benign_num_events=benign["num_events"],

            attack_seed=attacks["seed"],
            attack_num_campaigns=attacks["num_campaigns"],
            attack_start_timestamp=attacks["start_timestamp"],
            attack_timestamp_step=attacks["timestamp_step"],
            attack_max_attempts_per_campaign=attacks[
                "max_attempts_per_campaign"
            ],

            exporter_seed=exporter["seed"],
            exporter_train_campaigns=exporter[
                "train_campaigns"
            ],
            exporter_valid_campaigns=exporter[
                "valid_campaigns"
            ],
            exporter_test_campaigns=exporter[
                "test_campaigns"
            ],

            output_root=output["root"],

            format_version=raw_config["format_version"],
        )

    except KeyError as error:
        raise ValueError(
            "Missing required enterprise configuration field: "
            f"{error.args[0]}"
        ) from error

    _validate_enterprise_config(config)

    return config


def _validate_enterprise_config(
    config: EnterpriseConfig,
):
    if not config.enterprise_id:
        raise ValueError(
            "enterprise.id must be non-empty"
        )

    if not config.profile:
        raise ValueError(
            "enterprise.profile must be non-empty"
        )

    if not config.schema:
        raise ValueError(
            "enterprise.schema must be non-empty"
        )

    if not config.schema_version:
        raise ValueError(
            "enterprise.schema_version must be non-empty"
        )

    if not config.format_version:
        raise ValueError(
            "format_version must be non-empty"
        )

    if config.topology_seed < 0:
        raise ValueError(
            "generation.topology.seed must be non-negative"
        )

    if config.benign_seed < 0:
        raise ValueError(
            "generation.benign.seed must be non-negative"
        )

    if config.benign_num_events < 0:
        raise ValueError(
            "generation.benign.num_events must be non-negative"
        )

    if config.attack_seed < 0:
        raise ValueError(
            "generation.attacks.seed must be non-negative"
        )

    if config.attack_num_campaigns < 0:
        raise ValueError(
            "generation.attacks.num_campaigns must be non-negative"
        )

    if config.attack_start_timestamp < 0:
        raise ValueError(
            "generation.attacks.start_timestamp must be non-negative"
        )

    if config.attack_timestamp_step <= 0:
        raise ValueError(
            "generation.attacks.timestamp_step must be positive"
        )

    if config.attack_max_attempts_per_campaign <= 0:
        raise ValueError(
            "generation.attacks.max_attempts_per_campaign "
            "must be positive"
        )

    if config.exporter_seed < 0:
        raise ValueError(
            "generation.exporter.seed must be non-negative"
        )

    split_count = (
        config.exporter_train_campaigns
        + config.exporter_valid_campaigns
        + config.exporter_test_campaigns
    )

    if min(
        config.exporter_train_campaigns,
        config.exporter_valid_campaigns,
        config.exporter_test_campaigns,
    ) < 0:
        raise ValueError(
            "exporter campaign split counts must be non-negative"
        )

    if split_count != config.attack_num_campaigns:
        raise ValueError(
            "Exporter campaign split counts must sum to "
            "generation.attacks.num_campaigns"
        )

    if not config.output_root:
        raise ValueError(
            "output.root must be non-empty"
        )
