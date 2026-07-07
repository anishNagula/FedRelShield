import copy
import os
from pathlib import Path

import torch

from ultra import model_state


CHECKPOINT_FORMAT_VERSION = "3.0"


def build_federated_checkpoint(
    global_state,
    completed_round,
    seed,
    client_ids,
    federation_config,
    metrics_artifact,
):
    if completed_round < 0:
        raise ValueError(
            "completed_round must be non-negative"
        )

    client_ids = tuple(client_ids)

    if len(client_ids) < 2:
        raise ValueError(
            "Checkpoint requires at least two clients"
        )

    if len(client_ids) != len(set(client_ids)):
        raise ValueError(
            "Checkpoint client IDs must be unique"
        )

    expected_round_ids = list(
        range(completed_round + 1)
    )

    actual_round_ids = [
        round_metrics["round_id"]
        for round_metrics
        in metrics_artifact["rounds"]
    ]

    if actual_round_ids != expected_round_ids:
        raise ValueError(
            "Checkpoint metrics history must contain "
            "all completed rounds exactly once"
        )

    return {
        "format_version":
            CHECKPOINT_FORMAT_VERSION,
        "completed_round":
            completed_round,
        "seed":
            int(seed),
        "client_ids":
            list(client_ids),
        "federation_config":
            dict(federation_config),
        "global_state":
            model_state.clone_model_state(
                global_state,
                device="cpu",
            ),
        "metrics_artifact":
            copy.deepcopy(metrics_artifact),
    }


def save_federated_checkpoint(
    checkpoint,
    path,
):
    path = Path(
        os.path.expanduser(path)
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        path.name + ".tmp"
    )

    torch.save(
        checkpoint,
        temporary_path,
    )

    os.replace(
        temporary_path,
        path,
    )


def load_federated_checkpoint(
    path,
    map_location="cpu",
):
    path = Path(
        os.path.expanduser(path)
    )

    checkpoint = torch.load(
        path,
        map_location=map_location,
    )

    if checkpoint.get(
        "format_version"
    ) != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            "Unsupported federated checkpoint format"
        )

    required_fields = {
        "completed_round",
        "seed",
        "client_ids",
        "federation_config",
        "global_state",
        "metrics_artifact",
    }

    missing_fields = (
        required_fields - checkpoint.keys()
    )

    if missing_fields:
        raise ValueError(
            "Federated checkpoint missing fields: "
            f"{sorted(missing_fields)}"
        )

    return checkpoint


def validate_resume_compatibility(
    checkpoint,
    seed,
    client_ids,
    federation_config,
):
    if checkpoint["seed"] != int(seed):
        raise ValueError(
            "Resume seed does not match checkpoint"
        )

    if checkpoint["client_ids"] != list(client_ids):
        raise ValueError(
            "Resume client IDs do not match checkpoint"
        )

    if (
        checkpoint["federation_config"]
        != dict(federation_config)
    ):
        raise ValueError(
            "Resume federation configuration "
            "does not match checkpoint"
        )
