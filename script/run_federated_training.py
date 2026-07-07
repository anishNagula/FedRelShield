import argparse
import copy
import json
import os
import sys

import torch
import yaml
from easydict import EasyDict
from torch_geometric.data import Data


sys.path.append(
    os.path.dirname(
        os.path.dirname(__file__)
    )
)


from fedrelshield.data.dataset import (
    FedRelShieldDataset,
    resolve_dataset_root,
)
from fedrelshield.federation import (
    EnterpriseEvaluationResult,
    FedAvgAggregator,
    FederatedClient,
    FederatedCoordinator,
    FederatedServer,
    aggregate_enterprise_metrics,
    build_federated_checkpoint,
    evaluation_result_to_dict,
    load_federated_checkpoint,
    save_federated_checkpoint,
    validate_resume_compatibility,
)
from ultra import evaluation, model_state
from ultra.models import Ultra


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run config-driven federated "
            "ULTRA training."
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

    parser.add_argument(
        "--resume",
        default=None,
        help="federated checkpoint path",
    )

    parser.add_argument(
        "--stop-after-round",
        type=int,
        default=None,
        help=(
            "stop after this zero-based absolute round; "
            "used for controlled interruption tests"
        ),
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


def build_model(cfg):
    return Ultra(
        rel_model_cfg=copy.deepcopy(
            cfg.model.relation_model
        ),
        entity_model_cfg=copy.deepcopy(
            cfg.model.entity_model
        ),
    )


def build_seeded_model(
    cfg,
    seed,
    device,
):
    devices = []

    if device.type == "cuda":
        devices = [device]

    with torch.random.fork_rng(
        devices=devices,
        enabled=True,
    ):
        torch.manual_seed(seed)

        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

        model = build_model(cfg)

    return model.to(device)


def build_filtered_data(dataset):
    return Data(
        edge_index=dataset._data.target_edge_index,
        edge_type=dataset._data.target_edge_type,
        num_nodes=dataset[0].num_nodes,
    )


def load_enterprises(
    cfg,
    dataset_root,
    device,
):
    enterprises = {}

    for enterprise_id in cfg.clients:
        dataset = FedRelShieldDataset(
            root=dataset_root,
            enterprise_id=enterprise_id,
        )

        enterprises[enterprise_id] = {
            "dataset": dataset,
            "train_data": dataset[0].to(device),
            "valid_data": dataset[1].to(device),
            "test_data": dataset[2].to(device),
            "filtered_data":
                build_filtered_data(dataset).to(device),
        }

        print(
            f"Loaded {enterprise_id}: "
            f"train="
            f"{dataset[0].target_edge_index.shape[1]}, "
            f"valid="
            f"{dataset[1].target_edge_index.shape[1]}, "
            f"test="
            f"{dataset[2].target_edge_index.shape[1]}"
        )

    return enterprises


def build_global_evaluator(
    cfg,
    enterprises,
    device,
):
    split_name = cfg.evaluation.split

    if split_name not in {"valid", "test"}:
        raise ValueError(
            "evaluation.split must be valid or test"
        )

    def evaluate_global_model(model):
        enterprise_results = []

        for enterprise_id in cfg.clients:
            enterprise = enterprises[enterprise_id]

            eval_data = enterprise[
                f"{split_name}_data"
            ]

            metrics = evaluation.evaluate(
                cfg=cfg,
                model=model,
                eval_data=eval_data,
                filtered_data=enterprise["filtered_data"],
                device=device,
                logger=None,
                return_metrics=True,
            )

            enterprise_results.append(
                EnterpriseEvaluationResult(
                    enterprise_id=enterprise_id,
                    num_examples=int(
                        eval_data
                        .target_edge_index
                        .shape[1]
                    ),
                    metrics=metrics,
                )
            )

        return aggregate_enterprise_metrics(
            enterprise_results
        )

    return evaluate_global_model


def build_empty_metrics_artifact(
    cfg,
    seed,
):
    return {
        "format_version": "1.0",
        "seed": int(seed),
        "clients": list(cfg.clients),
        "evaluation_split":
            cfg.evaluation.split,
        "federation": {
            "num_rounds":
                cfg.federation.num_rounds,
            "local_epochs":
                cfg.federation.local_epochs,
            "batch_per_epoch":
                cfg.federation.batch_per_epoch,
        },
        "rounds": [],
    }


def append_round_metrics(
    artifact,
    result,
):
    if result.evaluation_result is None:
        raise RuntimeError(
            "Round evaluation result is missing"
        )

    expected_round_id = len(
        artifact["rounds"]
    )

    if result.round_id != expected_round_id:
        raise RuntimeError(
            "Round metrics are not contiguous: "
            f"expected={expected_round_id}, "
            f"actual={result.round_id}"
        )

    artifact["rounds"].append(
        {
            "round_id": result.round_id,
            "local_training": {
                client_result.client_id: {
                    "num_examples":
                        client_result.num_examples,
                    "average_loss":
                        float(
                            client_result.average_loss
                        ),
                }
                for client_result
                in result.client_results
            },
            "global_evaluation":
                evaluation_result_to_dict(
                    result.evaluation_result
                ),
        }
    )


def validate_metrics_artifact(
    artifact,
    require_complete=False,
):
    if artifact["format_version"] != "1.0":
        raise RuntimeError(
            "Unexpected metrics format version"
        )

    expected_clients = set(
        artifact["clients"]
    )

    configured_rounds = artifact[
        "federation"
    ]["num_rounds"]

    rounds = artifact["rounds"]

    if require_complete:
        if len(rounds) != configured_rounds:
            raise RuntimeError(
                "Metrics artifact is incomplete"
            )
    elif len(rounds) > configured_rounds:
        raise RuntimeError(
            "Metrics artifact contains too many rounds"
        )

    actual_round_ids = [
        round_metrics["round_id"]
        for round_metrics in rounds
    ]

    if actual_round_ids != list(
        range(len(rounds))
    ):
        raise RuntimeError(
            "Metrics round IDs are not contiguous"
        )

    for round_metrics in rounds:
        local_clients = set(
            round_metrics[
                "local_training"
            ].keys()
        )

        eval_clients = set(
            round_metrics[
                "global_evaluation"
            ]["enterprises"].keys()
        )

        if local_clients != expected_clients:
            raise RuntimeError(
                "Local-training client coverage mismatch"
            )

        if eval_clients != expected_clients:
            raise RuntimeError(
                "Evaluation client coverage mismatch"
            )

        global_eval = round_metrics[
            "global_evaluation"
        ]

        for aggregate_name in (
            "macro_metrics",
            "weighted_metrics",
        ):
            if "mrr" not in global_eval[aggregate_name]:
                raise RuntimeError(
                    f"{aggregate_name} is missing mrr"
                )


def federation_config_dict(cfg):
    return {
        "num_rounds":
            int(cfg.federation.num_rounds),
        "local_epochs":
            int(cfg.federation.local_epochs),
        "batch_per_epoch":
            int(cfg.federation.batch_per_epoch),
        "batch_size":
            int(cfg.train.batch_size),
        "evaluation_split":
            str(cfg.evaluation.split),
    }


def write_metrics_artifact(
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

    dataset_root = resolve_dataset_root(
        cfg.dataset_root
    )

    output_dir = os.path.expanduser(
        cfg.output_dir
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    metrics_path = os.path.join(
        output_dir,
        "round_metrics.json",
    )

    enterprises = load_enterprises(
        cfg=cfg,
        dataset_root=dataset_root,
        device=device,
    )

    server_model = build_seeded_model(
        cfg=cfg,
        seed=args.seed,
        device=device,
    )

    metrics_artifact = (
        build_empty_metrics_artifact(
            cfg=cfg,
            seed=args.seed,
        )
    )

    start_round = 0

    if args.resume is not None:
        checkpoint = load_federated_checkpoint(
            args.resume,
            map_location="cpu",
        )

        validate_resume_compatibility(
            checkpoint=checkpoint,
            seed=args.seed,
            client_ids=cfg.clients,
            federation_config=
                federation_config_dict(cfg),
        )

        model_state.load_model_state(
            server_model,
            checkpoint["global_state"],
        )

        metrics_artifact = checkpoint[
            "metrics_artifact"
        ]

        validate_metrics_artifact(
            metrics_artifact,
            require_complete=False,
        )

        start_round = (
            checkpoint["completed_round"] + 1
        )

        if len(
            metrics_artifact["rounds"]
        ) != start_round:
            raise RuntimeError(
                "Checkpoint round and metrics "
                "history length disagree"
            )

        print(
            "Resumed checkpoint: "
            f"completed_round="
            f"{checkpoint['completed_round']}"
        )

    else:
        checkpoint_path = os.path.expanduser(
            cfg.checkpoint
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        model_state.load_model_state(
            server_model,
            checkpoint["model"],
        )

    server = FederatedServer(
        model=server_model,
    )

    clients = []

    for client_index, enterprise_id in enumerate(
        cfg.clients
    ):
        client_seed = (
            args.seed
            + 100_000
            + client_index
        )

        clients.append(
            FederatedClient(
                client_id=enterprise_id,
                model=build_seeded_model(
                    cfg=cfg,
                    seed=client_seed,
                    device=device,
                ),
                train_data=enterprises[
                    enterprise_id
                ]["train_data"],
                cfg=cfg,
                device=device,
                base_seed=args.seed,
            )
        )

    evaluator = build_global_evaluator(
        cfg=cfg,
        enterprises=enterprises,
        device=device,
    )

    coordinator = FederatedCoordinator(
        server=server,
        clients=clients,
        aggregator=FedAvgAggregator(),
        evaluator=evaluator,
    )

    configured_rounds = int(
        cfg.federation.num_rounds
    )

    if start_round > configured_rounds:
        raise RuntimeError(
            "Checkpoint is beyond configured rounds"
        )

    if start_round == configured_rounds:
        validate_metrics_artifact(
            metrics_artifact,
            require_complete=True,
        )

        write_metrics_artifact(
            metrics_artifact,
            metrics_path,
        )

        print("Federated training already complete")
        print(f"Metrics: {metrics_path}")
        return

    checkpointing_enabled = bool(
        cfg.checkpointing.enabled
    )

    federated_checkpoint_path = (
        os.path.expanduser(
            cfg.checkpointing.checkpoint_path
        )
    )

    print()
    print(
        "Starting federated training: "
        f"round={start_round}"
    )

    for round_id in range(
        start_round,
        configured_rounds,
    ):
        result = coordinator.run_round(
            round_id=round_id,
            local_epochs=
                cfg.federation.local_epochs,
            batch_per_epoch=
                cfg.federation.batch_per_epoch,
        )

        append_round_metrics(
            metrics_artifact,
            result,
        )

        validate_metrics_artifact(
            metrics_artifact,
            require_complete=False,
        )

        write_metrics_artifact(
            metrics_artifact,
            metrics_path,
        )

        if checkpointing_enabled:
            federated_checkpoint = (
                build_federated_checkpoint(
                    global_state=result.global_state,
                    completed_round=round_id,
                    seed=args.seed,
                    client_ids=cfg.clients,
                    federation_config=
                        federation_config_dict(cfg),
                    metrics_artifact=
                        metrics_artifact,
                )
            )

            save_federated_checkpoint(
                federated_checkpoint,
                federated_checkpoint_path,
            )

        print()
        print(f"Round {round_id + 1}")

        for client_result in (
            result.client_results
        ):
            print(
                f"  {client_result.client_id}: "
                f"loss="
                f"{client_result.average_loss:.6f}"
            )

        print(
            "  macro MRR: "
            f"{result.evaluation_result.macro_metrics['mrr']:.6f}"
        )

        print(
            "  weighted MRR: "
            f"{result.evaluation_result.weighted_metrics['mrr']:.6f}"
        )

        if (
            args.stop_after_round is not None
            and round_id >= args.stop_after_round
        ):
            print()
            print(
                "Controlled stop after round "
                f"{round_id}"
            )
            print(
                f"Metrics: {metrics_path}"
            )
            return

    validate_metrics_artifact(
        metrics_artifact,
        require_complete=True,
    )

    print()
    print("Metrics artifact validation: PASS")
    print("Federated training: PASS")
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
