Here is the updated script with `base_seed=args.seed` added to the `FederatedClient` instantiation. Since `args.seed` is explicitly exposed by `parse_args()`, we can pass it directly into the client constructor.

```python
import argparse
import os
import sys
import copy

import torch
import yaml
from easydict import EasyDict


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
    FedAvgAggregator,
    FederatedClient,
    FederatedCoordinator,
    FederatedServer,
)
from ultra import model_state
from ultra.models import Ultra


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the minimal FedRelShield "
            "federated learning smoke test."
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

    return parser.parse_args()


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


def validate_state_contract(
    reference_state,
    candidate_state,
    description,
):
    if reference_state.keys() != candidate_state.keys():
        raise RuntimeError(
            f"{description}: model-state keys differ"
        )

    for name in reference_state:
        if (
            reference_state[name].shape
            != candidate_state[name].shape
        ):
            raise RuntimeError(
                f"{description}: shape mismatch for {name}"
            )

        if (
            reference_state[name].dtype
            != candidate_state[name].dtype
        ):
            raise RuntimeError(
                f"{description}: dtype mismatch for {name}"
            )


def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    cfg = load_config(args.config)

    device = torch.device("cpu")

    dataset_root = resolve_dataset_root(
        cfg.dataset_root
    )

    checkpoint_path = os.path.expanduser(
        cfg.checkpoint
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    server_model = build_model(cfg)

    model_state.load_model_state(
        server_model,
        checkpoint["model"],
    )

    server_model = server_model.to(device)

    server = FederatedServer(
        model=server_model,
    )

    initial_global_state = (
        server.get_global_state()
    )

    clients = []

    for client_id in cfg.clients:
        dataset = FedRelShieldDataset(
            root=dataset_root,
            enterprise_id=client_id,
        )

        train_data = dataset[0].to(device)

        client_model = build_model(cfg).to(device)

        # Added base_seed parameter here
        client = FederatedClient(
            client_id=client_id,
            model=client_model,
            train_data=train_data,
            cfg=cfg,
            device=device,
            base_seed=args.seed,
        )

        clients.append(client)

        print(
            f"Loaded client {client_id}: "
            f"{client.num_examples} train examples"
        )

    coordinator = FederatedCoordinator(
        server=server,
        clients=clients,
        aggregator=FedAvgAggregator(),
    )

    round_results = coordinator.run(
        num_rounds=cfg.federation.num_rounds,
        local_epochs=cfg.federation.local_epochs,
        batch_per_epoch=(
            cfg.federation.batch_per_epoch
        ),
    )

    previous_global_state = initial_global_state

    for result in round_results:
        print()
        print(f"Round {result.round_id + 1}")

        for client_result in result.client_results:
            validate_state_contract(
                previous_global_state,
                client_result.state,
                (
                    f"round {result.round_id + 1} "
                    f"{client_result.client_id}"
                ),
            )

            print(
                f"  {client_result.client_id}: "
                f"examples={client_result.num_examples}, "
                f"loss={client_result.average_loss:.6f}"
            )

        validate_state_contract(
            previous_global_state,
            result.global_state,
            f"round {result.round_id + 1} global",
        )

        if model_state.model_states_equal(
            previous_global_state,
            result.global_state,
        ):
            raise RuntimeError(
                "Global model state did not change "
                f"during round {result.round_id + 1}"
            )

        print("  aggregate contract: PASS")
        print("  server load contract: PASS")
        print("  global state changed: PASS")

        previous_global_state = result.global_state

    final_global_state = (
        server.get_global_state()
    )

    if not model_state.model_states_equal(
        final_global_state,
        round_results[-1].global_state,
    ):
        raise RuntimeError(
            "Final server state does not match "
            "the final federated round state"
        )

    print()
    print("Federated smoke test: PASS")
    print(
        f"Rounds: {len(round_results)}"
    )
    print(
        f"Clients: {len(clients)}"
    )
    print(
        "Final global ULTRA state: VALID"
    )


if __name__ == "__main__":
    main()

```
