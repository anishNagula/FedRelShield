from dataclasses import dataclass
import hashlib

import torch

from ultra import model_state, training


@dataclass(frozen=True)
class LocalTrainingResult:
    client_id: str
    state: dict
    num_examples: int
    average_loss: float


class FederatedClient:
    def __init__(
        self,
        client_id,
        model,
        train_data,
        cfg,
        device,
        base_seed,
    ):
        if not client_id:
            raise ValueError(
                "client_id must be non-empty"
            )

        if base_seed < 0:
            raise ValueError(
                "base_seed must be non-negative"
            )

        self.client_id = client_id
        self.model = model
        self.train_data = train_data
        self.cfg = cfg
        self.device = device
        self.base_seed = int(base_seed)

    @property
    def num_examples(self):
        return int(
            self.train_data.target_edge_index.shape[1]
        )

    def _client_seed_offset(self):
        digest = hashlib.sha256(
            self.client_id.encode("utf-8")
        ).digest()

        return int.from_bytes(
            digest[:8],
            byteorder="big",
            signed=False,
        )

    def _local_seed(
        self,
        round_id,
        local_epoch,
    ):
        if round_id < 0:
            raise ValueError(
                "round_id must be non-negative"
            )

        if local_epoch < 0:
            raise ValueError(
                "local_epoch must be non-negative"
            )

        max_seed = 2**63 - 1

        return (
            self.base_seed
            + self._client_seed_offset()
            + round_id * 1_000_003
            + local_epoch * 10_007
        ) % max_seed

    def _build_epoch_loader(
        self,
        local_seed,
    ):
        train_loader, sampler = (
            training.build_train_loader(
                train_data=self.train_data,
                batch_size=self.cfg.train.batch_size,
                world_size=1,
                rank=0,
            )
        )

        sampler.set_epoch(local_seed)

        return train_loader

    def train(
        self,
        global_state,
        local_epochs,
        batch_per_epoch,
        round_id,
    ):
        if local_epochs <= 0:
            raise ValueError(
                "local_epochs must be positive"
            )

        if batch_per_epoch <= 0:
            raise ValueError(
                "batch_per_epoch must be positive"
            )

        model_state.load_model_state(
            self.model,
            global_state,
        )

        optimizer = training.build_optimizer(
            self.cfg,
            self.model,
        )

        epoch_losses = []

        for local_epoch in range(local_epochs):
            local_seed = self._local_seed(
                round_id=round_id,
                local_epoch=local_epoch,
            )

            devices = []

            if self.device.type == "cuda":
                devices = [self.device]

            with torch.random.fork_rng(
                devices=devices,
                enabled=True,
            ):
                torch.manual_seed(local_seed)

                if self.device.type == "cuda":
                    torch.cuda.manual_seed_all(
                        local_seed
                    )

                train_loader = (
                    self._build_epoch_loader(
                        local_seed=local_seed,
                    )
                )

                average_loss = (
                    training.train_one_epoch(
                        cfg=self.cfg,
                        model=self.model,
                        train_data=self.train_data,
                        train_loader=train_loader,
                        optimizer=optimizer,
                        batch_per_epoch=batch_per_epoch,
                    )
                )

            epoch_losses.append(
                average_loss
            )

        local_state = model_state.get_model_state(
            self.model,
            device="cpu",
            clone=True,
        )

        return LocalTrainingResult(
            client_id=self.client_id,
            state=local_state,
            num_examples=self.num_examples,
            average_loss=(
                sum(epoch_losses)
                / len(epoch_losses)
            ),
        )
