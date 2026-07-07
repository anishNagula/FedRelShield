from dataclasses import dataclass

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
    ):
        if not client_id:
            raise ValueError(
                "client_id must be non-empty"
            )

        self.client_id = client_id
        self.model = model
        self.train_data = train_data
        self.cfg = cfg
        self.device = device

        self.train_loader, self.sampler = (
            training.build_train_loader(
                train_data=self.train_data,
                batch_size=self.cfg.train.batch_size,
                world_size=1,
                rank=0,
            )
        )

    @property
    def num_examples(self):
        return int(
            self.train_data.target_edge_index.shape[1]
        )

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
            sampler_epoch = (
                round_id * local_epochs
                + local_epoch
            )

            self.sampler.set_epoch(sampler_epoch)

            average_loss = training.train_one_epoch(
                cfg=self.cfg,
                model=self.model,
                train_data=self.train_data,
                train_loader=self.train_loader,
                optimizer=optimizer,
                batch_per_epoch=batch_per_epoch,
            )

            epoch_losses.append(average_loss)

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
                sum(epoch_losses) / len(epoch_losses)
            ),
        )
