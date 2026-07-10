import copy
import os
from dataclasses import dataclass

import torch
from torch_geometric.data import Data

from fedrelshield.data.dataset import (
    FedRelShieldDataset,
    resolve_dataset_root,
)
from fedrelshield.federation import (
    EnterpriseEvaluationResult,
    aggregate_enterprise_metrics,
    evaluation_result_to_dict,
)
from ultra import evaluation, model_state, tasks, training
from ultra.models import Ultra


BASELINE_NAMES = (
    "zero_shot",
    "local_only",
    "centralized",
    "federated",
    "random_init_federated",
)


@dataclass(frozen=True)
class BaselineRunConfig:
    baseline_name: str
    seed: int
    evaluation_split: str
    client_ids: tuple


class BaselineRunner:
    def __init__(
        self,
        cfg,
        run_config,
        device,
    ):
        self.cfg = cfg
        self.run_config = run_config
        self.device = device

        self._validate_run_config()

        self.dataset_root = resolve_dataset_root(
            cfg.dataset_root
        )

        self.enterprises = self._load_enterprises()

    def run(self):
        if self.run_config.baseline_name == "zero_shot":
            return self._run_zero_shot()

        if self.run_config.baseline_name == "local_only":
            return self._run_local_only()

        if self.run_config.baseline_name == "centralized":
            return self._run_centralized()

        raise NotImplementedError(
            "Baseline is registered but not implemented: "
            f"{self.run_config.baseline_name}"
        )

    def _validate_run_config(self):
        if (
            self.run_config.baseline_name
            not in BASELINE_NAMES
        ):
            raise ValueError(
                "Unknown baseline: "
                f"{self.run_config.baseline_name}"
            )

        if self.run_config.seed < 0:
            raise ValueError(
                "seed must be non-negative"
            )

        if (
            self.run_config.evaluation_split
            not in {"valid", "test"}
        ):
            raise ValueError(
                "evaluation_split must be valid or test"
            )

        if not self.run_config.client_ids:
            raise ValueError(
                "client_ids must be non-empty"
            )

        if len(
            self.run_config.client_ids
        ) != len(
            set(self.run_config.client_ids)
        ):
            raise ValueError(
                "client_ids must be unique"
            )

        if (
            self.run_config.baseline_name == "local_only"
        ):
            self._validate_training_config(
                config_name="local_training",
            )

        if (
            self.run_config.baseline_name == "centralized"
        ):
            self._validate_training_config(
                config_name="centralized_training",
            )

    def _validate_training_config(
        self,
        config_name,
    ):
        if not hasattr(
            self.cfg,
            config_name,
        ):
            raise ValueError(
                f"{self.run_config.baseline_name} "
                f"baseline requires {config_name} config"
            )

        training_cfg = getattr(
            self.cfg,
            config_name,
        )

        if training_cfg.local_epochs <= 0:
            raise ValueError(
                "local_epochs must be positive"
            )

        if training_cfg.batch_per_epoch <= 0:
            raise ValueError(
                "batch_per_epoch must be positive"
            )

    def _build_model(self):
        return Ultra(
            rel_model_cfg=copy.deepcopy(
                self.cfg.model.relation_model
            ),
            entity_model_cfg=copy.deepcopy(
                self.cfg.model.entity_model
            ),
        )

    def _build_seeded_model(self):
        devices = []

        if self.device.type == "cuda":
            devices = [self.device]

        with torch.random.fork_rng(
            devices=devices,
            enabled=True,
        ):
            torch.manual_seed(
                self.run_config.seed
            )

            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(
                    self.run_config.seed
                )

            model = self._build_model()

        return model.to(self.device)

    def _build_filtered_data(
        self,
        dataset,
    ):
        return Data(
            edge_index=(
                dataset._data.target_edge_index
            ),
            edge_type=(
                dataset._data.target_edge_type
            ),
            num_nodes=dataset[0].num_nodes,
        )

    def _load_enterprises(self):
        enterprises = {}

        for enterprise_id in (
            self.run_config.client_ids
        ):
            dataset = FedRelShieldDataset(
                root=self.dataset_root,
                enterprise_id=enterprise_id,
            )

            enterprises[enterprise_id] = {
                "dataset": dataset,
                "train_data":
                    dataset[0].to(self.device),
                "valid_data":
                    dataset[1].to(self.device),
                "test_data":
                    dataset[2].to(self.device),
                "filtered_data":
                    self._build_filtered_data(
                        dataset
                    ).to(self.device),
            }

        return enterprises

    def _load_pretrained_model(self):
        model = self._build_seeded_model()

        checkpoint_path = os.path.expanduser(
            self.cfg.checkpoint
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        if "model" not in checkpoint:
            raise RuntimeError(
                "Pretrained checkpoint is missing "
                "the model state"
            )

        model_state.load_model_state(
            model,
            checkpoint["model"],
        )

        return model

    def _evaluate_single_enterprise(
        self,
        model,
        enterprise_id,
    ):
        enterprise = self.enterprises[
            enterprise_id
        ]

        split_name = (
            self.run_config.evaluation_split
        )

        eval_data = enterprise[
            f"{split_name}_data"
        ]

        metrics = evaluation.evaluate(
            cfg=self.cfg,
            model=model,
            eval_data=eval_data,
            filtered_data=(
                enterprise["filtered_data"]
            ),
            device=self.device,
            logger=None,
            return_metrics=True,
        )

        return EnterpriseEvaluationResult(
            enterprise_id=enterprise_id,
            num_examples=int(
                eval_data
                .target_edge_index
                .shape[1]
            ),
            metrics=metrics,
        )

    def _evaluate_model(self, model):
        enterprise_results = []

        for enterprise_id in (
            self.run_config.client_ids
        ):
            enterprise_results.append(
                self._evaluate_single_enterprise(
                    model=model,
                    enterprise_id=enterprise_id,
                )
            )

        return aggregate_enterprise_metrics(
            enterprise_results
        )

    def _train_model(
        self,
        model,
        train_data,
        local_epochs,
        batch_per_epoch,
    ):
        train_loader, sampler = (
            training.build_train_loader(
                train_data=train_data,
                batch_size=self.cfg.train.batch_size,
                world_size=1,
                rank=0,
            )
        )

        optimizer = training.build_optimizer(
            self.cfg,
            model,
        )

        epoch_losses = []

        for local_epoch in range(local_epochs):
            sampler.set_epoch(local_epoch)

            average_loss = training.train_one_epoch(
                cfg=self.cfg,
                model=model,
                train_data=train_data,
                train_loader=train_loader,
                optimizer=optimizer,
                batch_per_epoch=batch_per_epoch,
            )

            epoch_losses.append(
                float(average_loss)
            )

        return epoch_losses

    def _train_local_model(
        self,
        enterprise_id,
    ):
        enterprise = self.enterprises[
            enterprise_id
        ]

        model = self._load_pretrained_model()

        train_data = enterprise[
            "train_data"
        ]

        epoch_losses = self._train_model(
            model=model,
            train_data=train_data,
            local_epochs=(
                self.cfg
                .local_training
                .local_epochs
            ),
            batch_per_epoch=(
                self.cfg
                .local_training
                .batch_per_epoch
            ),
        )

        return (
            model,
            sum(epoch_losses) / len(epoch_losses),
        )

    def _build_centralized_train_data(self):
        edge_indices = []
        edge_types = []
        target_edge_indices = []
        target_edge_types = []
    
        client_num_examples = {}
    
        node_offset = 0
    
        for enterprise_id in (
            self.run_config.client_ids
        ):
            train_data = self.enterprises[
                enterprise_id
            ]["train_data"]
    
            edge_indices.append(
                train_data.edge_index
                + node_offset
            )
    
            edge_types.append(
                train_data.edge_type
            )
    
            target_edge_indices.append(
                train_data.target_edge_index
                + node_offset
            )
    
            target_edge_types.append(
                train_data.target_edge_type
            )
    
            client_num_examples[
                enterprise_id
            ] = int(
                train_data
                .target_edge_index
                .shape[1]
            )
    
            node_offset += int(
                train_data.num_nodes
            )
    
        centralized_data = Data(
            edge_index=torch.cat(
                edge_indices,
                dim=1,
            ),
            edge_type=torch.cat(
                edge_types,
                dim=0,
            ),
            target_edge_index=torch.cat(
                target_edge_indices,
                dim=1,
            ),
            target_edge_type=torch.cat(
                target_edge_types,
                dim=0,
            ),
            num_nodes=node_offset,
        )
    
        centralized_data.num_relations = int(
            centralized_data.edge_type.max().item()
        ) + 1
    
        # IMPORTANT:
        # build_relation_graph mutates centralized_data in place.
        # Do not assign its return value to relation_graph.
        tasks.build_relation_graph(
            centralized_data
        )
    
        if not hasattr(
            centralized_data,
            "relation_graph",
        ):
            raise RuntimeError(
                "Centralized data is missing "
                "relation_graph after processing"
            )
    
        return (
            centralized_data.to(self.device),
            client_num_examples,
        )

    def _run_zero_shot(self):
        model = self._load_pretrained_model()

        evaluation_result = (
            self._evaluate_model(model)
        )

        return {
            "format_version": "1.0",
            "baseline": "zero_shot",
            "seed": int(
                self.run_config.seed
            ),
            "evaluation_split":
                self.run_config.evaluation_split,
            "clients": list(
                self.run_config.client_ids
            ),
            "evaluation":
                evaluation_result_to_dict(
                    evaluation_result
                ),
        }

    def _run_local_only(self):
        enterprise_results = []
        local_training = {}

        for enterprise_id in (
            self.run_config.client_ids
        ):
            model, average_loss = (
                self._train_local_model(
                    enterprise_id
                )
            )

            enterprise_results.append(
                self._evaluate_single_enterprise(
                    model=model,
                    enterprise_id=enterprise_id,
                )
            )

            local_training[
                enterprise_id
            ] = {
                "num_examples": int(
                    self.enterprises[
                        enterprise_id
                    ]["train_data"]
                    .target_edge_index
                    .shape[1]
                ),
                "average_loss": float(
                    average_loss
                ),
            }

        evaluation_result = (
            aggregate_enterprise_metrics(
                enterprise_results
            )
        )

        return {
            "format_version": "1.0",
            "baseline": "local_only",
            "seed": int(
                self.run_config.seed
            ),
            "evaluation_split":
                self.run_config.evaluation_split,
            "clients": list(
                self.run_config.client_ids
            ),
            "local_training": local_training,
            "evaluation":
                evaluation_result_to_dict(
                    evaluation_result
                ),
        }

    def _run_centralized(self):
        model = self._load_pretrained_model()

        (
            centralized_data,
            client_num_examples,
        ) = self._build_centralized_train_data()

        epoch_losses = self._train_model(
            model=model,
            train_data=centralized_data,
            local_epochs=(
                self.cfg
                .centralized_training
                .local_epochs
            ),
            batch_per_epoch=(
                self.cfg
                .centralized_training
                .batch_per_epoch
            ),
        )

        evaluation_result = (
            self._evaluate_model(model)
        )

        total_examples = sum(
            client_num_examples.values()
        )

        return {
            "format_version": "1.0",
            "baseline": "centralized",
            "seed": int(
                self.run_config.seed
            ),
            "evaluation_split":
                self.run_config.evaluation_split,
            "clients": list(
                self.run_config.client_ids
            ),
            "centralized_training": {
                "num_examples": int(
                    total_examples
                ),
                "client_num_examples":
                    client_num_examples,
                "epoch_losses": epoch_losses,
                "average_loss": float(
                    sum(epoch_losses)
                    / len(epoch_losses)
                ),
            },
            "evaluation":
                evaluation_result_to_dict(
                    evaluation_result
                ),
        }
